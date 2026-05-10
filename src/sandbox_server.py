#!/usr/bin/env python3
"""
sandbox_server.py
verl SandboxFusion 无 Docker 替代服务
功能：并发代码沙箱 / 资源硬限制 / 系统调用过滤 / 超时熔断
"""

import asyncio
import os
import sys
import time
import signal
import tempfile
import resource
import shutil
import traceback
import logging
from typing import Optional, Literal, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator
import uvicorn

# ==================== 可选依赖 ====================
try:
    import seccomp
    HAS_SECCOMP = True
except ImportError:
    HAS_SECCOMP = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ==================== 配置 ====================
MAX_CONCURRENT = int(os.getenv("SANDBOX_MAX_CONCURRENT", "16"))
DEFAULT_TIMEOUT = int(os.getenv("SANDBOX_DEFAULT_TIMEOUT", "10"))
DEFAULT_MEMORY_MB = int(os.getenv("SANDBOX_DEFAULT_MEMORY_MB", "256"))
DEFAULT_CPU_MS = int(os.getenv("SANDBOX_DEFAULT_CPU_MS", "5000"))
ENABLE_SECCOMP = os.getenv("SANDBOX_ENABLE_SECCOMP", "1") == "1" and HAS_SECCOMP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("sandbox")

# 全局并发信号量：防止同时跑太多进程把宿主机拖死
job_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# ==================== 数据模型 ====================
class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str = Field(..., description="待执行代码")
    language: Literal["python", "bash"] = Field(default="python")
    # 客户端发 run_timeout / compile_timeout，合并成实际执行超时
    run_timeout: int = Field(default=DEFAULT_TIMEOUT, ge=1, le=300)
    compile_timeout: int = Field(default=DEFAULT_TIMEOUT, ge=1, le=300)
    timeout: int = Field(default=0, description="内部用：取 run_timeout 和 compile_timeout 的较大值")
    memory_limit_mb: int = Field(default=DEFAULT_MEMORY_MB, ge=32, le=8192, validation_alias="memory_limit_MB")
    cpu_limit_ms: int = Field(default=DEFAULT_CPU_MS, ge=100, le=60000)
    stdin: Optional[str] = Field(default="", description="标准输入")
    args: list[str] = Field(default_factory=list, description="额外命令行参数")

    @model_validator(mode="after")
    def resolve_defaults(self) -> "ExecuteRequest":
        self.timeout = max(self.run_timeout, self.compile_timeout)
        if self.stdin is None:
            self.stdin = ""
        return self


class ExecuteResponse(BaseModel):
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time_ms: int = 0
    memory_used_mb: int = 0
    is_timeout: bool = False
    is_oom: bool = False
    is_killed: bool = False


# ==================== 沙箱核心：preexec_fn ====================
def _setup_sandbox_limits(memory_limit_mb: int, cpu_limit_ms: int) -> None:
    """
    在子进程 fork 后、exec 前执行。
    此处已经是子进程内部，任何操作只影响即将运行的用户代码。
    """
    # 创建新进程组，方便后续整组 kill
    os.setpgrp()

    # 1. 内存限制（虚拟地址空间，硬限制）
    max_mem = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))

    # 2. CPU 时间限制（秒）
    cpu_sec = max(1, cpu_limit_ms // 1000)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec + 1))

    # 3. 禁止核心转储
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # 4. 限制 fork 炸弹（该 UID 下进程数）
    # 注意：如果服务以独立系统用户运行，此限制是安全的；
    # 如果以普通用户运行且该用户有其他进程，请改为 (0,0) 完全禁止 fork
    resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))

    # 5. 限制输出文件大小 10MB
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))

    # 6. 栈空间 8MB
    resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, 8 * 1024 * 1024))

    # 7. 禁止获取新权限（防止利用 setuid 二进制逃逸）
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        PR_SET_NO_NEW_PRIVS = 38
        libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    except Exception:
        pass

    # 8. seccomp 系统调用过滤（如果启用）
    if ENABLE_SECCOMP and HAS_SECCOMP:
        try:
            f = seccomp.SyscallFilter(seccomp.ALLOW)
            # 禁止执行其他程序
            f.add_rule(seccomp.KILL, "execve")
            f.add_rule(seccomp.KILL, "execveat")
            # 禁止网络
            f.add_rule(seccomp.KILL, "socket")
            f.add_rule(seccomp.KILL, "connect")
            f.add_rule(seccomp.KILL, "accept")
            f.add_rule(seccomp.KILL, "bind")
            # 禁止 ptrace
            f.add_rule(seccomp.KILL, "ptrace")
            # 禁止挂载/卸载/切换根目录
            f.add_rule(seccomp.KILL, "mount")
            f.add_rule(seccomp.KILL, "umount2")
            f.add_rule(seccomp.KILL, "pivot_root")
            f.add_rule(seccomp.KILL, "chroot")
            # 禁止创建新命名空间（防止容器逃逸）
            f.add_rule(seccomp.KILL, "unshare")
            f.add_rule(seccomp.KILL, "setns")
            f.load()
        except Exception as e:
            # seccomp 加载失败不应静默，但这里记录到 stderr 会被父进程捕获
            print(f"[sandbox-warning] seccomp load failed: {e}", file=sys.stderr)


# ==================== 内存监控协程 ====================
async def _monitor_memory(proc: asyncio.subprocess.Process, limit_mb: int) -> tuple[int, bool]:
    """
    后台协程：每 50ms 采样一次进程 RSS。
    返回: (峰值MB, 是否触发了kill)
    """
    peak_mb = 0
    killed = False

    try:
        while proc.returncode is None:
            current_mb = 0
            try:
                if HAS_PSUTIL:
                    p = psutil.Process(proc.pid)
                    current_mb = p.memory_info().rss // (1024 * 1024)
                else:
                    # 无 psutil 时读 /proc
                    with open(f"/proc/{proc.pid}/status", "r") as f:
                        for line in f:
                            if line.startswith("VmRSS:"):
                                current_mb = int(line.split()[1]) // 1024
                                break
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                break

            peak_mb = max(peak_mb, current_mb)

            # 硬熔断：超过内存限制直接 kill
            if current_mb > limit_mb:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                killed = True
                break

            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        pass

    return peak_mb, killed


# ==================== 核心执行逻辑 ====================
async def execute_code(req: ExecuteRequest) -> ExecuteResponse:
    async with job_semaphore:
        work_dir = tempfile.mkdtemp(prefix="sandbox_")
        start_time = time.time()
        proc: Optional[asyncio.subprocess.Process] = None
        monitor_task: Optional[asyncio.Task] = None

        try:
            # 构造命令
            if req.language == "python":
                # -I: 隔离模式（不加载用户 site-packages，不设置 sys.path[0]）
                # -S: 不导入 site 模块（可选，如果不需要 site-packages 可以加上）
                cmd = [sys.executable, "-I", "-c", req.code]
            elif req.language == "bash":
                cmd = ["/bin/bash", "-c", req.code]
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {req.language}")

            cmd.extend(req.args)

            # 最小化环境变量，防止代码读取宿主机敏感配置
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": work_dir,
                "PWD": work_dir,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONHASHSEED": "0",
                "TMPDIR": work_dir,
                "TEMP": work_dir,
                "TMP": work_dir,
                "LC_ALL": "C.UTF-8",
                "LANG": "C.UTF-8",
            }

            # 启动子进程（preexec_fn 在 fork 后执行资源限制）
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if req.stdin else None,
                cwd=work_dir,
                env=env,
                preexec_fn=lambda: _setup_sandbox_limits(req.memory_limit_mb, req.cpu_limit_ms),
            )

            # 启动内存监控后台协程
            monitor_task = asyncio.create_task(
                _monitor_memory(proc, req.memory_limit_mb)
            )

            # 执行并等待（带超时）
            try:
                if req.stdin:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=req.stdin.encode()),
                        timeout=req.timeout
                    )
                else:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=req.timeout
                    )

                # 取消监控协程并获取峰值
                monitor_task.cancel()
                peak_mb, was_killed = 0, False
                try:
                    peak_mb, was_killed = await monitor_task
                except asyncio.CancelledError:
                    pass

                exec_time_ms = int((time.time() - start_time) * 1000)

                return ExecuteResponse(
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    return_code=proc.returncode if proc.returncode is not None else -1,
                    execution_time_ms=exec_time_ms,
                    memory_used_mb=peak_mb,
                    is_timeout=False,
                    is_oom=was_killed,
                    is_killed=was_killed,
                )

            except asyncio.TimeoutError:
                # 超时：整进程组 SIGKILL，防止用户代码的子进程（如 sleep 1000）泄漏
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()

                if monitor_task:
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass

                return ExecuteResponse(
                    stdout="",
                    stderr=f"Execution timed out after {req.timeout} seconds",
                    return_code=-signal.SIGKILL,
                    execution_time_ms=req.timeout * 1000,
                    is_timeout=True,
                    is_oom=False,
                    is_killed=True,
                )

        except HTTPException:
            raise
        except Exception as e:
            if proc and proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            logger.error(f"Internal error: {traceback.format_exc()}")
            return ExecuteResponse(
                stdout="",
                stderr=f"Sandbox internal error: {str(e)}",
                return_code=-1,
                execution_time_ms=int((time.time() - start_time) * 1000),
                is_killed=True,
            )
        finally:
            # 清理临时工作目录
            shutil.rmtree(work_dir, ignore_errors=True)


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="Verl Sandbox Server",
    description="无 Docker 原生并发代码沙箱，兼容 verl SandboxFusionTool",
    version="1.1.0",
)

@app.post("/run_code")
async def run_code(req: ExecuteRequest):
    """
    主端点，返回兼容 verl SandboxFusionTool 的嵌套格式。
    客户端期望: {"status": "Success/Failed/SandboxError",
                 "compile_result": {...}, "run_result": {...}}
    """
    resp = await execute_code(req)

    # 构造 SandboxFusion 兼容的嵌套响应
    run_status = "Finished"
    top_status = "Success"

    if resp.is_timeout:
        run_status = "TimeLimitExceeded"
        top_status = "Failed"
    elif resp.is_oom:
        run_status = "Error"
        top_status = "Failed"
    elif resp.is_killed:
        run_status = "Error"
        top_status = "Failed"
    elif resp.return_code != 0:
        run_status = "Finished"  # 正常退出但非零返回值，仍然 Finished
        top_status = "Failed"

    # Python 不需要编译阶段，compile_result 假装成功
    compile_result = {
        "status": "Finished",
        "return_code": 0,
        "stderr": "",
        "execution_time": 0,
    }

    run_result = {
        "status": run_status,
        "return_code": resp.return_code,
        "stdout": resp.stdout,
        "stderr": resp.stderr,
        "execution_time": resp.execution_time_ms / 1000.0,
    }

    return {
        "status": top_status,
        "compile_result": compile_result,
        "run_result": run_result,
    }

@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest):
    """备用端点"""
    return await execute_code(req)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "max_concurrent": MAX_CONCURRENT,
        "seccomp_enabled": ENABLE_SECCOMP,
        "psutil_enabled": HAS_PSUTIL,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.on_event("startup")
async def startup():
    logger.info(f"Sandbox Server started")
    logger.info(f"  Max concurrent: {MAX_CONCURRENT}")
    logger.info(f"  Seccomp: {'ON' if ENABLE_SECCOMP else 'OFF (pip install seccomp)'}")
    logger.info(f"  Psutil:  {'ON' if HAS_PSUTIL else 'OFF (pip install psutil)'}")
    logger.info(f"  Default limits: {DEFAULT_MEMORY_MB}MB / {DEFAULT_TIMEOUT}s")


if __name__ == "__main__":
    # 生产环境建议使用:
    # gunicorn sandbox_server:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)