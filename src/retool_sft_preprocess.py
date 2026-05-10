import re
import json
from typing import Dict, Any, List, Tuple
import os
import datasets
from omegaconf import OmegaConf

code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

def extract_code_message(content: str) -> Tuple[Dict[str, Any], str]:
    # Dict[str, Any] 是返回处理好的，str 是剩下该处理的
    start, stop = "<code>", "</code>"

    i = content.find(start)

    if i == -1:
        return None, content

    j = content.find(stop, i)
    if j == -1:
        raise RuntimeError

    code = content[i + len(start): j]
    matches = code_pattern.findall(code)
    if matches:
        code = matches[0].strip()

    message = {
        "role": "assistant",
        "content": content[: i].strip(),
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "arguments": {
                        "code": code,
                    },
                }
            }
        ]
    }
    return message, content[j + len(stop):]

def extract_interpreter_message(content: str) -> Tuple[Dict[str, Any], str]:
    start, stop = "<interpreter>", "</interpreter>"

    i = content.find(start)
    if i == -1:
        return None, content

    j = content.find(stop, i)
    if j == -1:
        raise RuntimeError

    interpreter = content[i + len(start): j]
    message = {
        "role": "tool",
        "content": interpreter.strip(),
    }
    return message, content[j + len(stop):]

def extract_answer_message(content: str) -> Tuple[Dict[str, Any], str]:
    start, stop = "<answer>", "</answer>"

    i = content.find(start)
    if i == -1:
        return None, content

    j = content.find(stop, i)
    if j == -1:
        raise RuntimeError

    answer = content[: i] + content[i + len(start): j]
    message = {
        "role": "assistant",
        "content": answer.strip(),
    }

    return message, content[j + len(stop): ]

def process(row: Dict[str, Any], tools: str):
    messages = []

    # 处理 user input
    content = row["messages"][0]["content"]

    start = "*user question:*"
    i = content.find(start)
    if i == -1:
        raise RuntimeError
    prompt = content[i + len(start):].replace("<answer>", "").replace("</answer>", "")

    messages.append({
        "role": "user",
        "content": prompt.strip(),
    })

    # 处理 assistant
    content = row["messages"][1]["content"]
    role = "assistant"
    while len(content) > 0:
        """
        截取 assistant 内容 直到 code, 无code默认输出答案， 
        截取到 code, role 转为 tool
        """
        if role == "assistant":
            message, content = extract_code_message(content)
            if message is None:
                message, content = extract_answer_message(content)
            assert message is not None

            messages.append(message)
            role = "tool"

        else:
            message, content = extract_interpreter_message(content)
            assert message is not None

            messages.append(message)
            role = "assistant"

    tools = json.loads(tools)

    return {"messages": messages, "tools": tools}

if __name__ == "__main__":
    tools_config_file = "sandbox_fusion_tool_config.yaml"

    tools_config = OmegaConf.load(tools_config_file)
    tool_schema = OmegaConf.to_container(tools_config["tools"][0]["tool_schema"])
    tools = json.dumps([tool_schema])

    data = datasets.load_dataset("/root/autodl-tmp/ReTool/data/JoeYing/ReTool-SFT")["train"]
    data = data.map(process, fn_kwargs={"tools": tools})
    sava_path = r"/root/autodl-tmp/ReTool/data/ReTool-SFT-Preprocess/train-00000-of-00001.parquet"
    data.to_parquet(sava_path)




