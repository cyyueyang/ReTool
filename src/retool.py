import logging
import re
import glob
import numpy as np
import os
from typing import List, Dict, Tuple, Any

import datasets
from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.schemas import ToolResponse
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.rollout_trace import rollout_trace_op
from retool_dataset_utils import map_fn, map_fn2

logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config=config, tool_schema=tool_schema)

        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        lines = code.split("\n")
        # 防止代码无输出，手动添加输出
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue

            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
                break

        code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)

        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)

        return result, None, None


class CustomRLHFDataset(RLHFDataset):
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # 根据数据集名称决定怎么加载
            if "gsm8k" in parquet_file:
                dataset_dict = datasets.load_dataset(parquet_file, "main")
            else:
                dataset_dict = datasets.load_dataset(parquet_file)

            # 选择可用的 split
            if 'train' in dataset_dict:
                dataframe = dataset_dict['train']
            elif 'test' in dataset_dict:
                dataframe = dataset_dict['test']
            else:
                first_split = list(dataset_dict.keys())[0]
                dataframe = dataset_dict[first_split]

            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025", "HuggingFaceH4/MATH-500",
                               "openai/gsm8k"]:
                dataframe = dataframe.map(map_fn, fn_kwargs={"data_source": data_source},
                                          remove_columns=dataframe.column_names)
            else:
                dataframe = dataframe.map(map_fn2, num_proc=16)

            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            rng = np.random.default_rng(self.seed if self.seed is not None else ())
            indices = rng.choice(total, size=self.max_samples, replace=False)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        # self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    num_turns = extra_info["num_turns"]

    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, tool_call_reward + result["score"])

    if result["pred"] is None:
        result["pred"] = ""

    return result