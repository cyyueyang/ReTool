import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import OrderedDict

# 必须导入，否则 DTensor 无法被 pickle 识别
import torch.distributed.tensor
from torch.distributed._tensor import DTensor

base_path = "/data/cyyang/WorkSpace/ReTool/checkpoint/Qwen_SFT/global_step_2000"
hf_path = os.path.join(base_path, "huggingface")

# 1. 合并所有 rank 的 state dict
full_state_dict = OrderedDict()
for i in range(4):
    rank_file = os.path.join(base_path, f"model_world_size_4_rank_{i}.pt")
    print(f"Loading {rank_file}...")

    sd = torch.load(rank_file, map_location="cpu", weights_only=False)

    # 处理可能的嵌套结构
    if isinstance(sd, dict):
        if 'model' in sd and isinstance(sd['model'], dict):
            sd = sd['model']
        elif 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']

    for k, v in sd.items():
        if k not in full_state_dict:
            full_state_dict[k] = v

print(f"Total keys collected: {len(full_state_dict)}")

# 2. 用 config 创建空模型架构（不从 huggingface 目录加载权重）
print("Creating model from config...")
config = AutoConfig.from_pretrained(hf_path, trust_remote_code=False)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

# 3. 加载合并后的权重
print("Loading weights...")
missing, unexpected = model.load_state_dict(full_state_dict, strict=False)
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

if missing:
    print("Warning: Some keys are missing.")
    print(missing[:10])
else:
    print("All weights loaded successfully!")

print(f"Saving HF model to {base_path}...")
model.save_pretrained(base_path)
tokenizer = AutoTokenizer.from_pretrained(hf_path)
tokenizer.save_pretrained(base_path)
print("Done!")