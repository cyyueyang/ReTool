from typing import List, Dict, Any

def extract_gsm8k_answer(answer_str: str) -> str:
    """Extract the final numeric answer after '####' in GSM8K format."""
    parts = answer_str.strip().split("####")
    return parts[-1].strip().replace(",", "")

def map_fn(row: Dict[str, Any], data_source: str = None):
    answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""
    if data_source == "Maxwell-Jia/AIME_2024":
        problem, answer = row["Problem"], row["Answer"]
    elif data_source == "yentinglin/aime_2025":
        problem, answer = row["problem"], row["answer"]
    elif data_source == "HuggingFaceH4/MATH-500":
        problem, answer = row["problem"], row["answer"]
    elif data_source == "openai/gsm8k":
        problem, answer = row["question"], extract_gsm8k_answer(row["answer"])

    prompt = problem + answer_format

    data = {
        "data_source": data_source.split("/")[-1].lower(),
        "prompt": [{
            "role": "user",
            "content": prompt,
        }],
        "ability": "MATH",
        "reward_model": {"ground_truth": str(answer)},
        "agent_name": "tool_agent",
    }

    return data

def map_fn2(row: Dict[str, Any]):
    answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""
    content = row["prompt"][0]["content"]
    row["prompt"][0]["content"] = content + answer_format
    row["agent_name"] = "tool_agent"
    return row
