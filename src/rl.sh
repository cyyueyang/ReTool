set -x

# export VLLM_USE_V1=1

# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_IB_DISABLE=1

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=/root/autodl-tmp/ReTool:$PYTHONPATH

# ================= data/model/tool =================

dapo_math_17k=/root/autodl-tmp/ReTool/data/BytedTsinghua-SIA/DAPO-Math-17k
aime_2024=/root/autodl-tmp/ReTool/data/Maxwell-Jia/AIME_2024
aime_2025=/root/autodl-tmp/ReTool/data/yentinglin/aime_2025
model_path=/root/autodl-tmp/ReTool/checkpoint/Qwen-3B-SFT-V1/global_step_1500/huggingface

train_files="['$dapo_math_17k']"
test_files="['$aime_2025', '$aime_2024']"

# tool
tool_config_path=/root/autodl-tmp/ReTool/src/sandbox_fusion_tool_config.yaml

# swanlab
project_name=retool
experiment_name=qwen2.5-3B-dapo-v1
default_local_dir=/root/autodl-tmp/ReTool/checkpoint/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.38

max_turns=16
max_prompt_length=2048
max_response_length=16384
actor_lr=1e-6

train_batch_size=4
ppo_mini_batch_size=2
n_resp_per_prompt=8
n_resp_per_prompt_val=8

# 数据集采样
train_max_samples=4000
# val_max_samples=60
# data.val_max_samples=$val_max_samples \
# ================= perfomance =================
infer_tp=1 # vllm
train_sp=1 # train: SP=1, 4GPU全做FSDP DP, 每卡只分1/4数据
offload=True

actor_max_token_len_per_gpu=$(( max_prompt_length + max_response_length ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.train_max_samples=$train_max_samples \
    data.custom_cls.path=/root/autodl-tmp/ReTool/src/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=/root/autodl-tmp/ReTool/src/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=True \
    trainer.log_val_generations=20 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    checkpoint.save_contents='[model,optimizer,extra,hf_model]'
