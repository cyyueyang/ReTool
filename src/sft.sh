#!/bin/bash
set -x

nproc_per_node=2
save_path=/root/autodl-tmp/ReTool/checkpoint/Qwen-3B-SFT-V1

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.sft_trainer \
    data.train_files=/root/autodl-tmp/ReTool/data/ReTool-SFT-Preprocess \
    data.val_files=/root/autodl-tmp/ReTool/data/ReTool-SFT-Preprocess \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=16 \
    data.max_length=16384 \
    data.train_batch_size=8 \
    model.path=/root/autodl-tmp/ReTool/model/Qwen/Qwen2.5-3B-Instruct \
    model.use_remove_padding=true \
    engine.ulysses_sequence_parallel_size=2 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-2.5-3b-instruct-lr1e-5-cosine \
    trainer.logger='["console","swanlab"]' \
    trainer.total_epochs=6 \
    trainer.save_freq=500 \
    optim.lr=1e-5 \
    optim.lr_scheduler_type=cosine \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.min_lr_ratio=0.01 \
    optim.clip_grad=1.0 \
    checkpoint.save_contents='[model,optimizer,extra,hf_model]'
