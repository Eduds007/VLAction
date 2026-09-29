#!/usr/bin/env bash
# Treina o ACT na simulação ALOHA (Transfer Cube). ~1 h numa RTX 4090.
# Uso: bash scripts/01_train_act_aloha.sh
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

P="--policy.type=act --policy.device=cuda --policy.push_to_hub=false"
D="--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human"
E="--env.type=aloha --env.task=AlohaTransferCube-v0"
T="--batch_size=8 --steps=100000 --env_eval_freq=20000 --save_freq=20000"
# use_async_envs=false: com processos paralelos o gym_aloha não é importado
V="--eval.n_episodes=10 --eval.batch_size=10 --eval.use_async_envs=false"
O="--output_dir=outputs/train/act_transfer_cube --job_name=act_transfer_cube"
W="--wandb.enable=${WANDB:-true}"

lerobot-train $P $D $E $T $V $O $W
