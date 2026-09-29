#!/usr/bin/env bash
# Avalia o ACT treinado no ALOHA (50 episódios).
# Uso: bash scripts/02_eval_act_aloha.sh [checkpoint]   (padrão: last)
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

CK=${1:-last}
C="outputs/train/act_transfer_cube/checkpoints/$CK/pretrained_model"
P="--policy.path=$C --policy.device=cuda"
E="--env.type=aloha --env.task=AlohaTransferCube-v0"
V="--eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false"

lerobot-eval $P $E $V
