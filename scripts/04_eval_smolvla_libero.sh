#!/usr/bin/env bash
# Avalia o SmolVLA no LIBERO (10 tarefas x 10 episódios por suíte).
# Uso: bash scripts/04_eval_smolvla_libero.sh [n_action_steps] [suite] [run]
#   n_action_steps: 10 (padrão, replaneja mais vezes) ou 50 (bloco inteiro)
#   suite: libero_spatial (padrão), ou p.ex.
#          libero_spatial,libero_object,libero_goal,libero_10
#   run:   smolvla_libero_v2 (padrão)
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

NAS=${1:-10}
SUITE=${2:-libero_spatial}
RUN=${3:-smolvla_libero_v2}

C="outputs/train/$RUN/checkpoints/last/pretrained_model"
P="--policy.path=$C --policy.device=cuda --policy.n_action_steps=$NAS"
E="--env.type=libero --env.task=$SUITE --env.max_parallel_tasks=1"
V="--eval.n_episodes=10 --eval.batch_size=1"
# Mesmo rename_map do treino
R1='"observation.images.image":"observation.images.camera1"'
R2='"observation.images.image2":"observation.images.camera2"'
R="--rename_map={$R1,$R2}"

lerobot-eval $P $E $V "$R"
