#!/usr/bin/env bash
# Fine-tuning do SmolVLA (lerobot/smolvla_base) no dataset LIBERO.
# Configuração v2: 60k passos, batch 64. ~9-10 h numa RTX 4090, ~15 GB de VRAM.
# Uso: bash scripts/03_train_smolvla_libero.sh
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

P="--policy.path=lerobot/smolvla_base --policy.device=cuda"
# decay_steps = steps, para a LR cair ao longo de todo o treino
H="--policy.push_to_hub=false --policy.scheduler_decay_steps=60000"
D="--dataset.repo_id=lerobot/libero"
# O smolvla_base espera câmeras camera1..3; o LIBERO usa image/image2
R1='"observation.images.image":"observation.images.camera1"'
R2='"observation.images.image2":"observation.images.camera2"'
R="--rename_map={$R1,$R2}"
T="--batch_size=${BATCH:-64} --steps=60000 --save_freq=10000 --log_freq=50"
O="--output_dir=outputs/train/smolvla_libero_v2 --job_name=smolvla_v2"
W="--wandb.enable=${WANDB:-true}"

lerobot-train $P $H $D "$R" $T $O $W
