#!/usr/bin/env bash
# Cria o ambiente conda `vla` e instala o LeRobot com os extras usados no projeto.
# Uso: bash scripts/00_setup.sh
set -e
cd "$(dirname "$0")/.."

LEROBOT_COMMIT=73ab890324b303424a8a5f07272f22c36be5d269

if [ ! -d lerobot ]; then
  git clone https://github.com/huggingface/lerobot.git
fi
git -C lerobot checkout "$LEROBOT_COMMIT"

# Python 3.12: no 3.14 o pygame (dependência do gym-pusht) não tem wheel
conda create -y -n vla python=3.12
eval "$(conda shell.bash hook)"
conda activate vla

pip install -e "./lerobot[training,smolvla,aloha,pusht,libero]"

# ffmpeg dentro do ambiente (necessário para decodificar os vídeos dos datasets)
conda install -y -c conda-forge "ffmpeg>=7,<8"

# Na primeira importação o LIBERO pergunta onde salvar a config; responde "N" (padrão)
echo N | python -c "import libero.libero" || true

python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
echo "Pronto. Rode: conda activate vla && export MUJOCO_GL=egl"
