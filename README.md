# vla-lerobot

First experiments with **Vision-Language-Action (VLA)** models using [LeRobot](https://github.com/huggingface/lerobot), running on a single GPU (RTX 4090, 24 GB).

The project compares two approaches in simulation:

- **ACT** (no language) on the ALOHA bimanual *Transfer Cube* task
- **SmolVLA** (~450M-parameter VLA) on the **LIBERO** benchmark, where a single model performs different tasks from text instructions

## Results

### ACT — ALOHA Transfer Cube

| Model | Steps | Batch | Success (50 episodes) |
|---|---|---|---|
| ACT | 100k | 8 | **62%** |

Most failures happen during the hand-over: the robot lifts the cube but does not complete the transfer to the other arm.

### SmolVLA — LIBERO Spatial (10 tasks × 10 episodes)

| Model | Steps | Batch | `n_action_steps` | Success |
|---|---|---|---|---|
| v1 | 20k | 32 | 50 | 53% |
| v2 | 60k | 64 | 50 | 66% |
| v2 | 60k | 64 | **10** | **84%** |

Per task (v2, `n_action_steps=10`):

| # | Instruction | Success |
|---|---|---|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 100% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 90% |
| 2 | pick up the black bowl from table center and place it on the plate | 90% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 90% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 80% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | **30%** |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 100% |
| 7 | pick up the black bowl on the stove and place it on the plate | 100% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 80% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 80% |

**Takeaways:**

1. **Longer training helped** (53% → 66%). In v1 the learning rate had already decayed to its minimum by step 20k; in v2 the decay was stretched to 60k (`scheduler_decay_steps`).
2. **Replanning more often helped even more** (66% → 84%, no retraining). With `n_action_steps=10`, the model executes 10 of its 50 predicted actions and then looks at the cameras again, correcting mistakes along the way.
3. **Task 5** (bowl stacked on top of the ramekin) remains the weak spot.

Full results for each evaluation are in [`results/`](results/).

## Reproducing

Requirements: Linux, NVIDIA GPU (tested on an RTX 4090; SmolVLA with batch 64 uses ~15 GB), conda.

```bash
bash scripts/00_setup.sh              # conda env `vla` + LeRobot pinned to a commit
conda activate vla
export MUJOCO_GL=egl                  # headless rendering (server over SSH)
wandb login                           # optional; or set WANDB=false for the scripts

bash scripts/01_train_act_aloha.sh    # ~1 h
bash scripts/02_eval_act_aloha.sh

bash scripts/03_train_smolvla_libero.sh        # ~9-10 h
bash scripts/04_eval_smolvla_libero.sh 10      # n_action_steps=10
bash scripts/04_eval_smolvla_libero.sh 50      # n_action_steps=50
```

For long runs, use `tmux` so the process survives an SSH disconnect.

Useful variables: `WANDB=false` disables wandb; `BATCH=32` lowers the SmolVLA batch size if you run out of memory.

## Issues we hit (and fixes)

| Issue | Cause | Fix |
|---|---|---|
| `pip install` fails building `pygame` | No pygame wheel for Python 3.14 | Use Python 3.12 |
| `'repo_id' argument missing` | `push_to_hub` is enabled by default | `--policy.push_to_hub=false` |
| `No module named wandb` | wandb lives in the `training` extra | Install `lerobot[training]` |
| `Namespace gym_aloha not found` during eval | Parallel envs (`forkserver`) don't import `gym_aloha` | `--eval.use_async_envs=false` |
| `EOFError` when importing LIBERO | First import asks an interactive question | `echo N \| python -c "import libero.libero"` |
| Camera key error when training SmolVLA | `smolvla_base` expects `camera1..3`, LIBERO provides `image`/`image2` | `--rename_map` (for training **and** evaluation) |
| `lerobot-dataset-viz` hangs over SSH | It tries to open a GUI window | `--mode distant` + SSH tunnel |

## Layout

```
scripts/   setup, training and evaluation commands
results/   eval_info.json for each evaluation (per-task and per-episode metrics)
lerobot/   (ignored) LeRobot clone created by the setup script
outputs/   (ignored) checkpoints, videos and logs
```

## Next steps

- Evaluate on all 4 LIBERO suites (`spatial`, `object`, `goal`, `10`)
- Test robustness with LIBERO-plus (camera, lighting and instruction variations)
- Try temporal ensembling for ACT
- Collect data and train on a real arm (SO-101)
