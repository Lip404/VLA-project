# VLA-project

## VLA-Agent: Vision-Language-Action Embodied Agent

> Embodied AI agent for vision-language robot manipulation & long-horizon task planning, with supervised fine-tuning on subtask-annotated RoboCasa episode data.

## Project Overview

This repository builds a hierarchical embodied agent that follows natural language instructions for robot manipulation in the RoboCasa simulation environment.
We split the long-horizon task solving pipeline into two parts (System1 / System2 paradigm):

- **System2 (Hierarchical Planner)**: Given a high-level task instruction and visual video observation, decompose the long task into an ordered sequence of executable subtasks. Implemented using LoRA-finetuned Qwen2-VL.
- **System1 (Low-level Action Executor)**: Take subtask planning results + visual input to predict robot low-level control actions. We plan to use RoboCasa built-in heuristic skill policies as System-1 executor.

The backbone vision-language model we use is **Qwen2-VL**, and we build an offline VLM annotation pipeline to automatically generate subtask labels for RoboCasa episodes, then use these annotated data for multimodal supervised fine-tuning.

> **Important design note**: End-to-end VLA models struggle with long-horizon household manipulation tasks. Inspired by human cognitive process, we decouple high-level task decomposition (System-2) from low-level motor execution (System-1), instead of pursuing monolithic end-to-end policy.

## Key Features

- Qwen2-VL multimodal backbone: multi-frame video observation input, standardized inference wrapper with optional LoRA adapter loading
- Offline subtask annotation pipeline: automatically generate ordered subtask list for each RoboCasa episode
- Sliding-window PyTorch Dataset for long video episode sampling (`robocasa_dataset.py`)
- Data export, split & validation tools for RoboCasa simulation episodes
- Structured output parser for planner model JSON outputs
- LoRA supervised fine-tuning script for System-2 subtask planner
- **Offline rolling closed-loop replay evaluator** (`scripts/planner/rolling_closed_planner.py`): state-aware System-2 offline simulation framework
  > ⚠️ Warning: embedding-cosine-similarity-based subtask completion judge is **offline-only simulation logic**. This logic will be completely removed when connecting to real RoboCasa simulator; real subtask done flag shall come from simulator environment state, not VLM semantic inference.
- Configurable prompt template, trace log export for offline closed-loop evaluation

## Environment & Dependencies

### 1. VLA training environment (`vla_env`)

```bash
# Create conda environment
conda create -n vla_env python=3.10
conda activate vla_env
# Install dependencies
pip install torch torchvision pillow
pip install transformers datasets peft accelerate
pip install tqdm sentence-transformers scikit-learn numpy
```

You can also recreate the environment from the backup yaml:

```bash
conda env create -f vla_env_backup.yml
```

### 2. RoboCasa simulation environment (`robocasa_env`)

The simulator is installed in a separate conda environment from the training one.

```bash
conda create -n robocasa_env python=3.10
conda activate robocasa_env

# IMPORTANT: always use `python -m pip`, never bare `pip`.
# On this machine bare `pip` resolves to the system Python 3.14 and silently installs
# packages into the wrong site-packages, which makes robosuite appear as a
# namespace package (robosuite.__file__ == None) and breaks env initialization.
python -m pip install -e ./robosuite --no-deps --config-settings editable_mode=compat
python -m pip install -e ./robocasa --no-deps --config-settings editable_mode=compat

# Download ~10 GB of kitchen assets (textures / fixtures / objects)
python ./robocasa/robocasa/scripts/download_kitchen_assets.py
```

Verification — both paths must point to the **inner** package directory and must never be `None`:

```bash
python -c "import robosuite, robocasa; print(robosuite.__file__); print(robocasa.__file__)"
```

Expected output:

```
.../VLA-project/robosuite/robosuite/__init__.py
.../VLA-project/robocasa/robocasa/__init__.py
```

After this check passes, run the environment smoke test in the next section.

## RoboCasa Simulator: Environment & Observation Schema

### Creating an environment

```python
from robocasa.utils.env_utils import create_env

env = create_env(
    env_name="PickPlaceCounterToCabinet",
    robots=["PandaOmron"],
    seed=0,
)
obs = env.reset()
```

Useful `create_env` arguments: `env_name`, `robots`, `seed`, `render_onscreen`, `camera_names`, `camera_widths`, `camera_heights`, `split`.

`camera_widths` / `camera_heights` accept a single integer that applies to all cameras, or a per-camera list aligned with `camera_names`.

### Smoke test: environment reset

This is the minimal check that the simulator is fully functional. Save it as a temporary script (e.g. `/tmp/check_env.py`) and run it inside `robocasa_env`:

```python
import robocasa
from robocasa.utils.env_utils import create_env

env = create_env(
    env_name="PickPlaceCounterToCabinet",
    robots=["PandaOmron"],
    seed=0,
)

obs = env.reset()
print("obs keys:", list(obs.keys()))
print("env reset success")
```

```bash
conda activate robocasa_env
python /tmp/check_env.py
```

Expected output ends with `env reset success`. First run may take 30 s – 2 min while the 23 GB asset tree is loaded.

### Observation keys (`obs` dict)

| Category | Key | Description |
| --- | --- | --- |
| Camera | `robot0_agentview_left_image` | Left agent view RGB |
| Camera | `robot0_agentview_right_image` | Right agent view RGB |
| Camera | `robot0_eye_in_hand_image` | Wrist / eye-in-hand RGB |
| Proprio | `robot0_joint_pos` | Joint positions |
| Proprio | `robot0_eef_pos` | End-effector position |
| Proprio | `robot0_eef_quat_site` | End-effector orientation |
| Proprio | `robot0_gripper_qpos` | Gripper joint positions |
| Object | `obj` / `obj_quat` | Object pose |
| Aggregated | `robot0_proprio-state` | Concatenated proprio vector |
| Aggregated | `object-state` | Concatenated object vector |

Image arrays are `(height, width, 3)`, `dtype=uint8`, and are **vertically flipped** by MuJoCo — reverse the first axis (`img[::-1]`) before saving or feeding into a VLM.

### Render sanity check

`scripts/debug/check_render.py` resets the environment, saves the three camera views as PNG files and prints the mean pixel value of each image, so you can tell at a glance whether rendering is black.

```bash
conda activate robocasa_env
```

```bash
mkdir -p outputs/render_check
```

```bash
cd outputs/render_check
python ../../scripts/debug/check_render.py
```

Run it from a directory you do not mind polluting: the three PNGs (`robot0_agentview_left_image.png`, `robot0_agentview_right_image.png`, `robot0_eye_in_hand_image.png`) are written into the current working directory. `outputs/` is gitignored, so this is a safe place.

A mean pixel value in the tens-to-~150 range means the scene renders correctly; a value near zero means a black frame.

**Camera resolution.** The default is `128 × 128`, which is too coarse for VLM input — the wrist view loses most of the grasp geometry. `scripts/debug/check_render.py` requests `256 × 256` via `camera_widths` / `camera_heights`:

```python
env = create_env(
    env_name="PickPlaceCounterToCabinet",
    robots=["PandaOmron"],
    seed=0,
    camera_widths=256,
    camera_heights=256,
)
```

Verified on this machine at two resolutions — near-identical mean pixel values mean the scene content is unchanged, only the sampling density is:

| Resolution | left / right / wrist mean pixel |
| --- | --- |
| 128 × 128 | 97.07 / 76.96 / 92.18 |
| 256 × 256 | 97.09 / 76.87 / 92.17 |

## Project Status

### ✅ Completed (merged in `feat-system2-lora-finetune` branch)

- ✅ RoboCasa frame export script: `scripts/data_export/robocasa_export_frames.py`
- ✅ Offline subtask annotation pipeline: `scripts/annotate/batch_gen_subtask_list.py`
- ✅ Train/val split & jsonl validation utilities under `scripts/data_utils/`
- ✅ Sliding-window RoboCasa PyTorch Dataset: `vla/datasets/robocasa_dataset.py`
- ✅ Qwen2-VL backbone wrapper: `vla/vlm_backbone/qwen2vl_backbone.py` (support base model + LoRA loading)
- ✅ Structured planner output parser & schema definition: `vla/structured_planner/`
- ✅ LoRA SFT training script for System-2 subtask planner: `scripts/train/train_lora_system2.py`
- ✅ Offline rolling closed-loop replay evaluator: `scripts/planner/rolling_closed_planner.py`
- ✅ Batch validation inference script for System-2 planner: `scripts/eval/val_batch_infer_system2.py`
- ✅ Unit tests for backbone inference and planner

### ✅ Simulator environment verified (on `feat-sim-system1-integration` branch)

- ✅ `robocasa_env` conda environment built, robosuite 1.5.2 + robocasa 1.0.1 installed as editable packages
- ✅ ~23 GB of kitchen assets downloaded (textures, generative textures, fixtures, objaverse, aigen objects, lightwheel objects)
- ✅ `create_env` + `env.reset()` verified working (see **Smoke test** section above)
- ✅ Camera rendering verified at 128 × 128 and 256 × 256 through `scripts/debug/check_render.py`

### ⏳ Work-in-Progress / Todo

- ⏳ Build System-1 glue adapter layer: map natural-language subtask output from System-2 to RoboCasa heuristic atomic skills (`system1_adapter.py`)
- ⏳ Full closed-loop agent runtime (`system2_runtime.py`) for real simulator interaction
- ⏳ End-to-end agent loop inside RoboCasa simulator (`sim_loop.py`)
- ⏳ Quantitative ablation study for hierarchical planning
- ⏳ Collect simulation-run failure cases for further model iteration

## Project Structure

```
VLA-project/
├── robocasa/                     # RoboCasa source (editable install, includes 23 GB assets)
├── robosuite/                    # Robosuite simulation dependency (editable install)
├── scripts/
│   ├── __init__.py
│   ├── annotate/
│   │   ├── batch_annotate_robot_behavior.py
│   │   └── batch_gen_subtask_list.py       # Offline VLM subtask annotation
│   ├── data_export/
│   │   └── robocasa_export_frames.py       # Export frames & raw jsonl from RoboCasa
│   ├── data_utils/
│   │   ├── build_train_index.py
│   │   ├── check_jsonl.py                  # Jsonl dataset validity checker
│   │   └── split_train_val.py              # Split dataset into train / val
│   ├── debug/
│   │   └── check_render.py                 # Render sanity check, saves 3 camera PNGs
│   ├── eval/
│   │   └── val_batch_infer_system2.py      # Batch validation inference for System-2 planner
│   ├── planner/
│   │   └── rolling_closed_planner.py       # Offline state-aware rolling closed-loop replay evaluator
│   └── train/
│       └── train_lora_system2.py           # LoRA SFT training script for System2 subtask planner
├── tests/
│   ├── assets/
│   ├── __init__.py
│   ├── test_qwen2vl_backbone.py             # Unit test for Qwen2-VL backbone
│   ├── test_qwen2vl_video.py
│   ├── test_robocasa_sliding_window_infer.py # Sliding window planning inference test
│   └── test_system2_lora_planner.py         # Unit test for System2 LoRA planner
├── vla/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── robocasa_dataset.py              # Sliding-window torch Dataset for SFT
│   ├── structured_planner/
│   │   ├── __init__.py
│   │   ├── model_output_json_parser.py      # Parse planner structured JSON output
│   │   └── schema.py                        # Schema definition for planner output
│   ├── vlm_backbone/
│   │   ├── __init__.py
│   │   └── qwen2vl_backbone.py              # Qwen2-VL model loading & inference API
│   └── __init__.py
├── .gitignore
├── LICENSE
├── README.md
└── vla_env_backup.yml                       # Conda environment backup
```

## Dataset Construction Workflow

### 1. Export raw episodes from RoboCasa simulator

Run frame export script to dump sampled frames and raw index jsonl.

```bash
export PYTHONPATH=$PWD
python scripts/data_export/robocasa_export_frames.py
```

Raw jsonl fields:

| Field | Description |
| --- | --- |
| episode_id | Unique identifier for a robot episode |
| image_paths | Ordered list of frame absolute paths |
| language_instruction | High-level natural language task instruction |
| anno_path | Path to original simulation annotation json |

### 2. Offline VLM subtask annotation

> ⚠️ Apple Silicon (Mac M series) note:
> MPS backend will cause segmentation fault on multi-frame Qwen2-VL inference. Please use **CPU** for annotation.

```bash
export PYTHONPATH=$PWD
# Small test (3 episodes)
python scripts/annotate/batch_gen_subtask_list.py \
  --jsonl_path exported_frames/train.jsonl \
  --out_jsonl exported_frames/tmp_test.jsonl \
  --limit 3 \
  --device cpu
# Full dataset annotation
python scripts/annotate/batch_gen_subtask_list.py \
  --jsonl_path exported_frames/train.jsonl \
  --out_jsonl exported_frames/train_episode_with_subtask_list.jsonl \
  --limit -1 \
  --device cpu
```

The annotated output jsonl adds one new key:

| New Field | Description |
| --- | --- |
| subtask_list | VLM-generated ordered subtask sequence, `List[str]` |

### 3. Split train / val dataset

```bash
python scripts/data_utils/split_train_val.py
```

### 4. Dataset quality check

```bash
# Count total lines
wc -l exported_frames/train*.jsonl
# Random sample episodes for manual inspection
shuf -n 10 exported_frames/train_episode_with_subtask_list.jsonl
```

Use `scripts/data_utils/check_jsonl.py` for automated format validation.

## Quick Start

### 1. Test Qwen2-VL backbone

```bash
export PYTHONPATH=$PWD
python tests/test_qwen2vl_backbone.py
```

### 2. Train System2 LoRA subtask planner

```bash
export PYTHONPATH=$PWD
python scripts/train/train_lora_system2.py
```

### 3. Offline rolling closed-loop replay for System-2 planner

> This script replays historical episode frames from jsonl dataset, simulates System-2 high-level planning loop in offline mode.
> Remember: embedding-based completion judge is only for offline simulation.

```bash
export PYTHONPATH=.
python scripts/planner/rolling_closed_planner.py \
  --dataset-jsonl ./exported_frames/dataset_subtask_val.jsonl \
  --model-name "Qwen/Qwen2-VL-2B-Instruct" \
  --lora-path ./weights/system2_subtask_lora \
  --device cpu \
  --output-dir ./outputs/exp_rolling \
  --limit-samples 1 \
  --sim-threshold 0.7
```

Output: `rolling_closed_trace.jsonl` contains full step-wise trace for every episode.

### 4. Batch validation inference for System-2 planner

```bash
export PYTHONPATH=$PWD
python scripts/eval/val_batch_infer_system2.py
```

### 5. Verify the RoboCasa simulator

```bash
conda activate robocasa_env
python scripts/debug/check_render.py       # writes 3 camera PNGs into cwd
```

For the minimal `create_env` + `env.reset()` check, run the snippet in **Smoke test: environment reset** above.

### 6. Load dataset (used for SFT)

`vla/datasets/robocasa_dataset.py` implements a PyTorch Dataset with sliding window sampling for long video episodes.
It supports two modes:

- Sliding window mode: split long episode into fixed-length frame windows
- Full episode mode: load the whole episode without slicing

## Experiment & Evaluation Plan

### Evaluation Metrics (offline & simulation)

- Subtask planning accuracy (match reference subtask sequence)
- JSON parse success rate of planner output
- Task success rate in RoboCasa simulation (future)
- Instruction following completion rate
- Action sequence consistency

### Ablation Plan

- With / without subtask intermediate planning chain
- Different sliding window sizes for long video
- Prompt template variants for subtask decomposition
- Different cosine similarity thresholds (offline-only)

## Future Work Roadmap

1. Iterate System-2 LoRA fine-tuning based on offline closed-loop bad-case traces
2. Implement System-1 glue adapter: map System-2 natural-language subtask → RoboCasa heuristic atomic skill calls
3. Build real-simulator runtime `system2_runtime.py`, remove offline embedding completion-judge logic
4. Build top-level simulation main loop `sim_loop.py`, realize full System-2 + System-1 closed-loop inside RoboCasa
5. Run ablation experiments and collect quantitative simulation results
6. Clean code, organize experiment logs and write project report

## Known Pitfalls (RoboCasa setup)

| Symptom | Root cause | Fix |
| --- | --- | --- |
| `robot_name is None` / `No private macro file found` | robosuite installed into system Python 3.14, not conda env; loaded as namespace package with `__file__ == None` | `python -m pip install -e ./robosuite --no-deps --config-settings editable_mode=compat` |
| `robosuite.__file__` is `None` even after reinstall | PEP 660 meta-path finder loses against the `robosuite/` directory in cwd | Use `--config-settings editable_mode=compat` |
| `Environment name PnPCounterToCab not found` | renaming in robocasa 1.0.x | use `PickPlaceCounterToCabinet` |
| `Kitchen.__init__() got unexpected keyword argument 'render'` | `create_env` has no `render` arg | use `render_onscreen`, or omit it |
| `Namespace robocasa not found` when using `gym.make` | robocasa registers into its own registry | call `create_env` from `robocasa.utils.env_utils` instead of `gym.make` |
| `ModuleNotFoundError: robosuite` in another env | robosuite only exists in `robocasa_env` | `conda activate robocasa_env` first |
| `No such file: setup_macros.py` | looks for single-level `robosuite/scripts/`, real path is `robosuite/robosuite/scripts/` | run `python robosuite/robosuite/scripts/setup_macros.py` |
| camera images come out `128 × 128`, too small for a VLM | robosuite default camera resolution | pass `camera_widths` / `camera_heights` to `create_env` |

Inspect the actual registered environment names with:

```bash
python -c "
import robocasa
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
print(len(list(ALL_KITCHEN_ENVIRONMENTS)))
"
```

## Notes

- All exported frames and generated jsonl dataset are stored under `exported_frames/`, ignored in `.gitignore`
- Model weights, LoRA adapters and experiment outputs under `outputs/` are excluded by gitignore; do NOT commit large weights
- `robocasa/models/assets/` now contains ~23 GB of downloaded assets and must be excluded from git
- `__pycache__` folders are ignored automatically
- In `robocasa_env`, always use `python -m pip` instead of bare `pip`
- Simulator verification lives in `scripts/debug/check_render.py`; the minimal env-reset check is kept as an inline snippet in this README
- Running `scripts/debug/check_render.py` writes the three camera PNGs into the current working directory — `outputs/render_check/` is the intended place, and it is gitignored
- This project is still work-in-progress, code and workflow keep updating
- `rolling_closed_planner.py` is purely an **offline replay evaluation tool**, not the runtime for real simulator.

## References

[1] RoboCasa
[2] RoboSuite
[3] Qwen2-VL
[4] Embodied Vision-Language-Action / hierarchical planning for long-horizon manipulation