# VLA-project

## VLA-Agent: Vision-Language-Action Embodied Agent

> Embodied AI agent for vision-language robot manipulation & long-horizon task planning, with supervised fine-tuning on subtask-annotated RoboCasa episode data.

## Project Overview

This repository builds a hierarchical embodied agent that follows natural language instructions for robot manipulation in the RoboCasa simulation environment.
We split the long-horizon task solving pipeline into two parts (System1 / System2 paradigm):

- **System2 (Hierarchical Planner)**: Given a high-level task instruction and visual video observation, decompose the long task into an ordered sequence of executable subtasks. Implemented using LoRA-finetuned Qwen2-VL.
- **System1 (Low-level Action Executor)**: Take subtask planning results + visual input to predict robot low-level control actions. Implemented as RoboCasa heuristic atomic skills driving the OSC pose / gripper / base / torso action channels directly (`scripts/sim/system1_skill_executor.py`).

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
- **Real-simulator System-1 executor** (`scripts/sim/system1_skill_executor.py`): five heuristic RoboCasa atomic skills (pick / place / navigate / open door / close door) that write directly into the OSC pose, gripper, base and torso action channels
- **Real-simulator closed loop** (`scripts/sim/sim_closed_loop.py`): System2 plan → subtask-to-skill mapper → System1 execution → `env.step`, with the episode verdict taken from `env._check_success()` instead of VLM inference

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

## RoboCasa Control Facts (verified on `PandaOmron`)

Everything below was measured on this machine inside `robocasa_env`; it saves re-discovering the same traps when a new task or robot is wired up.

| Fact | Behaviour |
| --- | --- |
| Action vector size | `env.action_dim == 12` **only after `env.reset()`** — before reset `env.robots`, `env.action_dim`, `env.fixtures`, `env.objects` are all unset |
| Layout | `[0:6]` arm `OSC_POSE` delta `[dx, dy, dz, dax, day, daz]`, `[6]` gripper (`-1` open / `+1` closed), `[7:9]` mobile base, `[10:12]` torso |
| Normalization | `default_pandaomron.json` sets `input_max 1 / output_max 0.05`, so commands are in `[-1, 1]` and `1.0` means 5 cm; metre deltas must be divided by `0.05` |
| Command frame | `OSC_POSE` is `input_ref_frame: base`, and the base orientation is randomized per reset (the same command can move the eef along different world axes). `_action_frame()` probes twice and applies `pinv(frame) @ world_delta`; the z axis is unaffected |
| Object registry | `env.objects == {"obj", "dist_counter", "dist_cab"}` while `env.model.mujoco_objects` holds only fixtures. Category names (`tongs`) are never registry keys, and every entity has `body_id == None`, so poses come from `entity.root_body` |
| Object truth | `obs["obj_pos"]` is the world pose (`obs["obj"]` does not exist); `obs["obj_to_robot0_eef_pos"]` is expressed in the eef frame and must not be used as a world direction |
| Task cabinet | `env.cab` is the place target; `_check_success()` is `OU.obj_inside_of(self, "obj", self.cab) and OU.gripper_obj_far(self)`, the latter requiring the gripper to end 0.25 m away from the object |
| Cabinet geometry | `env.cab.get_int_sites(relative=False)` returns the interior box as `(p0, px, py, pz)`; hinge doors start **already open** (joint `qpos` at its limit) |
| Base motion | `base_pos` and `robot0_base` are stale `[10, 10, 0]` placeholders, the robot has **no free joint**, and torso qpos is already at its lower limit — only the velocity dims `[7:9]` can move the base |
| Gripper state | fully open `±0.036`, empty closed `≈ 0`, **holding the tongs `≈ ±0.03`** — this is the cheapest grasp check, independent of the object pose |
| Contact blocking | counters carry fixtures (toaster ovens, microwaves) taller than the object, so a straight diagonal approach stalls mid-air; travel at a clearance height first, then descend vertically |

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

### ✅ System-1 heuristic execution verified (on `feat-sim-system1-integration` branch)

- ✅ Five atomic skills implemented in `scripts/sim/system1_skill_executor.py`: `PickObject`, `PlaceObject`, `NavigateTo`, `OpenSingleDoor`, `CloseSingleDoor`
- ✅ Action channel resolved: `action_dim == 12` → arm `[0:6]`, gripper `[6]`, mobile base `[7:9]`, torso `[10:]`; `OSC_POSE` commands normalized to `output_max = 0.05 m`
- ✅ Action frame calibrated at runtime (`_action_frame`): the OSC frame is `base`-relative and the base orientation changes per reset, so a 2×2 command→world rotation is measured before every move
- ✅ `PickObject` verified in closed loop: `yaw +0.5x10, offset 0.05 → rise +0.117 m`, gripper `qpos [0.0344, -0.0306]` (a held object keeps the fingers at ≈ ±0.03)
- ✅ `scripts/sim/sim_closed_loop.py` runs System2 → mapper → System1 → `env.step` end to end on `PickPlaceCounterToCabinet`; the episode verdict comes from `env._check_success()`
- ⚠️ `PlaceObject` lifts and positions the object at the cabinet opening but loses it on the frame lip during the final push — see **Known Limitations (System 1)**

### ⏳ Work-in-Progress / Todo

- ⏳ `PlaceObject` insertion: the object must be held at eef height (side grasp or wrist pitch), or the insertion height must be compensated by the measured dangle
- ⏳ Mobile-base transport for layouts where the object and `env.cab` are more than one arm span apart
- ⏳ Extend the mapper vocabulary (`开出抽屉`, `user picks tongs` without the `robot ` prefix)
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
│   │   ├── rolling_closed_planner.py       # Offline state-aware rolling closed-loop replay evaluator
│   │   ├── planner_online_runtime.py       # Online System-2 runtime: pick the first unfinished subtask
│   │   └── subtask_to_robocasa_skill_mapper.py  # NL subtask -> heuristic skill config
│   ├── sim/
│   │   ├── sim_closed_loop.py              # Real-simulator System2 -> mapper -> System1 loop
│   │   └── system1_skill_executor.py       # Five heuristic RoboCasa atomic skills
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

### 6. Run the real-simulator closed loop

```bash
conda activate robocasa_env
python scripts/sim/sim_closed_loop.py
```

The loop plans with System2, maps the subtask to a heuristic skill, executes it in the simulator and reports the verdict from `env._check_success()`. Expect `PickObject ok ... rise +0.1xx m` for the pick; the place step currently ends in `PlaceObject failed: cabinet entry timeout` (see **Known Limitations (System 1)**).

### 7. Load dataset (used for SFT)
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

1. ✅ System-1 glue adapter: System-2 natural-language subtask → RoboCasa heuristic atomic skills (`scripts/planner/subtask_to_robocasa_skill_mapper.py`)
2. ✅ Real-simulator runtime: `scripts/sim/sim_closed_loop.py` replaces the offline embedding completion judge with `env._check_success()`
3. ✅ Full System-2 + System-1 closed loop inside RoboCasa (pick verified end to end)
4. Finish `PlaceObject` insertion: grasp the object at eef height, or compensate the insertion height by the measured dangle
5. Mobile-base transport so layouts beyond one arm span are solvable
6. Iterate System-2 LoRA fine-tuning based on closed-loop bad-case traces
7. Run ablation experiments and collect quantitative simulation results
8. Clean code, organize experiment logs and write project report

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

## Known Limitations (System 1)

- `PlaceObject` reaches the cabinet opening and then drops the object on the frame lip during the final push. The tongs hangs about **0.22 m below the eef** (its grip site sits at eef height), so its body arrives at the opening lower than the commanded end pose. Fixing it requires a grasp that holds the object at eef height (side grasp / wrist pitch) or an insertion height compensated by the measured dangle.
- Mobile-base transport is not implemented. Episodes are therefore reset onto layouts where both the object and `env.cab` are inside one arm span (verified on 12/12 resets of `PickPlaceCounterToCabinet`, `obj→eef` 0.19–0.29 m, `cab→eef` 0.27–0.55 m).
- `OpenSingleDoor` / `CloseSingleDoor` never ran successfully: the door handle cannot be resolved through the currently known RoboCasa attributes, and `_door_diagnostics()` prints every candidate name it tried. Cabinet doors are already open at reset, so this does not block the pick-and-place demo.
- The mapper vocabulary does not cover `开出抽屉` or `user picks tongs` (a missing `robot ` prefix), and those subtasks abort the episode before System 1 is reached.
- System2 currently plans a single subtask (`robot picks up tongs`) for this task, so the loop ends after the pick and the environment verdict stays `False`.

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