```
# VLA-project
## VLA‑Agent: Vision‑Language‑Action Embodied Agent
>
> Embodied AI agent for vision‑language robot manipulation & long‑horizon task planning, with supervised fine‑tuning on subtask‑annotated RoboCasa episode data.

## Project Overview
This repository builds a hierarchical embodied agent that follows natural language instructions for robot manipulation in the RoboCasa simulation environment.
We split the long‑horizon task solving pipeline into two parts (System1 / System2 paradigm):
- **System2 (Hierarchical Planner)**: Given a high‑level task instruction and visual video observation, decompose the long task into an ordered sequence of executable subtasks. Implemented using LoRA‑finetuned Qwen2‑VL.
- **System1 (Low‑level Action Executor)**: Take subtask planning results + visual input to predict robot low‑level control actions. We plan to use RoboCasa built‑in heuristic skill policies as System‑1 executor.

The backbone vision‑language model we use is **Qwen2‑VL**, and we build an offline VLM annotation pipeline to automatically generate subtask labels for RoboCasa episodes, then use these annotated data for multimodal supervised fine‑tuning.

> **Important design note**: End‑to‑end VLA models struggle with long‑horizon household manipulation tasks. Inspired by human cognitive process, we decouple high‑level task decomposition (System‑2) from low‑level motor execution (System‑1), instead of pursuing monolithic end‑to‑end policy.

## Key Features
- Qwen2‑VL multimodal backbone: multi‑frame video observation input, standardized inference wrapper with optional LoRA adapter loading
- Offline subtask annotation pipeline: automatically generate ordered subtask list for each RoboCasa episode
- Sliding‑window PyTorch Dataset for long video episode sampling (`robocasa_dataset.py`)
- Data export, split & validation tools for RoboCasa simulation episodes
- Structured output parser for planner model JSON outputs
- LoRA supervised fine‑tuning script for System‑2 subtask planner
- **Offline rolling closed‑loop replay evaluator** (`scripts/planner/rolling_closed_planner.py`): state‑aware System‑2 offline simulation framework
  > ⚠️ Warning: embedding‑cosine‑similarity‑based subtask completion judge is **offline‑only simulation logic**. This logic will be completely removed when connecting to real RoboCasa simulator; real subtask done flag shall come from simulator environment state, not VLM semantic inference.
- Configurable prompt template, trace log export for offline closed‑loop evaluation

## Environment & Dependencies
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

```
conda env create -f vla_env_backup.yml
```

## Project Status

### ✅ Completed (merged in `feat‑system2‑lora‑finetune` branch)

- ✅ RoboCasa frame export script: `scripts/data_export/robocasa_export_frames.py`
- ✅ Offline subtask annotation pipeline: `scripts/annotate/batch_gen_subtask_list.py`
- ✅ Train/val split & jsonl validation utilities under `scripts/data_utils/`
- ✅ Sliding‑window RoboCasa PyTorch Dataset: `vla/datasets/robocasa_dataset.py`
- ✅ Qwen2‑VL backbone wrapper: `vla/vlm_backbone/qwen2vl_backbone.py` (support base model + LoRA loading)
- ✅ Structured planner output parser & schema definition: `vla/structured_planner/`
- ✅ LoRA SFT training script for System‑2 subtask planner: `scripts/train/train_lora_system2.py`
- ✅ Offline rolling closed‑loop replay evaluator: `scripts/planner/rolling_closed_planner.py`
- ✅ Batch validation inference script for System‑2 planner: `scripts/eval/val_batch_infer_system2.py`
- ✅ Unit tests for backbone inference and planner

### ⏳ Work‑in‑Progress / Todo

- ⏳ Build System‑1 glue adapter layer: map natural‑language subtask output from System‑2 to RoboCasa heuristic atomic skills
- ⏳ Full closed‑loop agent runtime (`system2_runtime.py`) for real simulator interaction
- ⏳ End‑to‑end agent loop inside RoboCasa simulator (`sim_loop.py`)
- ⏳ Quantitative ablation study for hierarchical planning
- ⏳ Collect simulation‑run failure cases for further model iteration

## Project Structure

```
VLA‑project/
├── robocasa/                     # RoboCasa related source
├── robosuite/                    # Robosuite simulation dependency
├── scripts/
│   ├── __init__.py
│   ├── annotate/
│   │   ├── batch_annotate_robot_behavior.py
│   │   └── batch_gen_subtask_list.py       # Offline VLM subtask annotation
│   ├── data_export/
│   │   └── robocasa_export_frames.py       # Export frames & raw jsonl from RoboCasa
│   ├── data_utils/
│   │   ├── build_train_index.py
│   │   ├── check_jsonl.py                 # Jsonl dataset validity checker
│   │   └── split_train_val.py             # Split dataset into train / val
│   ├── eval/
│   │   └── val_batch_infer_system2.py     # Batch validation inference for System‑2 planner
│   ├── planner/
│   │   └── rolling_closed_planner.py      # Offline state‑aware rolling closed‑loop replay evaluator
│   └── train/
│       └── train_lora_system2.py          # LoRA SFT training script for System2 subtask planner
├── tests/
│   ├── assets/
│   ├── __init__.py
│   ├── test_qwen2vl_backbone.py            # Unit test for Qwen2‑VL backbone
│   ├── test_qwen2vl_video.py
│   ├── test_robocasa_sliding_window_infer.py # Sliding window planning inference test
│   └── test_system2_lora_planner.py        # Unit test for System2 LoRA planner
├── vla/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── robocasa_dataset.py             # Sliding‑window torch Dataset for SFT
│   ├── structured_planner/
│   │   ├── __init__.py
│   │   ├── model_output_json_parser.py     # Parse planner structured JSON output
│   │   └── schema.py                       # Schema definition for planner output
│   ├── vlm_backbone/
│   │   ├── __init__.py
│   │   └── qwen2vl_backbone.py             # Qwen2‑VL model loading & inference API
│   └── __init__.py
├── .gitignore
├── LICENSE
├── README.md
└── vla_env_backup.yml                     # Conda environment backup
```

## Dataset Construction Workflow

### 1. Export raw episodes from RoboCasa simulator

Run frame export script to dump sampled frames and raw index jsonl.

```
export PYTHONPATH=$PWD
python scripts/data_export/robocasa_export_frames.py
```

Raw jsonl fields:

| Field | Description |
| --- | --- |
| episode_id | Unique identifier for a robot episode |
| image_paths | Ordered list of frame absolute paths |
| language_instruction | High‑level natural language task instruction |
| anno_path | Path to original simulation annotation json |

### 2. Offline VLM subtask annotation

> 
> ⚠️ Apple Silicon (Mac M series) note:
> MPS backend will cause segmentation fault on multi‑frame Qwen2‑VL inference. Please use **CPU** for annotation.

```
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
| subtask_list | VLM‑generated ordered subtask sequence, `List[str]` |

### 3. Split train / val dataset

```
python scripts/data_utils/split_train_val.py
```

### 4. Dataset quality check

```
# Count total lines
wc -l exported_frames/train*.jsonl
# Random sample episodes for manual inspection
shuf -n 10 exported_frames/train_episode_with_subtask_list.jsonl
```

Use `scripts/data_utils/check_jsonl.py` for automated format validation.

## Quick Start

### 1. Test Qwen2‑VL backbone

```
export PYTHONPATH=$PWD
python tests/test_qwen2vl_backbone.py
```

### 2. Train System2 LoRA subtask planner

```
export PYTHONPATH=$PWD
python scripts/train/train_lora_system2.py
```

### 3. Offline rolling closed‑loop replay for System‑2 planner

> 
> This script replays historical episode frames from jsonl dataset, simulates System‑2 high‑level planning loop in offline mode.
> Remember: embedding‑based completion judge is only for offline simulation.

```
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

Output: `rolling_closed_trace.jsonl` contains full step‑wise trace for every episode.

### 4. Batch validation inference for System‑2 planner

```
export PYTHONPATH=$PWD
python scripts/eval/val_batch_infer_system2.py
```

### 5. Load dataset (used for SFT)

`vla/datasets/robocasa_dataset.py` implements a PyTorch Dataset with sliding window sampling for long video episodes.
It supports two modes:

- Sliding window mode: split long episode into fixed‑length frame windows
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
- Different cosine similarity thresholds (offline‑only)

## Future Work Roadmap

1. Iterate System‑2 LoRA fine‑tuning based on offline closed‑loop bad‑case traces
2. Implement System‑1 glue adapter: map System‑2 natural‑language subtask → RoboCasa heuristic atomic skill calls
3. Build real‑simulator runtime `system2_runtime.py`, remove offline embedding completion‑judge logic
4. Build top‑level simulation main loop `sim_loop.py`, realize full System‑2 + System‑1 closed‑loop inside RoboCasa
5. Run ablation experiments and collect quantitative simulation results
6. Clean code, organize experiment logs and write project report

## Notes

- All exported frames and generated jsonl dataset are stored under `exported_frames/`, ignored in `.gitignore`
- Model weights, LoRA adapters and experiment outputs under `outputs/` are excluded by gitignore; do NOT commit large weights
- `__pycache__` folders are ignored automatically
- This project is still work‑in‑progress, code and workflow keep updating
- `rolling_closed_planner.py` is purely an **offline replay evaluation tool**, not the runtime for real simulator.

## References

[1] RoboCasa
[2] RoboSuite
[3] Qwen2‑VL
[4] Embodied Vision‑Language‑Action / hierarchical planning for long‑horizon manipulation