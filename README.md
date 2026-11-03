# VLA-project

## VLA-Agent: Vision-Language-Action Embodied Agent

> 
> Embodied AI agent for vision-language robot manipulation & long-horizon task planning, with supervised fine-tuning on subtask-annotated RoboCasa episode data.

## Project Overview

This repository builds a hierarchical embodied agent that follows natural language instructions for robot manipulation in the RoboCasa simulation environment.
We split the long-horizon task solving pipeline into two parts (System1 / System2 paradigm):

- **System2 (Hierarchical Planner)**: Given a high-level task instruction and visual video observation, decompose the long task into an ordered sequence of executable subtasks.
- **System1 (Low-level Action Executor)**: Take subtask planning results + visual input to predict robot low-level control actions.

The backbone vision-language model we use is **Qwen2-VL**, and we build an offline VLM annotation pipeline to automatically generate subtask labels for RoboCasa episodes, then use these annotated data for multimodal supervised fine-tuning.


## Key Features

- Qwen2-VL multimodal backbone: multi-frame video observation input, standardized inference wrapper
- Offline subtask annotation pipeline: automatically generate ordered subtask list for each RoboCasa episode
- Sliding-window PyTorch Dataset for long video episode sampling (`robocasa_dataset.py`)
- Data export & validation tools for RoboCasa simulation episodes
- Structured output parser for planner model JSON outputs
- Ready for LoRA supervised fine-tuning, planning inference test script
- Built-in inference test script for sliding window episode reasoning

## Environment & Dependencies

```
# Create conda environment
conda create -n vla_env python=3.10
conda activate vla_env

# Install dependencies
pip install torch torchvision pillow
pip install transformers datasets peft accelerate
pip install tqdm
```

You can also recreate the environment from the backup yaml:

```
conda env create -f vla_env_backup.yml
```

## Project Status

- ✅ RoboCasa frame export script: `scripts/data_export/robocasa_export_frames.py`
- ✅ Offline subtask annotation pipeline: `scripts/annotate/batch_gen_subtask_list.py`
- ✅ Sliding-window RoboCasa PyTorch Dataset: `vla/datasets/robocasa_dataset.py`
- ✅ Qwen2-VL backbone wrapper: `vla/vlm_backbone/qwen2vl_backbone.py`
- ✅ Structured planner output parser & schema definition: `vla/structured_planner/`
- ✅ Inference test scripts for backbone and sliding-window planning
- ⏳ Multimodal SFT training script (to be implemented)
- ⏳ Closed-loop agent evaluation inside RoboCasa simulator (to be implemented)
- ⏳ Ablation study for hierarchical planning (to be implemented)

## Project Structure

```
VLA-project/
├── robocasa/                     # RoboCasa related source
├── robosuite/                    # Robosuite simulation dependency
├── scripts/
│   ├── annotate/
│   │   ├── batch_annotate_robot_behavior.py
│   │   └── batch_gen_subtask_list.py       # Offline VLM subtask annotation
│   ├── data_export/
│   │   └── robocasa_export_frames.py       # Export frames & raw jsonl from RoboCasa
│   ├── data_utils/
│   │   ├── build_train_index.py
│   │   └── check_jsonl.py                 # Jsonl dataset validity checker
│   └── __init__.py
├── tests/
│   ├── assets/
│   ├── test_qwen2vl_backbone.py            # Unit test for Qwen2-VL backbone
│   ├── test_qwen2vl_video.py
│   └── test_robocasa_sliding_window_infer.py # Sliding window planning inference test
├── vla/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── robocasa_dataset.py             # Sliding-window torch Dataset for SFT
│   ├── structured_planner/
│   │   ├── __init__.py
│   │   ├── model_output_json_parser.py     # Parse planner structured JSON output
│   │   └── schema.py                       # Schema definition for planner output
│   ├── vlm_backbone/
│   │   ├── __init__.py
│   │   └── qwen2vl_backbone.py             # Qwen2-VL model loading & inference API
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
| language_instruction | High-level natural language task instruction |
| anno_path | Path to original simulation annotation json |

### 2. Offline VLM subtask annotation

> 
> ⚠️ Apple Silicon (Mac M series) note:
> MPS backend will cause segmentation fault on multi-frame Qwen2-VL inference. Please use **CPU** for annotation.

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
| subtask_list | VLM-generated ordered subtask sequence, `List[str]` |

### 3. Dataset quality check

```
# Count total lines
wc -l exported_frames/train*.jsonl
# Random sample episodes for manual inspection
shuf -n 10 exported_frames/train_episode_with_subtask_list.jsonl
```

Use `scripts/data_utils/check_jsonl.py` for automated format validation.

## Quick Start

### 1. Test Qwen2-VL backbone

```
export PYTHONPATH=$PWD
python tests/test_qwen2vl_backbone.py
```

### 2. Test sliding-window planning inference

```
export PYTHONPATH=$PWD
python tests/test_robocasa_sliding_window_infer.py
```

### 3. Load dataset (used for SFT)

`vla/datasets/robocasa_dataset.py` implements a PyTorch Dataset with sliding window sampling for long video episodes.
It supports two modes:

- Sliding window mode: split long episode into fixed-length frame windows
- Full episode mode: load the whole episode without slicing

## Experiment & Evaluation Plan

### Evaluation Metrics

- Subtask planning accuracy (match reference subtask sequence)
- Task success rate in RoboCasa simulation
- Instruction following completion rate
- Action sequence consistency

### Ablation Plan

- With / without subtask intermediate planning chain
- Different sliding window sizes for long video
- Prompt template variants for subtask decomposition

## Future Work

1. Complete SFT training script, use `train_episode_with_subtask_list.jsonl` for hierarchical planner fine-tuning
2. Build full agent loop: visual input → subtask planner → low-level action prediction
3. Deploy agent into RoboCasa simulator for closed-loop evaluation
4. Run ablation experiments and collect quantitative results
5. Clean code, organize experiment logs and write project report

## Notes

- All exported frames and generated jsonl dataset are stored under `exported_frames/`, ignored in `.gitignore`
- Model weights are cached locally by Hugging Face, do NOT commit large weights
- `__pycache__` folders are ignored automatically
- This project is still work-in-progress, code and workflow keep updating

## References

[1] RoboCasa
[2] RoboSuite
[3] Qwen2-VL
[4] Embodied Vision-Language-Action / hierarchical planning for long-horizon manipulation