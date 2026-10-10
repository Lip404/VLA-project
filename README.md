```
# VLA‑project
## VLA‑Agent: Vision‑Language‑Action Embodied Agent
> Embodied AI agent for visual‑language navigation and robot task planning, built with reinforcement learning and vision‑language foundation models.

## Project Overview
This repository implements an embodied agent that follows natural‑language instructions to complete navigation and manipulation tasks in simulated environments.

The agent combines:
- Multimodal vision‑language backbone (**Qwen2‑VL‑2B‑Instruct**, LLaVA‑1.5 as alternative baseline) for image / upcoming video understanding; alternative CLIP / DINOv2 for visual observation
- Large language model for high‑level task planning
- Reinforcement Learning (PPO via Stable‑Baselines3) for low‑level action control
- Habitat‑Sim / Gymnasium simulation environment

Target research direction: **Vision‑Language‑Action (VLA) embodied intelligence for robot instruction‑following tasks**.

## Key Features
- Multimodal VLM backbone (Qwen2‑VL‑2B‑Instruct): single‑image image‑text dialogue & visual scene understanding, output template stripping
- Visual observation preprocessing with pre‑trained vision encoders (CLIP / DINOv2)
- LLM‑based hierarchical task planning
- PPO reinforcement learning training pipeline
- Evaluation metrics for navigation success rate, SPL
- Reproducible training config and experiment logging

## Environment & Dependencies
```bash
# Create conda environment
conda create -n vla_env python=3.10
conda activate vla_env

# Core dependencies
pip install torch torchvision pillow
pip install gymnasium stable‑baselines3
pip install transformers datasets peft accelerate
pip install clip‑torch dino‑v2

# Multimodal modelscope (for model weight download)
pip install modelscope

# For habitat‑sim (simulation)
# follow official habitat installation guide
```

## Project Status

> 
> Personal research project for graduate‑school application.

- Project skeleton & repository setup
- ✅ Multimodal VLM backbone prototype: **Qwen2‑VL‑2B‑Instruct single‑image inference pipeline**
  - Model loading wrapper, image pre‑processing, chat‑template output cleaning
  - Unit test script for image‑caption inference
- ⏳ Extend VLM to support video frame input (ongoing)
- Vision encoder (CLIP / DINOv2) feature extraction module (to‑do)
- LLM high‑level task planner prototype (to‑do)
- PPO RL training pipeline based on Stable‑Baselines3 (to‑do)
- Habitat‑Sim / Gymnasium simulation wrapper (to‑do)
- End‑to‑end agent evaluation (to‑do)

## Project Structure

```
VLA‑project/
├── vla/                      # core source code package
│   ├── __init__.py
│   ├── vlm_backbone/         # Multimodal Vision‑Language backbone module
│   │   ├── __init__.py
│   │   └── llava_video.py    # Qwen2‑VL model load & single‑image inference API
│   ├── agent/                # VLA agent logic (to‑do)
│   ├── vision_encoder/       # CLIP / DINOv2 feature extraction (to‑do)
│   ├── planner/              # LLM high‑level planner (to‑do)
│   ├── rl/                   # PPO training, reward function (to‑do)
│   └── env_wrapper/          # simulation environment wrapper (to‑do)
├── tests/
│   ├── __init__.py
│   └── test_llava_video.py   # Unit test: run Qwen2‑VL single‑image inference
├── scripts/                   # main training / eval entry (to‑do)
│   ├── train_ppo.py
│   ├── eval_agent.py
│   └── run_demo.py
├── configs/                   # yaml config for training hyper‑parameters (to‑do)
├── docs/                      # experiment notes, literature review
├── logs/                      # training tensorboard / monitor logs (.gitignore)
├── checkpoints/               # model weights, simulation snapshots (.gitignore)
├── test.jpg                   # Local‑only test image for debugging, DO NOT commit
└── README.md
```

## Quick Start

### 1. Run Qwen2‑VL VLM single‑image inference test

> 
> Test multimodal vision‑language backbone, read local image and generate text description.

```
# add project root to python module search path
export PYTHONPATH=$PWD
# run test script
python tests/test_llava_video.py
```

### 2. Train PPO agent (to‑be implemented)

```
python scripts/train_ppo.py --config configs/train.yaml
```

### 3. Evaluate agent (to‑be implemented)

```
python scripts/eval_agent.py --checkpoint checkpoints/latest.zip
```

## Experiment Metrics

- Success Rate
- SPL (Success weighted by Path Length)
- Episode reward statistics
- Planning accuracy of LLM high‑level instructions

## Future Work

- Extend current VLM backbone to support video input via frame sampling
- Fine‑tune vision‑language backbone with LoRA/PEFT
- Connect VLM visual output to high‑level agent planning module
- Support multi‑stage long‑horizon embodied tasks
- Improve generalization across unseen environments
- Compare baseline algorithms for VLN / VLA tasks

## Notes

This is an ongoing personal embodied‑AI research project.
Code will be continuously updated as experiments proceed.

Tips:

- `__pycache__/` are auto‑generated python bytecode cache, ignored by `.gitignore`, do not commit them.
- Large model weights will be stored under huggingface cache directory, **do not commit weights to git repo**.
- `test.jpg` is local debugging asset only; excluded via `.gitignore`.

## References

```
[1] Habitat‑Sim
[2] Stable‑Baselines3 PPO
[3] CLIP, DINOv2
[4] LLaVA: Large Language and Vision Assistant
[5] Qwen2‑VL
[6] Embodied‑VLA related papers
```