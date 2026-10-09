# VLA‑project
## VLA‑Agent: Vision‑Language‑Action Embodied Agent
> Embodied AI agent for visual‑language navigation and robot task planning, built with reinforcement learning and vision‑language foundation models.

## Project Overview
This repository implements an embodied agent that follows natural‑language instructions to complete navigation and manipulation tasks in simulated environments.
The agent combines:
- Multimodal vision‑language backbone (**LLaVA‑1.5**) for image understanding / alternative CLIP / DINOv2 for visual observation
- Large language model for high‑level task planning
- Reinforcement Learning (PPO via Stable‑Baselines3) for low‑level action control
- Habitat‑Sim / Gymnasium simulation environment

Target research direction: **Vision‑Language‑Action (VLA) embodied intelligence for robot instruction‑following tasks**.

## Key Features
- Multimodal VLM backbone (LLaVA‑1.5) for image‑text dialogue & visual scene understanding
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

# Multimodal modelscope (for LLaVA weight download)
pip install modelscope

# For habitat‑sim (simulation)
# follow official habitat installation guide
```

## Project Status

> 
> Personal research project for graduate‑school application.

- Project skeleton & repository setup
- ✅ Multimodal VLM backbone prototype: LLaVA‑1.5 image‑text inference pipeline
- Vision encoder (CLIP / DINOv2) feature extraction module
- LLM high‑level task planner prototype
- PPO RL training pipeline based on Stable‑Baselines3
- Habitat‑Sim / Gymnasium simulation wrapper
- End‑to‑end agent evaluation

## Project Structure

```
VLA‑project/
├── vla/                      # core source code package
│   ├── __init__.py
│   ├── vlm_backbone/         # Multimodal Vision‑Language backbone module
│   │   ├── __init__.py
│   │   └── llava_video.py    # LLaVA‑1.5 model load & image inference API
│   ├── agent/                # VLA agent logic (to‑do)
│   ├── vision_encoder/       # CLIP / DINOv2 feature extraction (to‑do)
│   ├── planner/              # LLM high‑level planner (to‑do)
│   ├── rl/                   # PPO training, reward function (to‑do)
│   └── env_wrapper/          # simulation environment wrapper (to‑do)
├── tests/
│   ├── __init__.py
│   └── test_llava_video.py   # Unit test: run LLaVA image inference
├── scripts/                   # main training / eval entry (to‑do)
│   ├── train_ppo.py
│   ├── eval_agent.py
│   └── run_demo.py
├── configs/                   # yaml config for training hyper‑parameters (to‑do)
├── docs/                      # experiment notes, literature review
├── logs/                      # training tensorboard / monitor logs (.gitignore)
├── checkpoints/               # model weights, simulation snapshots (.gitignore)
├── test.jpg                   # local test image for LLaVA debugging
└── README.md
```

## Quick Start

### 1. Run LLaVA‑1.5 VLM inference test

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

- Fine‑tune vision‑language backbone with LoRA/PEFT
- Connect LLaVA VLM output to high‑level agent planning module
- Support multi‑stage long‑horizon embodied tasks
- Improve generalization across unseen environments
- Compare baseline algorithms for VLN / VLA tasks

## Notes

This is an ongoing personal embodied‑AI research project.
Code will be continuously updated as experiments proceed.

> 
> Tips:
> 
> 
> - `__pycache__/` are auto‑generated python bytecode cache, ignored by `.gitignore`, do not commit them.
> - Large model weights will be stored under `~/.cache/modelscope`, **do not commit weights to git repo**.

## References

```
[1] Habitat‑Sim
[2] Stable‑Baselines3 PPO
[3] CLIP, DINOv2
[4] LLaVA: Large Language and Vision Assistant
[5] Embodied‑VLA related papers
```