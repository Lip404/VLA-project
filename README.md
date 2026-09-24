# VLA-project
# VLA‑Agent: Vision‑Language‑Action Embodied Agent
> Embodied AI agent for visual‑language navigation and robot task planning, built with reinforcement learning and vision‑language foundation models.

## Project Overview
This repository implements an embodied agent that follows natural‑language instructions to complete navigation and manipulation tasks in simulated environments.
The agent combines:
- Vision encoder (CLIP / DINOv2) for visual observation
- Large language model for high‑level task planning
- Reinforcement Learning (PPO via Stable‑Baselines3) for low‑level action control
- Habitat‑Sim / Gymnasium simulation environment

Target research direction: **Vision‑Language‑Action (VLA) embodied intelligence for robot instruction‑following tasks**.

## Key Features
- Visual observation preprocessing with pre‑trained vision backbones
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
pip install torch torchvision
pip install gymnasium stable‑baselines3
pip install transformers datasets peft accelerate
pip install clip‑torch dino‑v2
# For habitat‑sim (simulation)
# follow official habitat installation guide
```

## Project Status
> Personal research project for graduate‑school application.

- Project skeleton & repository setup
- Vision encoder (CLIP / DINOv2) feature extraction module
- LLM high‑level task planner prototype
- PPO RL training pipeline based on Stable‑Baselines3
- Habitat‑Sim / Gymnasium simulation wrapper
- End‑to‑end agent evaluation

## Project Structure

```
VLA‑project/
├── src/                     # core source code
│   ├── agent/               # VLA agent logic
│   ├── vision_encoder/      # CLIP / DINOv2 feature extraction
│   ├── planner/             # LLM high‑level planner
│   ├── rl/                  # PPO training, reward function
│   └── env_wrapper/         # simulation environment wrapper
├── scripts/
│   ├── train_ppo.py         # main training entry
│   ├── eval_agent.py        # evaluation script
│   └── run_demo.py          # quick demo
├── configs/                 # yaml config for training hyper‑parameters
├── docs/                    # experiment notes, literature review
├── logs/                    # training tensorboard / monitor logs (.gitignore)
├── checkpoints/             # model weights (.gitignore)
└── README.md
```

## Quick Start

### Train

```
python scripts/train_ppo.py --config configs/train.yaml
```

### Evaluate

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
- Support multi‑stage long‑horizon embodied tasks
- Improve generalization across unseen environments
- Compare baseline algorithms for VLN / VLA tasks

## Notes

This is an ongoing personal embodied‑AI research project.
Code will be continuously updated as experiments proceed.

## References

```
[1] Habitat‑Sim
[2] Stable‑Baselines3 PPO
[3] CLIP, DINOv2
[4] Embodied‑VLA related papers
```