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

## Project Status
> Personal research project for graduate‑school application.
- [x] Project skeleton & repository setup
- [ ] Vision encoder (CLIP / DINOv2) feature extraction module
- [ ] LLM high‑level task planner prototype
- [ ] PPO RL training pipeline based on Stable‑Baselines3
- [ ] Habitat‑Sim / Gymnasium simulation wrapper
- [ ] End‑to‑end agent evaluation

## Notes
This is an ongoing personal embodied‑AI research project.
Code will be continuously updated as experiments proceed.
