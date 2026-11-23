"""
sim_closed_loop.py

Real simulator closed-loop: System2 Planner -> SubtaskSkillMapper -> RoboCasa (System1).
Trace output stays compatible with the offline planner replay trace.
"""

# Allow running this file directly as `python scripts/sim/sim_closed_loop.py`:
# put the project root (scripts/sim -> scripts -> root) on sys.path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os
from typing import Dict, List

import numpy as np
from PIL import Image

import robocasa
from robocasa.utils.env_utils import create_env

# local imports
from vla.vlm_backbone.qwen2vl_backbone import Qwen2VLBackbone
from scripts.planner.planner_online_runtime import system2_step
from scripts.planner.subtask_to_robocasa_skill_mapper import SubtaskSkillMapper
from scripts.sim.system1_skill_executor import read_rgb, system1_execute


def save_trace(trace_list: List[Dict], output_path: str):
    """Save the episode trace as jsonl, compatible with the offline trace format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in trace_list:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_env(seed: int = 0, camera_size: int = 256, max_attempts: int = 12, reach: float = 0.55):
    """RoboCasa env with a sample whose object is inside arm reach."""
    env = create_env(
        env_name="PickPlaceCounterToCabinet",
        robots=["PandaOmron"],
        seed=seed,
        camera_widths=camera_size,
        camera_heights=camera_size,
    )
    obs = None
    for attempt in range(max_attempts):
        obs = env.reset()
        d = float(
            np.linalg.norm(
                np.asarray(obs["obj_pos"], float)[:2] - np.asarray(obs["robot0_eef_pos"], float)[:2]
            )
        )
        print(f"[env] attempt {attempt}: object is {d:.2f} m from the eef")
        if d <= reach:
            break
    else:
        print(f"[env] WARNING: no sample within {reach} m; the arm cannot reach the object")
    env._last_obs = obs
    return env, obs


def main():
    # ================== Config ==================
    HIGH_LEVEL_INSTRUCTION = "Pick the tongs from the counter and place it in the cabinet."
    OUTPUT_TRACE_PATH = "./outputs/sim_trace.jsonl"
    MAX_EPISODE_STEP = 3              # upper bound on System2 planning rounds
    SKILL_STEP_BUDGET = 300            # action budget for a single atomic skill
    MAX_RETRY_PER_SUBTASK = 3          # give up after N consecutive failures on one subtask

    PROMPT_TEMPLATE = (
        "Given image observation. High-level task: {instruction}. "
        "Subtasks that have ALREADY BEEN FINISHED: [{done_subtasks}]. "
        "Rules you MUST strictly follow: "
        "1. ONLY output subtasks that still NEED to be executed. "
        "2. NEVER repeat any subtask from the already-finished list. "
        "3. If NO subtasks remain to do, output empty json array: []. "
        "4. Output only pure json array, no extra explanation text."
    )

    mapper = SubtaskSkillMapper()
    trace_list: List[Dict] = []

    # done_subtasks is the single source of truth handed to System2.
    # todo_subtasks is left empty on purpose: the planner decides what comes next.
    todo_subtasks: List[str] = []
    done_subtasks: List[str] = []

    env, obs = build_env(seed=0, camera_size=256)
    current_rgb_image = read_rgb(obs)

    # Default device is cpu: the MPS backend segfaults on multi-frame Qwen2-VL inference.
    backbone = Qwen2VLBackbone(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        lora_path="./weights/system2_subtask_lora",
        device="cpu",
    )

    print("Start real-sim closed-loop ...")

    last_subtask = None
    retry_count = 0

    for step_idx in range(MAX_EPISODE_STEP):
        # -------- Step 1: System2 planning --------
        pil_obs = Image.fromarray(current_rgb_image)
        s2_result = system2_step(
            backbone=backbone,
            image_path=None,
            image=pil_obs,
            high_instruction=HIGH_LEVEL_INSTRUCTION,
            todo_subtasks=todo_subtasks,
            done_subtasks=done_subtasks,
            prompt_template=PROMPT_TEMPLATE,
        )
        trace_list.append(s2_result["trace_record"])

        if s2_result["parse_error"] is not None:
            print(f"[System2 parse error] abort episode: {s2_result['parse_error']}")
            break

        active_subtask = s2_result["need_execute_subtask"]
        # The planner running out of subtasks ends the episode, but it does not
        # prove the task succeeded; the simulator's own check decides that.
        if active_subtask is None:
            print("[System2] no remaining subtasks")
            break

        # ------------- Step 2: subtask -> heuristic skill --------
        skill_config = mapper.match_skill(active_subtask)
        if skill_config is None:
            print(f"[Mapper] cannot map subtask: {active_subtask}, abort episode")
            break

        if active_subtask != last_subtask:
            last_subtask = active_subtask
            retry_count = 0

        print(f"step {step_idx}: subtask={active_subtask} skill={skill_config}")

        # -------- Step 3: System1 executes one atomic skill ---------
        try:
            subtask_done, current_rgb_image = system1_execute(
                skill_config, env, max_sim_steps=SKILL_STEP_BUDGET
            )
        except NotImplementedError as exc:
            print(f"[System1] unsupported skill, abort episode: {exc}")
            break

        # -------- Step 4: state machine driven by the simulator result --------
        if subtask_done:
            done_subtasks.append(active_subtask)
            print(f"[System1] subtask done: {active_subtask}")
        else:
            retry_count += 1
            print(f"[System1] subtask failed ({retry_count}/{MAX_RETRY_PER_SUBTASK}): {active_subtask}")
            if retry_count >= MAX_RETRY_PER_SUBTASK:
                print("[System1] retry budget exhausted, abort episode")
                break

    episode_success = bool(env._check_success())
    save_trace(trace_list, OUTPUT_TRACE_PATH)
    print(f"simulation finished (success={episode_success}), trace saved to {OUTPUT_TRACE_PATH}")
    env.close()


if __name__ == "__main__":
    main()
