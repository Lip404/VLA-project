"""
sim_closed_loop.py
Real simulator closed‑loop main loop: System2 Planner → Mapper → RoboCasa(System1)
Trace output format compatible with offline planner_offline_replay trace.
"""
import json
import os
from typing import List, Dict

# local import
from vla.vlm_backbone.qwen2vl_backbone import Qwen2VLBackbone
from scripts.planner.planner_online_runtime import system2_step
from scripts.planner.subtask_to_robocasa_skill_mapper import SubtaskSkillMapper
from scripts.sim.system1_skill_executor import system1_execute


def save_trace(trace_list: List[Dict], output_path: str):
    """Save episode trace to jsonl, compatible with offline trace format"""
    with open(output_path, "w", encoding="utf‑8") as f:
        for record in trace_list:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    # ================== Config ==================
    HIGH_LEVEL_INSTRUCTION = "Pick the tongs from the counter and place it in the cabinet."
    OUTPUT_TRACE_PATH = "./outputs/sim_trace.jsonl"
    MAX_EPISODE_STEP = 80  # Hard upper‑limit, avoid infinite simulation loop

    # Prompt template consistent with rolling_closed_planner offline script
    PROMPT_TEMPLATE = (
        "Given image observation. High‑level task: {instruction}. "
        "Subtasks that have ALREADY BEEN FINISHED: [{done_subtasks}]. "
        "Rules you MUST strictly follow: "
        "1. ONLY output subtasks that still NEED to be executed. "
        "2. NEVER repeat any subtask from the already‑finished list. "
        "3. If NO subtasks remain to do, output empty json array: []. "
        "4. Output only pure json array, no extra explanation text."
    )

    # init component
    mapper = SubtaskSkillMapper()
    trace_list: List[Dict] = []

    # todo/done state machine, maintained outside planner
    todo_subtasks: List[str] = []
    done_subtasks: List[str] = []

    # ========= RoboCasa env initialize (placeholder, replace with real robocasa api) =========
    # import robocasa
    # env = robocasa.make(...)
    # obs = env.reset()
    # current_rgb_image = obs["robot0_agentview_right"]["rgb"]
    env = None
    current_rgb_image = None

    # init System2 VLM backbone (fill your real model / lora path)
    backbone = Qwen2VLBackbone(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        lora_path="./weights/system2_subtask_lora",
        device="cuda"
    )

    print("Start real‑sim closed‑loop ...")
    for step_idx in range(MAX_EPISODE_STEP):
        # -------- Step1: call System2 planner --------
        s2_result = system2_step(
            backbone=backbone,
            image_path=None,
            high_instruction=HIGH_LEVEL_INSTRUCTION,
            todo_subtasks=todo_subtasks,
            done_subtasks=done_subtasks,
            prompt_template=PROMPT_TEMPLATE
        )
        trace_list.append(s2_result["trace_record"])

        # System2 parse error, abort this episode
        if s2_result["parse_error"] is not None:
            print(f"[System2 parse error], abort episode. error:{s2_result['parse_error']}")
            break

        active_subtask = s2_result["need_execute_subtask"]
        if active_subtask is None:
            print("[System2] No more subtasks, mission complete")
            break

        # -------- Step2: Subtask‑>RoboCasa skill glue mapper --------
        skill_config = mapper.match_skill(active_subtask)
        if skill_config is None:
            print(f"[Mapper WARN] Can not map subtask: {active_subtask}, abort episode")
            break
        print(f"step {step_idx}: active_subtask={active_subtask}, skill_config={skill_config}")

        # -------- Step3: Call RoboCasa System‑1 heuristic skill via system1_adapter --------
        if env is None:
            raise RuntimeError("RoboCasa env instance is None, please initialize real env!")
        subtask_done_flag, current_rgb_image = system1_execute(skill_config, env)

        # -------- Step4: Update state machine according to simulator real return flag --------
        if subtask_done_flag:
            todo_subtasks.pop(0)
            done_subtasks.append(active_subtask)

        # termination condition: no remaining subtask
        if len(todo_subtasks) == 0:
            print("All subtasks finished, exit simulation loop")
            break

    # save full trace log
    os.makedirs(os.path.dirname(OUTPUT_TRACE_PATH), exist_ok=True)
    save_trace(trace_list, OUTPUT_TRACE_PATH)
    print(f"simulation finished, trace saved to {OUTPUT_TRACE_PATH}")

    # if env is not None:
    #     env.close()


if __name__ == "__main__":
    main()
