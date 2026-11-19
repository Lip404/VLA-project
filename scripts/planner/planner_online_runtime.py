"""
planner_online_runtime.py
planner_online runtime for REAL ROBOCASA SIMULATOR.
IMPORTANT: All offline embedding‑based subtask completion‑judge logic has been REMOVED.
This module ONLY performs high‑level subtask decomposition.
State variables todo_subtasks / done_subtasks are maintained OUTSIDE (managed by sim_closed_loop).
No internal state mutation inside this module.
Trace output schema is fully compatible with planner_offline_replay offline trace.
"""
from typing import List, Dict, Optional, Any
from vla.vlm_backbone.qwen2vl_backbone import Qwen2VLBackbone
from vla.structured_planner.model_output_json_parser import extract_json_array


def system2_step(
    backbone: Qwen2VLBackbone,
    image_path: Optional[str],
    high_instruction: str,
    todo_subtasks: List[str],
    done_subtasks: List[str],
    prompt_template: str
) -> Dict[str, Any]:
    """
    Single step of System‑2 hierarchical planner for real‑sim environment.

    Args:
        backbone: loaded Qwen2‑VL backbone instance (with LoRA)
        image_path: path to current observation image from simulator
        high_instruction: top‑level natural‑language task instruction
        todo_subtasks: pending subtask list, maintained by upper sim_loop (READ‑ONLY inside function)
        done_subtasks: finished subtask list, maintained by upper sim_loop (READ‑ONLY inside function)
        prompt_template: prompt template with {instruction}, {done_subtasks} placeholders

    Returns:
        result dict, includes trace_record compatible with offline trace format
    """
    done_subtasks_text = ", ".join(done_subtasks)
    filled_prompt = prompt_template.format(
        instruction=high_instruction,
        done_subtasks=done_subtasks_text
    )

    model_raw_output: Optional[str] = None
    pred_subtasks: List[str] = []
    parse_error: Optional[str] = None

    try:
        if image_path is not None:
            model_raw_output = backbone.infer_single(
                image_path=image_path,
                prompt_text=filled_prompt
            )
        if model_raw_output is not None:
            pred_subtasks = extract_json_array(model_raw_output)
        else:
            pred_subtasks = []
    except Exception as e:
        parse_error = str(e)

    # active subtask: read‑only take first todo element, NO pop here!
    need_execute_subtask: Optional[str] = None
    if len(todo_subtasks) > 0:
        need_execute_subtask = todo_subtasks[0]

    # trace_record keep same fields as rolling_closed_planner step_record
    trace_record = {
        "prompt": filled_prompt,
        "model_raw_output": model_raw_output,
        "pred_subtasks": pred_subtasks,
        "parse_error": parse_error,
        "todo_subtasks": todo_subtasks.copy(),
        "done_subtasks": done_subtasks.copy(),
        "sim_executed_subtask": need_execute_subtask
    }

    return {
        "need_execute_subtask": need_execute_subtask,
        "todo_subtasks": todo_subtasks.copy(),
        "done_subtasks": done_subtasks.copy(),
        "prompt": filled_prompt,
        "model_raw_output": model_raw_output,
        "pred_subtasks": pred_subtasks,
        "parse_error": parse_error,
        "trace_record": trace_record
    }
