"""
planner_online_runtime.py

System-2 single step for the REAL RoboCasa simulator.

All offline embedding-based subtask completion logic has been removed: this module
only decomposes the task. Todo / done state is owned by sim_closed_loop.

Active subtask selection: the first entry of the planner's CURRENT output that is
not already in `done_subtasks`. `todo_subtasks` is kept only for trace compatibility
and is never mutated here.
"""

from typing import Any, Dict, List, Optional

from PIL import Image

from vla.vlm_backbone.qwen2vl_backbone import Qwen2VLBackbone
from vla.structured_planner.model_output_json_parser import extract_json_array


def _fill_template(template: str, instruction: str, done_subtasks: List[str]) -> str:
    """Fill the prompt without str.format, so JSON braces in the template cannot crash it."""
    done_text = ", ".join(done_subtasks)
    return template.replace("{instruction}", instruction).replace("{done_subtasks}", done_text)


def system2_step(
    backbone: Qwen2VLBackbone,
    image_path: Optional[str],
    high_instruction: str,
    todo_subtasks: List[str],
    done_subtasks: List[str],
    prompt_template: str,
    image: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """
    One planning step for the real simulator.

    Args:
        backbone: Qwen2-VL backbone with the System-2 LoRA adapter loaded
        image_path: fallback observation on disk
        high_instruction: top-level task instruction
        todo_subtasks: read-only, kept for trace compatibility
        done_subtasks: subtasks already executed in this episode
        prompt_template: template containing {instruction} and {done_subtasks}
        image: in-memory PIL observation, takes priority over image_path

    Returns:
        dict with `need_execute_subtask` (None means the task is finished) and
        `trace_record` compatible with the offline replay trace.
    """
    filled_prompt = _fill_template(prompt_template, high_instruction, done_subtasks)

    model_raw_output: Optional[str] = None
    pred_subtasks: List[str] = []
    parse_error: Optional[str] = None

    if image is None and image_path is None:
        parse_error = "no observation provided: both image and image_path are None"
    else:
        try:
            if image is not None:
                model_raw_output = backbone.infer_single_pil(
                    pil_image=image, prompt_text=filled_prompt
                )
            else:
                model_raw_output = backbone.infer_single(
                    image_path=image_path, prompt_text=filled_prompt
                )
            if model_raw_output is not None:
                pred_subtasks = extract_json_array(model_raw_output)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

    # Active subtask: first fresh prediction that has not been executed yet.
    done_set = {text.strip() for text in done_subtasks}
    pending = [
        text.strip()
        for text in pred_subtasks
        if text and text.strip() and text.strip() not in done_set
    ]
    need_execute_subtask: Optional[str] = pending[0] if pending else None

    trace_record = {
        "prompt": filled_prompt,
        "model_raw_output": model_raw_output,
        "pred_subtasks": pred_subtasks,
        "pending_subtasks": pending,
        "parse_error": parse_error,
        "todo_subtasks": list(todo_subtasks),
        "done_subtasks": list(done_subtasks),
        "sim_executed_subtask": need_execute_subtask,
    }

    return {
        "need_execute_subtask": need_execute_subtask,
        "pending_subtasks": pending,
        "todo_subtasks": list(todo_subtasks),
        "done_subtasks": list(done_subtasks),
        "prompt": filled_prompt,
        "model_raw_output": model_raw_output,
        "pred_subtasks": pred_subtasks,
        "parse_error": parse_error,
        "trace_record": trace_record,
    }
