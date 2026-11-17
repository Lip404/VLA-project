"""
Rolling closed‑loop subtask planner entry.
Offline simulation mode: replay existing dataset sequential frames to simulate robot observation feedback loop.
Future: replace observation provider for real‑simulation online running.
State‑aware version: maintain todo/done subtask state, simulate subtask execution, prepare for System‑1 integration.
"""
import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from vla.vlm_backbone.qwen2vl_backbone import Qwen2VLBackbone
from vla.structured_planner.model_output_json_parser import extract_json_array


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf‑8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], out_path: str):
    with open(out_path, "w", encoding="utf‑8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_episode_rolling_planner(
    backbone: Qwen2VLBackbone,
    embedding_model: SentenceTransformer,
    episode_sample: Dict[str, Any],
    prompt_template: str,
    max_loop: int = 100,
    sim_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Run rolling closed‑loop for single episode with subtask state machine.
    Offline: iterate pre‑stored sequential image_paths as new observations after simulated execution.
    Subtask completion is judged by VLM image observation + embedding semantic similarity.
    :param backbone: loaded VLM backbone with/without LoRA
    :param embedding_model: sentence‑transformers embedding model for semantic similarity check
    :param episode_sample: one sample from dataset_subtask jsonl
    :param prompt_template: prompt template fed to planner, supports {instruction}, {done_subtasks} placeholders
    :param max_loop: hard upper bound to avoid infinite loop
    :param sim_threshold: cosine similarity threshold, below value treat subtask as finished
    :return: episode trace dict including every step inference result and subtask state
    """
    episode_id = episode_sample["episode_id"]
    high_instruction = episode_sample["instruction"]
    image_paths: List[str] = episode_sample["image_paths"]
    gt_subtask_list = episode_sample.get("subtask_list", [])

    trace_steps = []
    frame_ptr = 0
    total_frames = len(image_paths)
    # State machine for System‑2 rolling planner
    todo_subtasks: List[str] = []
    done_subtasks: List[str] = []
    eval_interval = 3  # Evaluate subtask progress every N frames to reduce VLM inference cost

    while (frame_ptr + 1) < total_frames and len(trace_steps) < max_loop:
        current_img_path = image_paths[frame_ptr]
        done_subtasks_text = ", ".join(done_subtasks)
        filled_prompt = prompt_template.format(
            instruction=high_instruction,
            done_subtasks=done_subtasks_text
        )

        step_record = {
            "episode_id": episode_id,
            "step_idx": len(trace_steps),
            "frame_ptr": frame_ptr,
            "current_image_path": current_img_path,
            "prompt": filled_prompt,
            "model_raw_output": None,
            "pred_subtasks": None,
            "parse_error": None,
            "gt_subtask_list": gt_subtask_list,
            "todo_subtasks": todo_subtasks.copy(),
            "done_subtasks": done_subtasks.copy(),
            "sim_executed_subtask": todo_subtasks[0] if len(todo_subtasks) > 0 else None
        }

        need_eval = False
        if len(todo_subtasks) == 0:
            # No pending subtask, must invoke planner to generate new subtask list
            need_eval = True
        else:
            # Evaluate subtask completion every eval_interval frames
            need_eval = (len(trace_steps) % eval_interval) == 0

        if need_eval:
            try:
                raw_out = backbone.infer_single(
                    image_path=current_img_path,
                    prompt_text=filled_prompt
                )
                step_record["model_raw_output"] = raw_out
                pred_subtasks = extract_json_array(raw_out)
                step_record["pred_subtasks"] = pred_subtasks

                if len(todo_subtasks) > 0:
                    active_sub = todo_subtasks[0]
                    # Skip completion judge if pred_subtasks is empty
                    if len(pred_subtasks) > 0:
                        # Semantic similarity based completion judge using embedding
                        emb_active = embedding_model.encode(active_sub, show_progress_bar=False)
                        emb_preds = embedding_model.encode(pred_subtasks, show_progress_bar=False)
                        similarities = cosine_similarity([emb_active], emb_preds)[0]
                        max_sim = float(np.max(similarities))

                        # If max similarity below threshold, treat active subtask as finished
                        if max_sim < sim_threshold:
                            todo_subtasks.pop(0)
                            done_subtasks.append(active_sub)
                else:
                    # todo is empty, accept VLM output as new todo list
                    todo_subtasks = pred_subtasks.copy()

            except json.JSONDecodeError as e:
                step_record["parse_error"] = f"JSONDecodeError: {str(e)}"
            except Exception as e:
                step_record["parse_error"] = f"RuntimeError: {str(e)}"
        else:
            # Skip VLM inference in non‑evaluation step, reuse current todo state
            step_record["pred_subtasks"] = todo_subtasks.copy()

        # Append record AFTER all field modification is finished, avoid reference side‑effect
        trace_steps.append(step_record)

        # Terminate condition: no remaining subtasks to execute
        if len(todo_subtasks) == 0:
            break

        # Advance frame pointer, simulate continuous video observation
        frame_ptr += 1
        if frame_ptr >= total_frames:
            frame_ptr = total_frames - 1

    return {
        "episode_id": episode_id,
        "high_level_instruction": high_instruction,
        "gt_subtask_list": gt_subtask_list,
        "total_available_frames": total_frames,
        "actual_roll_steps": len(trace_steps),
        "final_todo_subtasks": todo_subtasks.copy(),
        "final_done_subtasks": done_subtasks.copy(),
        "step_traces": trace_steps
    }


def main():
    parser = argparse.ArgumentParser(description="Rolling closed‑loop subtask planner offline replay")
    parser.add_argument("--dataset-jsonl", type=str, required=True,
                        help="Path to dataset_subtask_xxx.jsonl")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Base Qwen2‑VL model identifier or local path")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="LoRA adapter path; leave None for vanilla base model")
    parser.add_argument("--device", type=str, default="cpu", help="inference device: cpu/mps/cuda")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for rolling trace jsonl, e.g ./outputs/exp_xxx")
    parser.add_argument("--prompt-template", type=str,
                        default="Given image observation. High‑level task: {instruction}. Subtasks that have ALREADY BEEN FINISHED: [{done_subtasks}]. Rules you MUST strictly follow: 1. ONLY output subtasks that still NEED to be executed. 2. NEVER repeat any subtask from the already‑finished list. 3. If NO subtasks remain to do, output empty json array: []. 4. Output only pure json array, no extra explanation text.",
                        help="Prompt template, use {instruction} and {done_subtasks} as placeholders")
    parser.add_argument("--max-loop-per-episode", type=int, default=100,
                        help="Hard upper bound for each episode, prevent infinite loop")
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Only run N samples for quick test; None for full dataset")
    parser.add_argument("--sim-threshold", type=float, default=0.7,
                        help="Embedding cosine similarity threshold for subtask completion judge")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    trace_out_path = os.path.join(args.output_dir, "rolling_closed_trace.jsonl")
    failure_out_path = os.path.join(args.output_dir, "rolling_failure_cases.jsonl")

    print(f"Loading dataset from: {args.dataset_jsonl}")
    dataset = load_jsonl(args.dataset_jsonl)
    if args.limit_samples is not None:
        dataset = dataset[:args.limit_samples]
    print(f"Total episodes to run: {len(dataset)}")

    print(f"Initialize Qwen2VL backbone. model={args.model_name}, lora={args.lora_path}, device={args.device}")
    backbone = Qwen2VLBackbone(
        model_name=args.model_name,
        lora_path=args.lora_path,
        device=args.device
    )

    print("Initialize sentence embedding model: all‑MiniLM‑L6‑v2")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)

    all_episode_traces = []
    failure_cases = []
    for idx, sample in enumerate(dataset):
        ep_id = sample["episode_id"]
        print(f"Processing episode [{idx+1}/{len(dataset)}]: {ep_id}")
        try:
            ep_trace = run_episode_rolling_planner(
                backbone=backbone,
                embedding_model=embedding_model,
                episode_sample=sample,
                prompt_template=args.prompt_template,
                max_loop=args.max_loop_per_episode,
                sim_threshold=args.sim_threshold
            )
            all_episode_traces.append(ep_trace)
        except Exception as e:
            err_msg = f"Episode {ep_id} top‑level failure: {str(e)}"
            print(err_msg)
            failure_cases.append({"episode_id": ep_id, "error": err_msg})

    write_jsonl(all_episode_traces, trace_out_path)
    write_jsonl(failure_cases, failure_out_path)

    print("="*60)
    print(f"Finished rolling closed‑loop planner offline run.")
    print(f"Episode trace saved -> {trace_out_path}")
    print(f"Failure cases saved -> {failure_out_path}")
    print(f"Success episodes: {len(all_episode_traces)}, failure episodes: {len(failure_cases)}")


if __name__ == "__main__":
    main()
