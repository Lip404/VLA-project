"""
Rolling closed‑loop subtask planner entry.
Offline simulation mode: replay existing dataset sequential frames to simulate robot observation feedback loop.
Future: replace observation provider for real‑simulation online running.
"""
import argparse
import json
import os
from typing import List, Dict, Any

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
    episode_sample: Dict[str, Any],
    prompt_template: str,
    max_loop: int = 100
) -> Dict[str, Any]:
    """
    Run rolling closed‑loop for single episode.
    Offline: iterate pre‑stored sequential image_paths as new observations after simulated execution.
    :param backbone: loaded VLM backbone with/without LoRA
    :param episode_sample: one sample from dataset_subtask jsonl
    :param prompt_template: prompt template fed to planner
    :param max_loop: hard upper bound to avoid infinite loop
    :return: episode trace dict including every step inference result
    """
    episode_id = episode_sample["episode_id"]
    high_instruction = episode_sample["instruction"]
    image_paths: List[str] = episode_sample["image_paths"]
    gt_subtask_list = episode_sample.get("subtask_list", [])

    trace_steps = []
    frame_ptr = 0
    total_frames = len(image_paths)

    while frame_ptr < total_frames and len(trace_steps) < max_loop:
        current_img_path = image_paths[frame_ptr]
        filled_prompt = prompt_template.format(instruction=high_instruction)

        step_record = {
            "episode_id": episode_id,
            "step_idx": len(trace_steps),
            "frame_ptr": frame_ptr,
            "current_image_path": current_img_path,
            "prompt": filled_prompt,
            "model_raw_output": None,
            "pred_subtasks": None,
            "parse_error": None,
            "gt_subtask_list": gt_subtask_list
        }

        try:
            raw_out = backbone.infer_single(
                image_path=current_img_path,
                prompt_text=filled_prompt
            )
            step_record["model_raw_output"] = raw_out
            pred_subtasks = extract_json_array(raw_out)
            step_record["pred_subtasks"] = pred_subtasks

        except json.JSONDecodeError as e:
            step_record["parse_error"] = f"JSONDecodeError: {str(e)}"
        except Exception as e:
            step_record["parse_error"] = f"RuntimeError: {str(e)}"

        trace_steps.append(step_record)
        # Offline simulation: simulate "execute action → get next observation"
        frame_ptr += 1

    return {
        "episode_id": episode_id,
        "high_level_instruction": high_instruction,
        "gt_subtask_list": gt_subtask_list,
        "total_available_frames": total_frames,
        "actual_roll_steps": len(trace_steps),
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
                        default="Given image observation, decompose the task: {instruction}. Output subtasks as pure json array.",
                        help="Prompt template, use {instruction} as placeholder for high‑level instruction")
    parser.add_argument("--max-loop-per-episode", type=int, default=100,
                        help="Hard upper loop limit for each episode, prevent infinite loop")
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Only run N samples for quick test; None for full dataset")

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

    all_episode_traces = []
    failure_cases = []

    for idx, sample in enumerate(dataset):
        ep_id = sample["episode_id"]
        print(f"Processing episode [{idx+1}/{len(dataset)}]: {ep_id}")
        try:
            ep_trace = run_episode_rolling_planner(
                backbone=backbone,
                episode_sample=sample,
                prompt_template=args.prompt_template,
                max_loop=args.max_loop_per_episode
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
