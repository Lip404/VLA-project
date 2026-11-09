# scripts/eval/val_batch_infer_system2.py
import json
import os
from pathlib import Path
from tqdm import tqdm
from vla.vlm_backbone.qwen2vl_backbone import Qwen2VLBackbone
from vla.structured_planner.model_output_json_parser import extract_json_array

# ===================== Configuration, modify as needed =====================
VAL_DATA_PATH = "./exported_frames/dataset_subtask_val.jsonl"  # Path of validation dataset
LORA_WEIGHT_PATH = "./weights/system2_subtask_lora"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_RESULT_PATH = "./outputs/exp_system2_lora_v1/val_system2_infer_result.jsonl"
FAILURE_LOG_PATH = "./outputs/exp_system2_lora_v1/val_failure_cases.jsonl"
DEVICE = "cpu"
# ==========================================================================

def main():
    # Create output directory if it does not exist
    Path("./outputs").mkdir(exist_ok=True)

    # Load System2 LoRA‑based subtask planner
    print("Loading System2 LoRA planner...")
    planner = Qwen2VLBackbone(
        model_name=MODEL_NAME,
        lora_path=LORA_WEIGHT_PATH,
        device=DEVICE
    )

    total = 0
    success_parse = 0
    infer_result_list = []
    failure_case_list = []

    # Read validation dataset file
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Iterate over all validation samples
    for line in tqdm(lines, desc="Running System2 val inference"):
        total += 1
        item = json.loads(line.strip())
        high_level_instruction = item["instruction"]
        image_path = item["image_paths"][0]
        prompt_text = (
            "Given the scene image, finish this long‑horizon robot task: "
            f"{high_level_instruction}. "
            "Output the subtask list as a valid JSON array only."
        )

        try:
            raw_output = planner.infer_single(image_path, prompt_text, max_new_tokens=512, temperature=0.1)
            subtask_list = extract_json_array(raw_output)
            success_parse += 1
            infer_result_list.append({
                "episode_id": item.get("episode_id"),
                "instruction": high_level_instruction,
                "gt_subtask": item["subtask_list"],
                "pred_subtask": subtask_list,
                "raw_output": raw_output,
                "parse_status": "ok"
            })
        except Exception as e:
            failure_case_list.append({
                "episode_id": item.get("episode_id"),
                "instruction": high_level_instruction,
                "gt_subtask": item["subtask_list"],
                "raw_output": raw_output if "raw_output" in locals() else None,
                "error": str(e),
                "parse_status": "fail"
            })

    # Save successful inference results to jsonl
    with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as f:
        for row in infer_result_list:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save failure cases for further debugging
    with open(FAILURE_LOG_PATH, "w", encoding="utf-8") as f:
        for row in failure_case_list:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print final evaluation statistics
    print("\n" + "="*50)
    print(f"Total val samples: {total}")
    print(f"JSON parse success: {success_parse}")
    print(f"Parse success rate: {success_parse / total * 100:.2f} %")
    print(f"Result saved to: {OUTPUT_RESULT_PATH}")
    print(f"Failure cases saved to: {FAILURE_LOG_PATH}")
    print("="*50)


if __name__ == "__main__":
    main()
