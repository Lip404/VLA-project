import argparse
import json
from pathlib import Path
from tqdm import tqdm
import math

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def load_vlm(model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cpu"):
    """
    Load Qwen2‑VL multimodal model and its processor.
    Args:
        model_name: huggingface model identifier
        device: target device (cpu / cuda / mps)
    Returns:
        processor, model
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to(device)
    return processor, model


def uniform_sample_frames(image_path_list: list, sample_num: int = 8):
    """
    Uniformly sample sample_num frames from full frame list, preserve temporal order.
    If total frames <= sample_num, return original list directly.
    Args:
        image_path_list: full list of image file paths for one episode
        sample_num: target number of frames to sample
    Returns:
        list of sampled image paths
    """
    total = len(image_path_list)
    if total <= sample_num:
        return image_path_list
    indices = [int(math.floor(i * total / sample_num)) for i in range(sample_num)]
    return [image_path_list[i] for i in indices]


def generate_subtask_sequence(processor, model, pil_images, instruction, device="cpu"):
    """
    Call Qwen2‑VL to generate ordered subtask list for one complete robot episode.
    Model is required to output pure JSON array of subtask sentences.
    Args:
        processor: vlm processor instance
        model: loaded Qwen2‑VL model
        pil_images: list of PIL images (sampled key frames from episode)
        instruction: high‑level robot task instruction string
        device: inference device
    Returns:
        subtask_list: python list of ordered subtask sentences
    Raises:
        ValueError: when model output cannot be parsed into valid json array
    """
    prompt_text = (
        "You are an observer for robot manipulation tasks. "
        "These images are sequential frames from one complete robot episode. "
        f"The high‑level task instruction: {instruction}.\n"
        "Output a JSON array only, which contains ordered, concise sentences for every sequential subtask the robot executes. "
        "Do not output any extra explanation, markdown, or text outside the json array."
        "Example: [\"robot approaches bottle\", \"robot grasps bottle\", \"robot lifts bottle\"]"
    )

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in pil_images],
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=pil_images,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False
        )
    raw_out = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    raw_out = raw_out.strip()
    start = raw_out.find("[")
    end = raw_out.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Model output cannot parse as array: {raw_out}")
    json_str = raw_out[start:end+1]
    subtask_list = json.loads(json_str)
    return subtask_list


def main():
    parser = argparse.ArgumentParser(description="Batch annotate subtask sequence for each robocasa episode in jsonl")
    parser.add_argument("--jsonl_path", type=str, required=True, help="original train jsonl path")
    parser.add_argument("--out_jsonl", type=str, required=True, help="output jsonl path with added subtask_list field")
    parser.add_argument("--sample_n_img", type=int, default=8, help="uniformly sample N frames per episode as vlm input")
    parser.add_argument("--limit", type=int, default=-1, help="debug: limit number of episodes to process, -1 for process all")
    parser.add_argument("--device", type=str, default="cpu", help="inference device")
    args = parser.parse_args()

    processor, model = load_vlm(device=args.device)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load all episode lines from source jsonl
    all_lines = []
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)

    end_idx = len(all_lines) if args.limit <= 0 else min(args.limit, len(all_lines))
    with open(out_path, "w", encoding="utf-8") as f_out:
        for line_idx in tqdm(range(end_idx), desc="Annotate episode subtask sequence"):
            try:
                row = json.loads(all_lines[line_idx])
                all_img_paths = row["image_paths"]
                sampled_paths = uniform_sample_frames(all_img_paths, args.sample_n_img)
                pil_imgs = [Image.open(p).convert("RGB") for p in sampled_paths]
                # Read language instruction from jsonl
                instr = row["language_instruction"]
                subtask_list = generate_subtask_sequence(processor, model, pil_imgs, instr, device=args.device)
                row["subtask_list"] = subtask_list
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()
            except Exception as e:
                print(f"\n[WARN] Episode line {line_idx} failed, skip. Error: {str(e)}")
                continue
    print(f"\nDone. Output saved to {out_path}")


if __name__ == "__main__":
    main()
