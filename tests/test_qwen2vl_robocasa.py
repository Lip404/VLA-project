import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from vla.datasets import RoboCasaVLA

# ====== 1. Load model and processor ======
# Use 2B version, fits in 24GB Mac memory
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu", 
)

# ====== 2. Load our dataset, take 1 sample ======
# IMPORTANT: pass transform=None, because Qwen2VL processor will handle image resize/normalize itself.
# Do NOT use our old Resize+ToTensor here, it will conflict with the model.
ds = RoboCasaVLA("export_out/train.jsonl", transform=None, num_sample_frames=4, return_pil=True)
sample = ds[0]

# sample["images"] is already a list of PIL images (because return_pil=True)
pil_images = sample["images"]
instruction = sample["instruction"]

print(f"Loaded {len(pil_images)} frames, instruction: {instruction}")

# ====== 3. Build chat prompt ======
# Qwen2VL uses chat template: each <image> tag in text aligns with one PIL image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"} for _ in pil_images
        ] + [
            {
                "type": "text",
                "text": (
                    f"You are a robotic subtask annotator. Here are {len(pil_images)} sequential frames from robot execution.\n"
                    f"High-level task: {instruction}\n\n"
                    "Observe the robot's state, gripper and object position carefully. "
                    "Describe the ongoing subtask in a complete, detailed English sentence. "
                    "Do NOT shorten your description. Include the action, target object and current state. "
                    "Only output the subtask description, no extra comments."

                )
            }
        ]
    }
]

# Apply chat template: convert messages list to formatted text string
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# ====== 4. Process: text + images -> model input tensors ======
inputs = processor(
    text=[text],
    images=pil_images,
    return_tensors="pt",
    padding=True
).to(model.device)

print("Input keys:", inputs.keys())
print("input_ids shape:", inputs["input_ids"].shape)
print("pixel_values shape:", inputs["pixel_values"].shape)
print("image_grid_thw:", inputs["image_grid_thw"])

# ====== 5. Run generation ======
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

# Decode only the newly generated part (strip the prompt)
generated_text = processor.batch_decode(
    generated_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print("\n===== Model Output =====")
print(generated_text)
