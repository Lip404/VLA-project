from PIL import Image
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# -------------------------- Configuration --------------------------
# Base multimodal model name
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# Path to the trained LoRA adapter (moved under weights folder)
LORA_PATH = "./weights/system2_subtask_lora"
# Replace this with your real image path from validation set
TEST_IMAGE_PATH = "/Users/qianliaopeng/VLA_local/VLA-project/exported_frames/episode_000000/observation.images.robot0_agentview_left/frames/frame_000000.jpg"
# High-level robot task instruction
HIGH_LEVEL_INSTRUCTION = "Pick the honey bottle from the counter and place it in the cabinet."
# -------------------------------------------------------------------

def main():
    # Load base Qwen2-VL model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    # Attach our trained LoRA for System2 subtask planning
    model = PeftModel.from_pretrained(model, LORA_PATH)
    # Load multimodal processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Chat template, must keep exactly the same format as training
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Given the scene image, finish this long‑horizon robot task: "
                        f"{HIGH_LEVEL_INSTRUCTION}. "
                        "Output the subtask list as a valid JSON array only."
                    )
                }
            ]
        }
    ]

    # Load input image
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    # Build prompt text with official chat template
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Preprocess image + text into model input tensors
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    ).to("cpu")

    # Inference, disable gradient computation to save memory
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9
        )

    # Decode token sequence to readable text
    model_output = processor.decode(output_tokens[0], skip_special_tokens=True)

    # Extract only assistant response content
    sections = model_output.split("assistant")
    if len(sections) >= 2:
        pure_json_str = sections[-1].strip()
    else:
        pure_json_str = model_output.strip()

    print("=" * 60)
    print("Full raw model output:")
    print("=" * 60)
    print(model_output)

    print("\n" + "=" * 60)
    print("Extracted subtask json string:")
    print("=" * 60)
    print(pure_json_str)

    # Validate json format
    try:
        subtask_list = json.loads(pure_json_str)
        print("\n✅ JSON parse success! Parsed subtask list:")
        print(subtask_list)
    except json.JSONDecodeError as e:
        print("\n❌ JSON parse failed:", e)


if __name__ == "__main__":
    main()