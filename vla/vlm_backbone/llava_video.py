import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

def load_vlm_model(model_id: str, device: str = "cpu"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager"
    )
    return processor, model

def run_vlm_inference(processor, model, image, prompt_text:str):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to("cpu", dtype=torch.float32)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    result = processor.decode(output[0], skip_special_tokens=True)
    return result
