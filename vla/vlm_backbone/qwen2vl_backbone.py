import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

def load_vlm_model(model_id: str, device: str = "cpu"):
    """
    Load Qwen2-VL processor and model.
    :param model_id: HuggingFace model identifier
    :param device: target device for model inference
    :return: processor, model
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager"
    )
    return processor, model

def run_vlm_inference(processor, model, image, prompt_text:str):
    """
    Run single-image VLM inference.
    :param processor: vision-language processor
    :param model: loaded VLM model
    :param image: PIL input image
    :param prompt_text: user instruction text
    :return: cleaned model response (without chat template)
    """
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
    
    # Decode raw model output
    full_result = processor.decode(output[0], skip_special_tokens=True)

    # Strip chat template, only keep assistant's response
    if "assistant" in full_result:
        result = full_result.split("assistant")[-1].strip()
    else:
        result = full_result

    return result