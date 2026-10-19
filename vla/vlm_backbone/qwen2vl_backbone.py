import torch
import cv2
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
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    return model, processor

def run_vlm_inference(model, processor, image, prompt_text:str):
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



def sample_video_frames(video_path: str, num_frames: int = 8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Cannot read video or video has zero frames")
    
    sample_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frame_list = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # OpenCV loads frames in BGR order; Qwen2-VL expects RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frame_list.append(pil_img)
    cap.release()
    return frame_list

def run_video_inference(model, processor, video_path: str, prompt: str, num_frames=8):
    # 1. Sample frames from the input video
    frames = sample_video_frames(video_path, num_frames)
    
    content = []
    for _ in frames:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Feed video frames into processor, use videos argument for temporal video input
    inputs = processor(text=[text], images=frames, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
    result = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return result
