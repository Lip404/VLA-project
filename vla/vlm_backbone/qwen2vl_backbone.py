import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

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
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.05
        )
    
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
    
    for frame in frames:
        frame.thumbnail((512, 512))
        
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
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.05
        )
    result = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return result

    
class Qwen2VLBackbone:
    """
    Wrapper class for Qwen2‑VL with optional LoRA adapter.
    Reuses existing load_vlm_model and run_vlm_inference functions.
    """
    def __init__(self, model_name: str, lora_path: str = None, device: str = "cpu"):
        """
        Initialize base model, processor and optionally load LoRA adapter.
        :param model_name: base vlm model name or local path
        :param lora_path: path to lora adapter weights, set None for base model only
        :param device: inference device
        """
        self.model_name = model_name
        self.lora_path = lora_path
        self.device = device

        # Load base model and processor from existing helper
        self.model, self.processor = load_vlm_model(model_id=model_name, device=device)

        # Attach LoRA adapter if lora_path is provided
        if self.lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)

    def infer_single(self, image_path: str, prompt_text: str, max_new_tokens: int = 512, temperature: float = 0.1):
        """
        Run single image inference given image file path and prompt string.
        :param image_path: local disk path to input image
        :param prompt_text: user text prompt
        :param max_new_tokens: maximum new generated tokens
        :param temperature: sampling temperature
        :return: raw cleaned model output string
        """
        # Open image from disk
        image = Image.open(image_path).convert("RGB")

        # Call existing inference helper function
        output_str = run_vlm_inference(
            model=self.model,
            processor=self.processor,
            image=image,
            prompt_text=prompt_text
        )
        return output_str