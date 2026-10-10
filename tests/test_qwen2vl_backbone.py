from vla.vlm_backbone.qwen2vl_backbone import load_vlm_model, run_vlm_inference
from PIL import Image

if __name__ == "__main__":
    print("Start loading Qwen2-VL-Instruct ...")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor, model = load_vlm_model(model_id, device="cpu")

    # Load image
    test_img = Image.open("test.jpg").convert("RGB")
    # Resize image to reduce patch count and memory consumption
    max_edge = 512
    w, h = test_img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        test_img = test_img.resize((int(w*scale), int(h*scale)))

    # Define prompt
    prompt = "Describe the content of this image in detail."

    print("Start inference ...")
    answer = run_vlm_inference(processor, model, test_img, prompt)
    print("==== Model Output ====")
    print(answer)