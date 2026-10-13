# tests/test_qwen2vl_backbone.py
from vla.vlm_backbone.qwen2vl_backbone import load_vlm_model, run_image_inference
from PIL import Image


def main():
    # ========== Configurations ==========
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    image_path = "tests/assets/test.jpg"
    prompt = "Describe the content of this image in detail."
    # =====================================

    print("===== Loading Qwen2-VL model and processor =====")
    model, processor = load_vlm_model(model_id)
    print("Model loaded successfully.\n")

    print("===== Load test image =====")
    image = Image.open(image_path).convert("RGB")

    print("===== Start image inference =====")
    result = run_image_inference(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt
    )

    print("\n===== Model Output =====")
    print(result)


if __name__ == "__main__":
    main()
