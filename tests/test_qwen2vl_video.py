# tests/test_qwen2vl_video.py
from vla.vlm_backbone.qwen2vl_backbone import load_vlm_model, run_video_inference


def main():
    # ========== Configurations ==========
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    video_path = "tests/assets/test_video.mp4"
    prompt = "Describe what is happening in this video in detail."
    num_frames = 8
    # =====================================

    print("===== Loading Qwen2-VL model and processor =====")
    model, processor = load_vlm_model(model_id)
    print("Model loaded successfully.\n")

    print("===== Start video inference =====")
    result = run_video_inference(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt=prompt,
        num_frames=num_frames
    )

    print("\n===== Model Output =====")
    print(result)


if __name__ == "__main__":
    main()
