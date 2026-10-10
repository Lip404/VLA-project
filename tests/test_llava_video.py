from vla.vlm_backbone.llava_video import load_vlm_model, run_vlm_inference
from PIL import Image

if __name__ == "__main__":
    print("开始加载 Qwen2-VL-Instruct ...")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor, model = load_vlm_model(model_id, device="mps")

    # 读取图片
    test_img = Image.open("test.jpg").convert("RGB")
    # 缩小图片，降低patch数量，减少内存压力
    max_edge = 512
    w, h = test_img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        test_img = test_img.resize((int(w*scale), int(h*scale)))

    # ========== 定义prompt变量，解决 NameError ==========
    prompt = "详细描述这张图片里面有什么"

    print("开始推理 ...")
    answer = run_vlm_inference(processor, model, test_img, prompt)
    print("====模型输出====")
    print(answer)