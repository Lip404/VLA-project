"""
End-to-end demo: video -> Qwen2VL -> structured JSON -> parse_video_behavior validation
Pipeline: frame extraction -> VLM inference -> structured parsing + schema validation
"""
import sys
from pathlib import Path
# Add project root to sys.path so that imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from vla.structured_planner import parse_video_behavior
from vla.vlm_backbone.qwen2vl_backbone import load_vlm_model, run_video_inference
from vla.schema import OBSERVE_VIDEO_SYSTEM_PROMPT
 
# -------------------------- Configuration --------------------------
VIDEO_PATH = Path(__file__).parent / "assets" / "test_video.mp4"
NUM_FRAMES = 5  # Number of uniformly sampled frames, adjust as needed
SYSTEM_PROMPT = OBSERVE_VIDEO_SYSTEM_PROMPT
# ------------------------------------------------------------
 
def main():
    print(f"[INFO] Load video: {VIDEO_PATH}")
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        return
 
    # 1. Load VLM model and processor
    model, processor = load_vlm_model("Qwen/Qwen2-VL-2B-Instruct")
    # 2. Run video inference
    llm_raw_output = run_video_inference(
        model=model,
        processor=processor,
        video_path=str(VIDEO_PATH),
        prompt=SYSTEM_PROMPT,
        num_frames=NUM_FRAMES
    )
 
    print("\n" + "="*60)
    print("==== RAW OUTPUT FROM Qwen2VL ====")
    print(llm_raw_output)
    print("="*60)
 
    # 3. Feed the raw output into our structured parser (core glue logic)
    result, err_msg = parse_video_behavior(llm_raw_output)
 
    print("\n==== PARSE & SCHEMA VALIDATION RESULT ====")
    if err_msg == "":
        print("SUCCESS: JSON parsed and schema validated!")
        # Print the structured result in a readable format
        print(result.model_dump_json(indent=2))
    else:
        print("FAILED: ")
        print(err_msg)
 
 
if __name__ == "__main__":
    main()