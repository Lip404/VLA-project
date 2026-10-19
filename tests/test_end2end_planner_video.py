"""
End-to-end demo: video -> Qwen2VL -> structured JSON -> parse_video_behavior validation
Pipeline: frame extraction -> VLM inference -> structured parsing + schema validation
"""
import sys
from pathlib import Path
# Add project root to sys.path so that imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from vla.structured_planner import parse_video_behavior
from vla.vlm_backbone.qwen2vl_backbone import Qwen2VLBackbone
 
# -------------------------- Configuration --------------------------
VIDEO_PATH = Path(__file__).parent / "assets" / "test_video.mp4"
NUM_FRAMES = 5  # Number of uniformly sampled frames, adjust as needed
SYSTEM_PROMPT = """
You are a video behavior annotator.
Analyze the sequence of image frames to understand what the robot or agent is doing.
You MUST output ONLY a markdown JSON code block wrapped by ```json and ```.
Do NOT add any extra explanation, commentary, or natural language outside the code block.
 
Allowed action list: ["grasp", "look", "pull", "move_forward"]
You can only pick action from this list, never invent new action names.
 
The JSON schema:
{
  "video_summary": str, # one sentence summary of what happens in the video
  "objects_in_scene": list[str], # all important objects visible in this video
  "action_sequence": [
    {
      "action": str,
      "target": str | null,
      "target_object": str | null,
      "confidence": float between 0 and 1
    }
  ]
}
 
Valid example output:
```json
{
  "video_summary": "The robot moves forward toward the cube block",
  "objects_in_scene": ["cube block", "robot"],
  "action_sequence": [
    {
      "action": "move_forward",
      "target_object": "cube block",
      "confidence": 0.92
    }
  ]
}
```
"""
# ------------------------------------------------------------
 
def main():
    print(f"[INFO] Load video: {VIDEO_PATH}")
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        return
 
    # 1. Initialize the VLM backbone
    vlm = Qwen2VLBackbone()
    # 2. Run inference: frame extraction + multi-frame inference are encapsulated inside
    llm_raw_output = vlm.infer_video(
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