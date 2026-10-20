import json
import logging
from pathlib import Path
from typing import List

from vla.vlm_backbone.qwen2vl_backbone import load_vlm_model, run_video_inference
from vla.structured_planner.json_parser import parse_video_behavior
from vla.structured_planner.schema import VideoBehaviorRecord

# ===================== Configuration Section =====================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cpu"
VIDEO_INPUT_DIR = "./tests/assets/batch_videos"
ANNOTATION_OUTPUT_DIR = "./annotations"
NUM_FRAMES = 8

REJECT_KEYWORDS = [
    "occlusion",
    "blocked",
    "unclear",
    "cannot observe",
    "human-only action",
    "not reproducible",
    "confusing",
    "ambiguous",
]

OBSERVE_VIDEO_SYSTEM_PROMPT = """
You are a video behavior observer for robot imitation learning.
You will receive several frames sampled from a continuous video.
Your job is to observe what actions HAPPENED in this video, in chronological order.

Rules:
1. ONLY output valid JSON. NO extra natural language explanation, no comments outside JSON. Do NOT wrap JSON inside markdown ```json code block.
2. Action must be chosen strictly from this fixed set: reach, grasp, move, release, rotate, push, pull, move_forward
3. Record actions in the SAME TIME ORDER they appear in the video.
4. Fill frame range if you can infer when this action occurs.
5. Follow exactly this schema:
{
  "video_summary": "short one-sentence summary of this video clip",
  "objects_in_scene": ["list of unique distinct visible object names, each object appears only once"],
  "action_sequence": [
    {
      "action": "action_name",
      "target_object": "object name",
      "start_frame": null or integer,
      "end_frame": null or integer,
      "spatial_change": "describe object position change"
    }
  ],
  "notes": null
}
If you cannot identify any manipulation action, leave action_sequence empty and put your reason in notes.

IMPORTANT CONSTRAINTS YOU MUST FOLLOW:
1. NEVER infinitely repeat the same object name. List each distinct object ONLY ONCE in objects_in_scene.
2. Limit the length of action_sequence to at most 4 action steps. Do NOT generate unlimited repeated actions.
3. You must produce a fully complete, well-formed JSON. All quotes, brackets and braces must be properly closed.
4. Do NOT leave sentences, strings or JSON structures unfinished. Stop generation right after finishing the complete JSON object.
"""
# =================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def is_sample_valid(record: VideoBehaviorRecord) -> tuple[bool, str]:
    """
    Simple filtering function: judge sample validity based on the notes field.
    return: (is_valid: bool, reason: str)
    """
    notes = record.notes or ""
    notes_lower = notes.lower()

    for kw in REJECT_KEYWORDS:
        if kw.lower() in notes_lower:
            return False, f"notes contains reject keyword: {kw}"

    if len(record.action_sequence) == 0:
        return False, "action_sequence is empty, no valid actions detected"

    return True, "ok"


def batch_annotate(video_dir: str, out_dir: str):
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    suffix_whitelist = {".mp4", ".mov", ".avi", ".mkv"}
    video_paths: List[Path] = [
        f for f in video_dir.iterdir() if f.suffix.lower() in suffix_whitelist
    ]

    if len(video_paths) == 0:
        logger.warning("No supported video files found in %s", video_dir)
        return

    logger.info("Total videos found for annotation: %d", len(video_paths))

    # Load model and processor ONCE before looping videos
    logger.info(f"Loading VLM model: {MODEL_ID} on {DEVICE}")
    model, processor = load_vlm_model(model_id=MODEL_ID, device=DEVICE)
    logger.info("Model loaded successfully.")

    for idx, vid_path in enumerate(video_paths, start=1):
        logger.info("===== [%d/%d] Processing: %s =====", idx, len(video_paths), vid_path.name)

        try:
            raw_output = run_video_inference(
                model=model,
                processor=processor,
                video_path=str(vid_path),
                prompt=OBSERVE_VIDEO_SYSTEM_PROMPT,
                num_frames=NUM_FRAMES
            )
            logger.info(f"Raw model output:\n{raw_output}")

            parse_res, err_msg = parse_video_behavior(raw_output)
            if err_msg:
                logger.error("Parse failed for %s: %s", vid_path.name, err_msg)
                continue

            record = VideoBehaviorRecord(**parse_res)

            valid, filter_msg = is_sample_valid(record)
            logger.info("Sample check result: valid=%s, reason=%s", valid, filter_msg)

            anno_data = record.model_dump()
            anno_data["_is_valid_sample"] = valid
            anno_data["_filter_message"] = filter_msg
            anno_data["_source_video"] = str(vid_path)

            out_json_name = f"{vid_path.stem}_anno.json"
            out_json_path = out_dir / out_json_name

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(anno_data, f, indent=2)

            logger.info("Annotation saved -> %s", out_json_path)

        except Exception as e:
            logger.exception("Exception occurred while processing %s: %s", vid_path.name, str(e))
            continue


if __name__ == "__main__":
    batch_annotate(VIDEO_INPUT_DIR, ANNOTATION_OUTPUT_DIR)
