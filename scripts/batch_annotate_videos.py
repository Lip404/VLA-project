import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from typing import List

from vla.vlm_backbone.qwen2vl_backbone import load_vlm_model, run_video_inference
from vla.structured_planner.json_parser import parse_video_behavior
from vla.structured_planner.schema import VideoBehaviorRecord, OBSERVE_VIDEO_SYSTEM_PROMPT

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

            record = parse_res

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
