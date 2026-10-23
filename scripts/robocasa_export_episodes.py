import os
import cv2
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

# ===================== CONFIG =====================
BASE_VIDEO_PATH = Path("/Users/qianliaopeng/VLA_local/VLA-project/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot/videos/chunk-000")
BASE_PARQUET_PATH = Path("/Users/qianliaopeng/VLA_local/VLA-project/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot/data/chunk-000")
BASE_EXTRAS_PATH = Path("/Users/qianliaopeng/VLA_local/VLA-project/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot/extras")

OUTPUT_ROOT = Path("export_out")
VIEWS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right"
]
# ===================================================

def sample_frames(total_frames: int, fps: float) -> List[int]:
    """
    Fixed interval sampling: save 1 frame every sample_step frames
    sample_step = 5: for 30fps video → effective 6 fps
    """
    sample_step = 5
    indices = list(range(0, total_frames, sample_step))
    return indices

def read_episode_parquet(parquet_file: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_file)
    return df

def export_episode(ep_name: str):
    ep_id = ep_name.replace(".mp4", "")
    parquet_fp = BASE_PARQUET_PATH / f"{ep_id}.parquet"
    ep_meta_fp = BASE_EXTRAS_PATH / ep_id / "ep_meta.json"

    if not parquet_fp.exists():
        print(f"[WARN] {ep_id} parquet not found, skip")
        return

    df = read_episode_parquet(parquet_fp)
    ep_out_dir = OUTPUT_ROOT / ep_id
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    ep_meta = None
    lang_instruction = None
    if ep_meta_fp.exists():
        try:
            with open(ep_meta_fp, "r", encoding="utf-8") as f:
                ep_meta = json.load(f)
            lang_instruction = ep_meta.get("lang")
        except Exception as e:
            print(f"[WARN] load ep_meta failed {ep_meta_fp}, err: {e}")
    else:
        print(f"[INFO] ep_meta missing: {ep_meta_fp}")

    anno_payload = {
        "episode_id": ep_id,
        "views": VIEWS,
        "task_name_id": int(df["annotation.human.task_name"].iloc[0]) if "annotation.human.task_name" in df.columns else None,
        "task_desc_id": int(df["annotation.human.task_description"].iloc[0]) if "annotation.human.task_description" in df.columns else None,
        "language_instruction": lang_instruction,
        "ep_meta": ep_meta,
        "raw_parquet_columns": list(df.columns)
    }

    for view in VIEWS:
        video_fp = BASE_VIDEO_PATH / view / ep_name
        if not video_fp.exists():
            print(f"[WARN] {ep_id} view {view} video missing, skip this view")
            continue
        view_out = ep_out_dir / view / "frames"
        view_out.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_fp))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = sample_frames(total_frames, fps)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in sample_indices:
                out_img_path = view_out / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_img_path), frame)
            frame_idx += 1
        cap.release()

    with open(ep_out_dir / "anno.json", "w", encoding="utf-8") as f:
        json.dump(anno_payload, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Exported {ep_id} | instruction: {lang_instruction}")

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ref_view = VIEWS[0]
    video_files = sorted([
        f for f in os.listdir(BASE_VIDEO_PATH / ref_view)
        if f.startswith("episode_") and f.endswith(".mp4")
    ])
    print(f"Found {len(video_files)} episodes to export")
    for fname in video_files[:5]:
        export_episode(fname)
    print("✅ All export finished!")

if __name__ == "__main__":
    main()
