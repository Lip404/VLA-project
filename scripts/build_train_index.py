import json
from pathlib import Path

OUTPUT_ROOT = Path("export_out")
TRAIN_JSONL_PATH = OUTPUT_ROOT / "train.jsonl"
VIEWS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right"
]

def main():
    f_out = open(TRAIN_JSONL_PATH, "w", encoding="utf-8")
    ep_dirs = sorted([p for p in OUTPUT_ROOT.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    for ep_dir in ep_dirs:
        ep_id = ep_dir.name
        anno_file = ep_dir / "anno.json"
        if not anno_file.exists():
            print(f"[WARN] {ep_id} missing anno.json, skip")
            continue
        with open(anno_file, "r", encoding="utf-8") as f:
            anno_data = json.load(f)
        instruction = anno_data["language_instruction"]

        for view in VIEWS:
            view_dir = ep_dir / view / "frames"
            if not view_dir.exists():
                print(f"[WARN] {ep_id} view {view} missing, skip this view")
                continue
            
            img_files = sorted(
                [p for p in view_dir.glob("*.jpg")],
                key=lambda x: int(x.stem.replace("frame_", ""))
            )
            if len(img_files) == 0:
                print(f"[WARN] {ep_id} {view} no frames, skip")
                continue
            img_abs_paths = [str(p.resolve()) for p in img_files]

            sample = {
                "episode_id": ep_id,
                "view_name": view,
                "image_paths": img_abs_paths,
                "language_instruction": instruction,
                "anno_path": str(anno_file.resolve())
            }
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    f_out.close()
    print(f"✅ Index built, saved to: {TRAIN_JSONL_PATH}")
    print(f"Total samples: {sum(1 for _ in open(TRAIN_JSONL_PATH))}")

if __name__ == "__main__":
    main()
