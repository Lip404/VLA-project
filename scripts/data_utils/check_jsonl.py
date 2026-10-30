import json
from PIL import Image

# Path to the jsonl annotation file
jsonl_path = "export_out/train.jsonl"

# Open and read all lines from the jsonl file
with open(jsonl_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Print total number of episodes
print(f"Total samples: {len(lines)}")
# Test with the first sample
sample = json.loads(lines[0])
print("===== Information of the first sample =====")
print(f"episode_id: {sample['episode_id']}")
print(f"view: {sample['view_name']}")
print(f"instruction: {sample['language_instruction']}")
print(f"frame count: {len(sample['image_paths'])}")
print(f"path of the first frame: {sample['image_paths'][0]}")

# Load and preview the first image frame
img = Image.open(sample["image_paths"][0])
img.show()
