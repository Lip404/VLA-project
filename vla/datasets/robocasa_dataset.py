import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class RoboCasaVLA(Dataset):
    def __init__(self, jsonl_path, transform=None, num_sample_frames=None, return_pil=False, use_sliding_window: bool = False, window_size: int = 8, stride: int = 4,):
        """
        Args:
            jsonl_path: file path to the train.jsonl file
            transform: torchvision image preprocessing transform
            num_sample_frames: fixed number of frames to sample; None = return all frames
            return_pil: return PIL image list if True, else torch tensor
            use_sliding_window: whether enable sliding‑window slicing for episodes
                When True: each dataset item is a small frame window, requires jsonl with "subtask_list" field.
                When False: keep original behavior, one item corresponds to one full episode.
            window_size: length of each sliding window (only valid when use_sliding_window=True)
            stride: step size for sliding window (only valid when use_sliding_window=True)
        """
        self.jsonl_path = Path(jsonl_path)
        self.transform = transform
        self.num_sample_frames = num_sample_frames
        self.return_pil = return_pil
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.stride = stride

        self.raw_episodes = []
        self.samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.raw_episodes.append(json.loads(line))

        if self.use_sliding_window:
            # Pre-generate all window-based training samples
            self.samples = []
            for ep in self.raw_episodes:
                image_paths = ep["image_paths"]
                total_frames = len(image_paths)
                #print(f"[DEBUG] episode {ep['episode_id']}, total_frames={total_frames}, window_size={self.window_size}")

                if total_frames < self.window_size:
                    print(f"[DEBUG] skip this episode: frame count less than window size")
                    continue

                # Require subtask_list field when sliding window is enabled
                if "subtask_list" not in ep:
                    print(f"[DEBUG] skip this episode: missing subtask_list key")
                    continue

                subtask_groups = ep["subtask_list"]
                #print(f"[DEBUG] subtask_groups count: {len(subtask_groups)}")
                lang_instr = ep["language_instruction"]
                episode_id = ep["episode_id"]

                start = 0
                while start + self.window_size <= total_frames:
                    end = start + self.window_size
                    frame_slice = image_paths[start:end]

                    # simple mapping: map window position to subtask group index
                    group_idx = min(int(start / (total_frames / len(subtask_groups))), len(subtask_groups) - 1)
                    current_subtasks = subtask_groups[group_idx]

                    self.samples.append({
                        "frame_paths": frame_slice,
                        "language_instruction": lang_instr,
                        "episode_id": episode_id,
                        "subtask_labels": current_subtasks
                    })
                    #print(f"[DEBUG] add window start={start}, group_idx={group_idx}, subtask={current_subtasks}")
                    start += self.stride

        else:
            # keep original behavior: samples = full episodes
            self.samples = self.raw_episodes

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, path_list):
        if self.num_sample_frames is None:
            return path_list
        # uniform sample frames
        total = len(path_list)
        indices = torch.linspace(0, total - 1, self.num_sample_frames).long()
        return [path_list[i] for i in indices]

    def __getitem__(self, idx):
        item = self.samples[idx]

        if self.use_sliding_window:
            # item is pre‑sliced window sample
            img_paths = item["frame_paths"]
            instruction = item["language_instruction"]
            episode_id = item["episode_id"]
            subtask_labels = item["subtask_labels"]
        else:
            # original full‑episode logic
            img_paths = self._sample_frames(item["image_paths"])
            instruction = item["language_instruction"]
            episode_id = item["episode_id"]
            subtask_labels = item.get("subtask_list", None)  # may be None for original jsonl
            view_name = item["view_name"]
            anno_path = item["anno_path"]

        img_list = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            if self.transform is not None and not self.return_pil:
                img = self.transform(img)
            img_list.append(img)

        if self.return_pil:
            out_images = img_list
        else:
            out_images = torch.stack(img_list, dim=0)

        ret = {
            "images": out_images,  # (T, C, H, W)
            "instruction": instruction,
            "episode_id": episode_id,
            "subtask_labels": subtask_labels,
        }
        # keep old keys only in non‑sliding‑window mode for backward compatibility
        if not self.use_sliding_window:
            ret["view_name"] = view_name
            ret["anno_path"] = anno_path
        return ret


if __name__ == "__main__":
    # self-test: run from project root
    from torchvision import transforms
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Test sliding window mode for new jsonl with subtask_list field
    print("=== Test sliding window mode (jsonl with subtask_list) ===")
    ds_win = RoboCasaVLA(
        "export_out/train_episode_with_subtask_list.jsonl",
        transform=None,
        use_sliding_window=True,
        window_size=8,
        stride=4,
        return_pil=True
    )
    print("window-based dataset len:", len(ds_win))
    if len(ds_win) > 0:
        win_sample = ds_win[0]
        print("num frames in window:", len(win_sample["images"]))
        print("instruction:", win_sample["instruction"])
        print("subtask_labels for this window:", win_sample["subtask_labels"])
    else:
        print("Warning: no window samples loaded, check jsonl path or frame count")
