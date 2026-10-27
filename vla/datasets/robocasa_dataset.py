import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class RoboCasaVLA(Dataset):
    def __init__(self, jsonl_path, transform=None, num_sample_frames=None, return_pil=False):
        """
        Args:
            jsonl_path: file path to the train.jsonl file
            transform: torchvision image preprocessing transform
            num_sample_frames: fixed number of frames to sample; None = return all frames
        """
        self.jsonl_path = Path(jsonl_path)
        self.transform = transform
        self.num_sample_frames = num_sample_frames
        self.return_pil = return_pil

        self.samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

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
        img_paths = self._sample_frames(item["image_paths"])

        img_list = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            img_list.append(img)

        if self.return_pil:
            out_images = img_list
        else:
            out_images = torch.stack(img_list, dim=0)

        ret = {
            "images": out_images,  # (T, C, H, W)
            "instruction": item["language_instruction"],
            "episode_id": item["episode_id"],
            "view_name": item["view_name"],
            "anno_path": item["anno_path"]
        }
        return ret


if __name__ == "__main__":
    # self-test: run from project root
    from torchvision import transforms
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    ds = RoboCasaVLA("export_out/train.jsonl", transform=test_tf, num_sample_frames=8)
    print("dataset len:", len(ds))
    batch = ds[0]
    print("image tensor shape (T,C,H,W):", batch["images"].shape)
    print("instruction:", batch["instruction"])
