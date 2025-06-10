
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob

class PurePPGRPPGDataset(Dataset):
    def __init__(self, root_dir, transform=None, output_len=300):
        self.root_dir = root_dir
        self.samples = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.transform = transform
        self.output_len = output_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec_dir = os.path.join(self.root_dir, self.samples[idx])
        frame_dir = os.path.join(rec_dir, self.samples[idx])
        json_path = os.path.join(rec_dir, self.samples[idx] + ".json")

        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))[:self.output_len]
        frames = [self.transform(Image.open(fp)) for fp in frame_paths]
        video_tensor = torch.stack(frames)

        with open(json_path, "r") as f:
            data = json.load(f)
            rppg_value = data["Value"]["waveform"]
            rppg_signal = np.full((self.output_len,), rppg_value, dtype=np.float32)

        return video_tensor, torch.tensor(rppg_signal)
