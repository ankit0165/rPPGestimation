
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RPPGToPhysioDataset(Dataset):
    def __init__(self, root_dir, input_len=300):
        '''
        Expects directory structure:
        root_dir/
            sample1/
                rppg.npy
                ppg.npy
                ecg.npy
            sample2/
                ...
        '''
        self.root_dir = root_dir
        self.input_len = input_len
        self.samples = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        rppg = np.load(os.path.join(sample_dir, "rppg.npy"))
        ppg = np.load(os.path.join(sample_dir, "ppg.npy"))
        ecg = np.load(os.path.join(sample_dir, "ecg.npy"))

        # Clip or pad to fixed length
        def pad_or_crop(x):
            if len(x) > self.input_len:
                return x[:self.input_len]
            elif len(x) < self.input_len:
                return np.pad(x, (0, self.input_len - len(x)), mode='constant')
            return x

        rppg = pad_or_crop(rppg)
        ppg = pad_or_crop(ppg)
        ecg = pad_or_crop(ecg)

        rppg = torch.tensor(rppg, dtype=torch.float32).unsqueeze(0)  # (1, L)
        ppg = torch.tensor(ppg, dtype=torch.float32)                # (L,)
        ecg = torch.tensor(ecg, dtype=torch.float32)                # (L,)

        return rppg, ppg, ecg
