import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os

class PickledDataset(Dataset):
    data_path: str
    pickle_files: List[str]
    windows: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int]]

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.windows = []

    def load_block(self, block_path: str):
        print('Loading block: ' + block_path)
        self.windows.extend(torch.load(block_path))
        print('Done loading block: ' + block_path)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int]:
        return self.windows[index]