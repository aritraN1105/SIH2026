from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class SatelliteDataset(Dataset):
    """
    PyTorch dataset for satellite GeoTIFF images.

    Expected structure:

    data/
    └── processed/
        ├── low_resolution/
        └── high_resolution/
    """

    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)

        self.lr_files = sorted(self.lr_dir.glob("*.tif"))
        self.hr_files = sorted(self.hr_dir.glob("*.tif"))

        if len(self.lr_files) != len(self.hr_files):
            raise ValueError(
                f"Number of LR images ({len(self.lr_files)}) "
                f"does not match HR images ({len(self.hr_files)})"
            )

        if len(self.lr_files) == 0:
            raise ValueError("No .tif files found in the dataset directories.")

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, index):
        lr_path = self.lr_files[index]
        hr_path = self.hr_files[index]

        with rasterio.open(lr_path) as src:
            lr = src.read().astype(np.float32)

        with rasterio.open(hr_path) as src:
            hr = src.read().astype(np.float32)

        # Convert to PyTorch tensors
        lr = torch.from_numpy(lr)
        hr = torch.from_numpy(hr)

        return lr, hr