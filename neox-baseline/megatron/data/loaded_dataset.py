import os
import time

import numpy as np
import torch

from megatron import mpu, print_rank_0


class LoadedDataset(torch.utils.data.Dataset):
    def __init__(self, data_prefix, num_batches=1000, batch_size=1024, seq_length=2049):
        self.dataset = np.memmap(
            data_prefix + ".bin",
            dtype=np.uint16,
            mode="r",
            order="C",
            shape=(num_batches * batch_size, seq_length)
        )

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {"text": torch.tensor(sample, dtype=torch.long)}