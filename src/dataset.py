import numpy as np
import os
import torch


class TinyShakespeareDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        block_size=1024,
        tokenizer_name="shakespeare_tokenizer",
        dataset_path="data/tinyshakespeare",
    ):
        super().__init__()
        self.block_size = block_size

        if split in ["train", "val", "test"]:
            path = os.path.join(dataset_path, f"{tokenizer_name}_{split}")

            offsets = np.load(f"{path}_offsets.npy")
            total_tokens = offsets[-1]
            self.tokens = torch.from_file(
                f"{path}_tokens.bin", shared=False, size=total_tokens, dtype=torch.int16
            ).to(torch.int64)
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        return (self.tokens.size(0) - 1) // self.block_size

    def __getitem__(self, index):
        start = index * self.block_size
        end = start + self.block_size + 1

        chunk = self.tokens[start:end]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y


class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        block_size=1024,
        tokenizer_name="tinystories_tokenizer",
        dataset_path="data/tinystories",
    ):
        super().__init__()
        self.block_size = block_size

        if split in ["train", "val", "test"]:
            path = os.path.join(dataset_path, f"{tokenizer_name}_{split}")

            offsets = np.load(f"{path}_offsets.npy")
            total_tokens = offsets[-1]
            self.tokens = torch.from_file(
                f"{path}_tokens.bin", shared=False, size=total_tokens, dtype=torch.int16
            ).to(torch.int64)
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        return (self.tokens.size(0) - 1) // self.block_size

    def __getitem__(self, index):
        start = index * self.block_size
        end = start + self.block_size + 1

        chunk = self.tokens[start:end]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y


class OpenWebTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        block_size=1024,
        tokenizer_name="openwebtext16k_tokenizer",
        dataset_path="data/openwebtext",
    ):
        super().__init__()
        self.block_size = block_size

        if split in ["train", "val", "test"]:
            path = os.path.join(dataset_path, f"{tokenizer_name}_{split}")

            offsets = np.load(f"{path}_offsets.npy")
            total_tokens = offsets[-1]
            self.tokens = torch.from_file(
                f"{path}_tokens.bin", shared=False, size=total_tokens, dtype=torch.int16
            ).to(torch.int64)
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        return (self.tokens.size(0) - 1) // self.block_size

    def __getitem__(self, index):
        start = index * self.block_size
        end = start + self.block_size + 1

        chunk = self.tokens[start:end]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y
