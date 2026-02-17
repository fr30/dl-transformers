import numpy as np
import os
import torch


class TinyShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer_name="shakespeare_tokenizer", dataset_path="data/tinyshakespeare"):
        super().__init__()
        if split == "train" or split == "val" or split == "test":
            path = os.path.join(dataset_path, f"{tokenizer_name}_{split}")
            # self.offsets = torch.load(f"{path}_offsets.pt")
            self.offsets = np.load(f"{path}_offsets.npy")
            self.tokens = torch.from_file(f"{path}_tokens.bin", size=self.offsets[-1], dtype=torch.int16).to(torch.int64)
        else:
            raise ValueError("Wrong split")
            
    def __getitem__(self, index):
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return self.tokens[start: end][:-1], self.tokens[start: end][1:] 
    
    def __len__(self):
        return len(self.offsets) - 1
    
class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer_name="tinystories_tokenizer", dataset_path="data/tinystories"):
        super().__init__()
        if split == "train" or split == "val" or split == "test":
            path = os.path.join(dataset_path, f"{tokenizer_name}_{split}")
            self.tokens = torch.from_file(f"{path}_tokens.bin", size=self.offsets[-1], dtype=torch.int64)
            self.offsets = torch.load(f"{path}_offsets.pt")
        else:
            raise ValueError("Wrong split")
            
    def __getitem__(self, index):
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return self.tokens[start: end][:-1], self.labels[start: end][1:] 
    
    def __len__(self):
        return len(self.offsets) - 1
    
class OpenWebTextDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer_name="openwebtext_tokenizer", dataset_path="data/openwebtext"):
        super().__init__()
        if split == "train" or split == "val" or split == "test":
            path = os.path.join(dataset_path, f"{tokenizer_name}_{split}")
            self.tokens = torch.from_file(f"{path}_tokens.bin", size=self.offsets[-1], dtype=torch.int64)
            self.offsets = torch.load(f"{path}_offsets.pt")
        else:
            raise ValueError("Wrong split")
            
    def __getitem__(self, index):
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return self.tokens[start: end][:-1], self.labels[start: end][1:] 
    
    def __len__(self):
        return len(self.offsets) - 1
    