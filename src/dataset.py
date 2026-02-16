import os
import torch


class TinyShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer_name="shakespeare_tokenizer", dataset_path="data/tinyshakespeare"):
        super().__init__()
        if split == "train" or split == "val" or split == "test":
            path = os.path.join(dataset_path, f"{tokenizer_name}_{split}")
            self.tokens = torch.load(f"{path}_tokens.pt")
            self.labels = torch.load(f"{path}_labels.pt")
            self.offsets = torch.load(f"{path}_offsets.pt")
        else:
            raise ValueError("Wrong split")
            
    def __getitem__(self, index):
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return self.tokens[start: end], self.labels[start: end] 
    
    def __len__(self):
        return len(self.offsets) - 1
    
class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer_name="tinystories_tokenizer", dataset_path="data/tinystories"):
        super().__init__()
        if split == "train" or split == "val" or split == "test":
            path = os.path.join(dataset_path, f"{tokenizer_name}_{split}")
            self.tokens = torch.load(f"{path}_tokens.pt")
            self.labels = torch.load(f"{path}_labels.pt")
            self.offsets = torch.load(f"{path}_offsets.pt")
        else:
            raise ValueError("Wrong split")
            
    def __getitem__(self, index):
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return self.tokens[start: end], self.labels[start: end] 
    
    def __len__(self):
        return len(self.offsets) - 1
    
class OpenWebTextDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer_name="openwebtext_tokenizer", dataset_path="data/openwebtext"):
        super().__init__()
        if split == "train" or split == "val" or split == "test":
            path = os.path.join(dataset_path, f"{tokenizer_name}_{split}")
            self.tokens = torch.load(f"{path}_tokens.pt")
            self.labels = torch.load(f"{path}_labels.pt")
            self.offsets = torch.load(f"{path}_offsets.pt")
        else:
            raise ValueError("Wrong split")
            
    def __getitem__(self, index):
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return self.tokens[start: end], self.labels[start: end] 
    
    def __len__(self):
        return len(self.offsets) - 1