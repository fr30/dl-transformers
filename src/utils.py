import numpy as np
from torch.nn.utils.rnn import pad_sequence


class GPT2LRScheduler:
    def __init__(self, optim, max_lr=3e-4, min_lr=3e-5, warmup_steps=2000, total_steps=200000):
        self.optim = optim
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.step = 0

    def adjust_lr(self):
        self.step += 1
        s = self.step

        if s <= self.warmup_steps:
            # linear warmup: 0 -> max_lr
            lr = self.max_lr * (s / self.warmup_steps)
        else:
            # cosine decay: max_lr -> min_lr
            progress = (s - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine

        for pg in self.optim.param_groups:
            pg["lr"] = lr

        return lr
    
def collate_batch_fn(args, pad_token):
    x, y = zip(*args)
    xlen = len(x)
    padded = pad_sequence(list(x) + list(y), padding_value=pad_token, batch_first=True)

    return padded[:xlen], padded[xlen:]