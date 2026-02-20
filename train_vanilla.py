import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from functools import partial
from tokenizers import Tokenizer
from src.dataset import OpenWebTextDataset
from src.model import NanoGPT
from src.utils import GPT2LRScheduler, collate_batch_fn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torchinfo import summary
from tqdm import tqdm

# Import DDP utilities
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


##############################################################################
# DDP Training Logic
###############################################################################


def setup_distributed():
    # Environment variables LOCAL_RANK, RANK, WORLD_SIZE are set by torchrun
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def train():
    local_rank = setup_distributed()
    is_main_process = local_rank == 0

    batch_size = 48
    num_epochs = 1
    tokenizer_path = "tokenizers/openwebtext16k_tokenizer.json"

    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_token = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()

    train_ds = OpenWebTextDataset("train")
    val_ds = OpenWebTextDataset("val")

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    collate_fn = partial(collate_batch_fn, pad_token=pad_token)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn
    )

    model = (
        NanoGPT(vocab_size=vocab_size, tokenizer_path=tokenizer_path, attn_type="flash")
        .to(local_rank)
        .to(torch.bfloat16)
    )
    model = DDP(model, device_ids=[local_rank])

    optim = AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-10,
    )
    total_steps = len(train_loader) * num_epochs
    warmup_steps = total_steps * 0.05
    lr_scheduler = GPT2LRScheduler(
        optim, warmup_steps=warmup_steps, total_steps=total_steps
    )

    if is_main_process:
        run = wandb.init(
            project="llm_sandbox",
            name="openwebtext_320kk",
        )
        os.makedirs("checkpoints", exist_ok=True)
        print(summary(model, input_size=(batch_size, 1024), dtypes=[torch.long]))

    step = 0
    grad_acc_steps = max(1, 512 // (batch_size * int(os.environ["WORLD_SIZE"])))
    log_freq = 100 // grad_acc_steps
    avg_val_loss = torch.tensor([0.0])
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        optim.zero_grad(set_to_none=True)

        model.train()

        pbar = tqdm(train_loader, disable=not is_main_process)
        for i, (x, y) in enumerate(pbar):
            should_update_model = (i + 1) % grad_acc_steps == 0
            model.require_backward_grad_sync = should_update_model

            x, y = x.to(local_rank), y.to(local_rank)

            y_pred = model(x).to(torch.float32)
            loss = F.cross_entropy(y_pred.transpose(1, 2), y) / grad_acc_steps
            loss.backward()

            if not should_update_model:
                continue

            lr_scheduler.adjust_lr()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

            if is_main_process:
                actual_train_loss = loss.item() * grad_acc_steps
                pbar.set_description(
                    f"Step {step} Train Loss: {actual_train_loss:.4f} | Val Loss: {avg_val_loss.item():.4f} | PPL: {np.exp(avg_val_loss.item()):.2f}"
                )

            if step % log_freq == 0:
                model.eval()

                val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(local_rank), y.to(local_rank)
                        y_pred = model(x)
                        val_loss += F.cross_entropy(y_pred.transpose(1, 2), y).item()

                avg_val_loss = torch.tensor(val_loss / len(val_loader)).to(local_rank)
                dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
                avg_val_loss /= dist.get_world_size()

                model.train()

                if is_main_process:
                    pbar.set_description(
                        f"Step {step} Train Loss: {actual_train_loss:.4f} | Val Loss: {avg_val_loss.item():.4f} | PPL: {np.exp(avg_val_loss.item()):.2f}"
                    )
                    run.log(
                        {
                            "Train loss": actual_train_loss,
                            "Val loss": avg_val_loss.item(),
                            "Grad norm": grad_norm.item(),
                            "LR": lr_scheduler.lr,
                        }
                    )

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.module.state_dict(), f"checkpoints/weights.pt")

            step += 1

    cleanup()


if __name__ == "__main__":
    train()
