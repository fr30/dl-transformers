import os
import re
import torch
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
import numpy as np

# ---- multiprocessing worker state ----
_WORKER_TOKENIZER = None


def preprocess_tinyshakespare(tokenizer_path="tokenizers/shakespeare_tokenizer.json"):
    tokenizer_save_path = "shakespeare_tokenizer.json"
    dset_name = "tinyshakespeare"
    raw_ds = load_dataset("Trelis/tiny-shakespeare")

    ds_iter = (elem["Text"] for elem in raw_ds["train"])
    train_tokenizer(ds_iter, tokenizer_save_path)

    test_ds = raw_ds["test"]
    train_ds = raw_ds["train"]
    train_ds, val_ds = _split_dataset_sequential(train_ds, 0.9)
    train_factory = lambda: (elem["Text"] for elem in train_ds)
    val_factory = lambda: (elem["Text"] for elem in val_ds)
    test_factory = lambda: (elem["Text"] for elem in test_ds)
    dsets = [
        ("train", train_factory, len(train_ds)),
        ("val", val_factory, len(val_ds)),
        ("test", test_factory, len(test_ds)),
    ]

    _tokenize_datasets(dset_name, dsets, tokenizer_path)


def preprocess_tinystories(tokenizer_path="tokenizers/tinystories_tokenizer.json"):
    tokenizer_save_path = "tinystories_tokenizer.json"
    dset_name = "tinystories"
    raw_ds = load_dataset("roneneldan/TinyStories")

    ds_iter = (elem["text"] for elem in raw_ds["train"])
    train_tokenizer(ds_iter, tokenizer_save_path)

    test_ds = raw_ds["validation"]
    train_ds = raw_ds["train"]
    train_ds, val_ds = _split_dataset_sequential(train_ds, 0.99)

    train_factory = lambda: (elem["text"] for elem in train_ds)
    val_factory = lambda: (elem["text"] for elem in val_ds)
    test_factory = lambda: (elem["text"] for elem in test_ds)

    dsets = [
        ("train", train_factory, len(train_ds)),
        ("val", val_factory, len(val_ds)),
        ("test", test_factory, len(test_ds)),
    ]

    _tokenize_datasets(dset_name, dsets, tokenizer_path)


def preprocess_openwebtext(tokenizer_path="tokenizers/openwebtext16k_tokenizer.json"):
    tokenizer_save_path = "openwebtext16k_tokenizer.json"
    dset_name = "openwebtext"
    raw_ds = load_dataset("openwebtext", num_proc=16)
    tokenizer_train_size = int(len(raw_ds["train"]) * 0.01)

    ds_iter = (raw_ds["train"][i]["text"] for i in range(tokenizer_train_size))
    train_tokenizer(ds_iter, tokenizer_save_path)

    train_ds = raw_ds["train"]
    train_ds, val_ds = _split_dataset_sequential(train_ds, 0.9995)

    train_factory = lambda: (elem["text"] for elem in train_ds)
    val_factory = lambda: (elem["text"] for elem in val_ds)

    dsets = [
        ("train", train_factory, len(train_ds)),
        ("val", val_factory, len(val_ds)),
    ]

    _tokenize_datasets(dset_name, dsets, tokenizer_path)


def train_tokenizer(ds_iter, tokenizer_path):
    print("Training tokenizer...")
    os.makedirs("tokenizers", exist_ok=True)

    output_path = f"tokenizers/{tokenizer_path}"

    if os.path.exists(output_path):
        return

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    vocab_size = 16384
    min_frequency = 10
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(ds_iter, trainer)
    tokenizer.decoder = ByteLevelDecoder()

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    tokenizer.save(output_path)
    print(f"Saved tokenizer to: {output_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


def _split_dataset_sequential(dataset, ratio):
    n_total = len(dataset)
    n_first = int(n_total * ratio)

    indices = range(n_total)
    return (
        torch.utils.data.Subset(dataset, indices[:n_first]),
        torch.utils.data.Subset(dataset, indices[n_first:]),
    )


def _tokenize_datasets(dset_name, dsets, tokenizer_path, max_seq_len=1024):
    print("Tokenizing datasets...")
    os.makedirs(f"data/{dset_name}", exist_ok=True)

    for split, dset_factory, n in dsets:
        _build_one_split(
            dset_name=dset_name,
            split=split,
            dset_factory=dset_factory,
            dset_len=n,
            max_seq_len=max_seq_len,
            tokenizer_path=tokenizer_path,
            chunksize=1024,
            max_workers=16,
        )

    print("Done.")


def _build_one_split(
    dset_name,
    split,
    dset_factory,
    dset_len,
    tokenizer_path,
    max_seq_len,
    chunksize=1024,
    max_workers=None,
):
    tokenizer_name = tokenizer_path.split("/")[-1].split(".")[0]
    out_tokens_path = f"data/{dset_name}/{tokenizer_name}_{split}_tokens.bin"
    out_offsets_path = f"data/{dset_name}/{tokenizer_name}_{split}_offsets.npy"  # Switched to npy for simplicity

    if os.path.exists(out_tokens_path):
        print(f"{split} already exists. Skipping.")
        return

    print(f"Processing {dset_name} {split} in a single pass...")

    # We use an array.array or numpy array to store offsets efficiently
    # Initializing with 0 for the first offset
    offsets = [0]

    # Open file in binary append mode
    with open(out_tokens_path, "wb") as f:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(tokenizer_path,),
        ) as ex:
            # Note: we only do ONE map operation now
            texts = (
                text for text in dset_factory()
            )  # removed regex for speed; add back if vital
            it = ex.map(_tokenize_only_ids, texts, chunksize=chunksize)

            current_total = 0
            for ids in tqdm(it, total=dset_len, desc=f"Writing {split}"):
                # Truncate to max_seq_len
                tokens = np.array(
                    ids[:max_seq_len], dtype=np.uint16
                )  # uint16 if vocab < 65k

                # Write directly to disk
                f.write(tokens.tobytes())

                # Update metadata
                n_tokens = len(tokens)
                current_total += n_tokens
                offsets.append(current_total)

    # Save offsets as a memory-mapped-friendly numpy file
    np.save(out_offsets_path, np.array(offsets, dtype=np.uint64))
    print(f"Saved {current_total} tokens to {out_tokens_path}")


def _tokenize_only_ids(text):
    import re

    text = re.sub(r"\n+", "\n", text)
    return _WORKER_TOKENIZER.encode(text).ids


def _init_worker(tokenizer_path: str):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)
