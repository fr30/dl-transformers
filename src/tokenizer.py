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


# ---- multiprocessing worker state ----
_WORKER_TOKENIZER = None

def preprocess_tinyshakespare(tokenizer_path="tokenizers/shakespeare_tokenizer.json"):
    tokenizer_save_path = "shakespeare_tokenizer.json"
    dset_name = "tinyshakespeare"
    raw_ds = load_dataset("Trelis/tiny-shakespeare")
    
    ds_iter = (elem['Text'] for elem in raw_ds['train'])
    train_tokenizer(ds_iter, tokenizer_save_path)

    test_ds = raw_ds["test"]
    train_ds = raw_ds["train"]
    train_ds, val_ds = _split_dataset_sequential(train_ds, 0.99)
    train_iter = (elem['Text'] for elem in train_ds)
    val_iter = (elem['Text'] for elem in val_ds)
    test_iter = (elem['Text'] for elem in test_ds)
    dsets = [("train", train_iter), ("val", val_iter), ("test", test_iter)]

    _tokenize_datasets(dset_name, dsets, tokenizer_path)

def preprocess_tinystories(tokenizer_path="tokenizers/tinystories_tokenizer.json"):
    tokenizer_save_path = "tinystories_tokenizer.json"
    dset_name = "tinystories"
    raw_ds = load_dataset("roneneldan/TinyStories")
    
    ds_iter = (elem['text'] for elem in raw_ds['train'])
    train_tokenizer(ds_iter, tokenizer_save_path)

    test_ds = raw_ds["validation"]
    train_ds = raw_ds["train"]
    train_ds, val_ds = _split_dataset_sequential(train_ds, 0.99)

    train_iter = (elem['text'] for elem in train_ds)
    val_iter = (elem['text'] for elem in val_ds)
    test_iter = (elem['text'] for elem in test_ds)

    dsets = [("train", train_iter), ("val", val_iter), ("test", test_iter)]

    _tokenize_datasets(dset_name, dsets, tokenizer_path)

def preprocess_openwebtext(tokenizer_path="tokenizers/openwebtext_tokenizer.json"):
    tokenizer_save_path = "openwebtext_tokenizer.json"
    dset_name = "openwebtext"
    raw_ds = load_dataset("Skylion007/openwebtext")
    
    ds_iter = (elem['text'] for elem in raw_ds['train'])
    train_tokenizer(ds_iter, tokenizer_save_path)

    train_ds = raw_ds["train"]
    train_ds, val_ds = _split_dataset_sequential(train_ds, 0.98)
    val_ds, test_ds = _split_dataset_sequential(val_ds, 0.5)

    train_iter = (elem['text'] for elem in train_ds)
    val_iter = (elem['text'] for elem in val_ds)
    test_iter = (elem['text'] for elem in test_ds)

    dsets = [("train", train_iter), ("val", val_iter), ("test", test_iter)]

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

    vocab_size = 8124
    min_frequency = 2
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet()
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
        torch.utils.data.Subset(dataset, indices[n_first:])
    )

def _tokenize_datasets(dset_name, dsets, tokenizer_path, max_seq_len=1024):
    print("Tokenizing datasets...")
    os.makedirs(f"data/{dset_name}", exist_ok=True)

    for split, dset in dsets:
        _build_one_split(
            dset_name,
            split, 
            dset,
            max_seq_len=max_seq_len,
            tokenizer_path=tokenizer_path,
            chunksize=1024,
            max_workers=16,
        )

    print("Done.")

def _build_one_split(dset_name, split, dset_iter, tokenizer_path, max_seq_len, chunksize=256, max_workers=None):
    tokenizer_name = tokenizer_path.split("/")[-1].split(".")[-2]
    out_tokens_path = f"data/{dset_name}/{tokenizer_name}_{split}_tokens.pt"
    out_labels_path = f"data/{dset_name}/{tokenizer_name}_{split}_labels.pt"
    out_offsets_path = f"data/{dset_name}/{tokenizer_name}_{split}_offsets.pt"

    if os.path.exists(out_tokens_path) and os.path.exists(out_labels_path) and os.path.exists(out_offsets_path):
        return

    dset_list = list(dset_iter) 
    n = len(dset_list)
    texts = (re.sub(r'\n+', '\n', text) for text in dset_list)
    tasks = ((i, text) for i, text in enumerate(texts))
    results = [None] * n
    
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(tokenizer_path,),
    ) as ex:
        it = ex.map(_tokenize_text, tasks, chunksize=chunksize)
        for i, ids, labels, length in tqdm(it, total=n, desc=f"{split} tokenizing"):
            results[i] = (ids, labels, length)

    all_ids = []
    all_labels = []
    lengths = []

    for ids, labels, length in results:
        all_ids.extend(ids[:max_seq_len])
        all_labels.extend(labels[:max_seq_len])
        lengths.append(length)

    tokens = torch.tensor(all_ids, dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.long)

    offsets = [0]
    total = 0
    for ln in lengths:
        total += ln
        offsets.append(total)
    offsets = torch.tensor(offsets, dtype=torch.long)

    torch.save(tokens, out_tokens_path)
    torch.save(labels, out_labels_path)
    torch.save(offsets, out_offsets_path)


def _init_worker(tokenizer_path: str):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)

def _tokenize_text(args):
    i, text = args
    # Use per-process tokenizer (loaded once by initializer)
    ids = _WORKER_TOKENIZER.encode(text).ids
    x = ids[:-1]
    y = ids[1:]
    return i, x, y, len(ids)




