import logging
import math
import time
import torch
import torch.nn.functional as F
import datasets
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm
import logging

def get_dataset(name: str) -> datasets.DatasetDict:
    """
    Get the dataset from the HuggingFace datasets library.

    Args:
        name: The name of the HuggingFace dataset to load. Must be one of "wikitext2", "ptb", "c4" or "alpaca".

    Returns:
        The dataset.
    """
    logging.info(f"Loading dataset: {name}")

    ds_properties = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "ptb": {"path": "ptb_text_only", "config_name": "penn_treebank"},
        "c4": {
            "path": "allenai/c4",
            "config_name": "en",
            "data_files": {
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            "cols_to_remove": ['url', 'timestamp'],
        },
        "alpaca": {"path": "tatsu-lab/alpaca", "cols_to_remove": ['input', 'output', 'instruction']},
    }

    if name not in ds_properties:
        raise NotImplementedError("The provided dataset is not supported")

    properties = ds_properties[name]
    ds = datasets.load_dataset(
        properties["path"], name=properties.get("config_name"), data_files=properties.get("data_files")
    )

    if "cols_to_remove" in properties:
        ds = ds.remove_columns(properties["cols_to_remove"])

    # if alpaca, create a test and validation set from the training set
    if name == "alpaca":
        ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        temp_ds = ds.pop("test")
        temp_ds = temp_ds.train_test_split(test_size=0.5, seed=42)
        ds["test"] = temp_ds["train"]
        ds["validation"] = temp_ds["test"]

    logging.info("Loading dataset done")
    return ds

def prepare_test_dataloader(
    dataset: datasets.Dataset, tokenizer: PreTrainedTokenizerBase, seqlen: int = 2048, batch_size: int = 1
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a test dataset. This dataloader should be used when comparing WikiText2 perplexities with other papers, e.g. SparseGPT (arxiv.org/abs/2301.00774).

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        seqlen: The sequence length of sequences in the dataset.
        batch_size: The batch size.

    Returns:
        A DataLoader.
    """

    logging.info(f"Preparing test dataloader")

    class TestDataset(Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):
            """Tokenize the entire dataset and reshape it into sequences of length seqlen."""

            tokenized_ds = tokenizer("\n\n".join(ds['text']), return_tensors='pt')
            nsamples = tokenized_ds.input_ids.numel() // seqlen

            input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen]
            input_ids = input_ids.reshape(nsamples, seqlen)
            attn_mask = tokenized_ds.attention_mask[0, : nsamples * seqlen]
            attn_mask = attn_mask.reshape(nsamples, seqlen)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attn_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, seqlen)
    loader = DataLoader(test_ds, batch_size=batch_size)
    logging.info(f"Preparing test dataloader done")
    return loader

def prepare_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    varied_seqlen: bool = False,
    seed=42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a dataset.

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        max_seqlen: The maximum sequence length, used for truncation of sequences in the dataset.
        batch_size: The batch size.
        nsamples: The number of samples to produce.
        varied_seqlen: If False, concatenate multiple examples from the dataset into one example until max_seqlen is reached.
        seed: The seed for sampling the dataset.

    Returns:
        A DataLoader.
    """
    logging.info(f"Preparing dataloader")

    if not varied_seqlen and not nsamples:
        logging.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to tokenization of the entire dataset, which will be slow."
        )

    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
        data_list = ds[data_name]
        new_data_list = []

        torch.manual_seed(seed)
        indices = list(range(len(data_list)))

        while len(new_data_list) < nsamples and len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,)).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else "\n\n"
                tokens += tokenizer.tokenize(sep + item)
                idx += 1

            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(data_batch):
        # tokenize then pad all sequences to max_seqlen to ensure consistent batch sizes
        batch = tokenizer(
            data_batch[data_name],
            padding="max_length",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)

    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(ds))[:nsamples])

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    logging.info(f"Preparing dataloader done")
    return loader

def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)
        
def map_tensors(obj, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj


class _ScaledDotProductAttentionCapture:
    """
    Lightweight hook manager that intercepts torch.nn.functional.scaled_dot_product_attention
    calls so we can observe the query/key tensors without modifying model code.
    """

    _orig_fn = None
    _monitor_stack: "list[QKDotProductMonitor]" = []

    @classmethod
    def enable(cls, monitor: "QKDotProductMonitor | None"):
        if monitor is None:
            return
        if cls._orig_fn is None:
            cls._orig_fn = F.scaled_dot_product_attention

            def wrapped(*args, **kwargs):
                query = kwargs.get("query") if "query" in kwargs else (args[0] if len(args) > 0 else None)
                key = kwargs.get("key") if "key" in kwargs else (args[1] if len(args) > 1 else None)
                if cls._monitor_stack and (query is not None and key is not None):
                    try:
                        cls._monitor_stack[-1].observe(query, key)
                    except Exception:
                        logging.exception("Failed to capture QK dot products during attention call")
                return cls._orig_fn(*args, **kwargs)

            F.scaled_dot_product_attention = wrapped  # type: ignore[assignment]
        cls._monitor_stack.append(monitor)

    @classmethod
    def disable(cls, monitor: "QKDotProductMonitor | None"):
        if monitor is None:
            return
        if monitor in cls._monitor_stack:
            cls._monitor_stack = [m for m in cls._monitor_stack if m is not monitor]
        if not cls._monitor_stack and cls._orig_fn is not None:
            F.scaled_dot_product_attention = cls._orig_fn  # type: ignore[assignment]
            cls._orig_fn = None


class QKDotProductMonitor:
    """
    Collects QK dot products per sequence, with K averaged over head dimension.
    Stores complete dot product matrices per sequence (no sampling within sequence).
    Can save to disk to avoid OOM.
    """

    def __init__(self, label: str = "", samples_per_call: int = 2048, save_dir: str | None = None, max_sequences: int = 10):
        self.label = label
        self.samples_per_call = samples_per_call  # Not used for within-sequence sampling, kept for compatibility
        self.save_dir = save_dir  # Directory to save dot products to disk
        self.max_sequences = max_sequences  # Only process first N sequences
        # Store per-sequence: _samples[sequence_idx] = list of tensors from that sequence
        self._samples: list[list[torch.Tensor]] = []
        self._current_batch_start_idx = 0  # Starting index for current batch
        self._sequence_count = 0  # Total sequences processed so far

    def reset(self):
        self._samples.clear()
        self._current_batch_start_idx = 0
        self._sequence_count = 0

    def start_batch(self, batch_size: int):
        """Call at the start of each new batch to track sequences separately."""
        # Only process if we haven't reached max_sequences
        remaining_slots = max(0, self.max_sequences - self._sequence_count)
        if remaining_slots == 0:
            return  # Skip this batch if we already have enough sequences
        
        # Reserve slots for sequences in this batch (up to max_sequences)
        self._current_batch_start_idx = len(self._samples)
        actual_batch_size = min(batch_size, remaining_slots)
        for _ in range(actual_batch_size):
            self._samples.append([])

    def observe(self, query: torch.Tensor | None, key: torch.Tensor | None):
        if query is None or key is None:
            return
        if query.ndim < 3 or key.ndim < 3:
            return
        
        # Skip if we've already collected enough sequences
        if self._sequence_count >= self.max_sequences:
            return
            
        with torch.no_grad():
            q = query.detach().float()
            k = key.detach().float()
            
            # Average K over head dimension if it's not already 1
            # Expected shapes: Q: [B, H, L, feature_dim], K: [B, H_kv, L, feature_dim] or [B, 1, L, feature_dim]
            if k.shape[1] > 1:  # If K has multiple heads, average them
                k = k.mean(dim=1, keepdim=True)  # [B, 1, L, feature_dim]
            
            # Now both Q and K have head dimension: Q: [B, H, L, feature_dim], K: [B, 1, L, feature_dim]
            b, h_q, l, d = q.shape
            b_k, h_k, l_k, d_k = k.shape
            assert b == b_k and l == l_k and d == d_k, f"Shape mismatch: Q {q.shape} vs K {k.shape}"
            
            # Only process sequences up to max_sequences
            remaining_slots = self.max_sequences - self._sequence_count
            actual_b = min(b, remaining_slots)
            
            # Process each sequence in the batch separately - NO SAMPLING WITHIN SEQUENCE
            for batch_idx in range(actual_b):
                seq_idx = self._current_batch_start_idx + batch_idx
                
                if seq_idx >= len(self._samples):
                    continue
                
                # Get data for this sequence
                q_seq = q[batch_idx]  # [H, L, d]
                k_seq = k[batch_idx].squeeze(0)  # [1, L, d] -> [L, d]
                
                # Compute complete Q @ K^T for each head: [H, L, d] @ [d, L] = [H, L, L]
                # q_seq: [H, L, d], k_seq: [L, d]
                # For each head h: q_seq[h] @ k_seq^T = [L, d] @ [d, L] = [L, L]
                scores = torch.bmm(q_seq, k_seq.unsqueeze(0).expand(h_q, l, d).transpose(1, 2))  # [H, L, L]
                
                # Print shape for first sequence, first layer
                if seq_idx == 0 and len(self._samples[seq_idx]) == 0:
                    print(f"[{self.label}] First sequence (seq_0) dot product tensor shape:")
                    print(f"  - Shape: {scores.shape} (Q @ K^T matrix: [H={h_q}, L={l}, L={l}])")
                    print(f"  - Will compute KL per head: each head has shape [L={l}, L={l}]")
                
                if scores.numel() == 0:
                    continue
                
                # Store per head: split [H, L, L] into H separate [L, L] matrices
                # Keep as [L, L] shape (not flattened) so we can compute KL per row
                layer_data = []  # List to store data for this layer (one per head)
                for head_idx in range(h_q):
                    head_scores = scores[head_idx]  # [L, L] - keep 2D shape
                    
                    if self.save_dir is not None:
                        import os
                        os.makedirs(self.save_dir, exist_ok=True)
                        import numpy as np
                        file_path = os.path.join(self.save_dir, f"seq_{seq_idx}_layer_{len(self._samples[seq_idx])}_head_{head_idx}.npy")
                        np.save(file_path, head_scores.cpu().numpy())  # Save as [L, L] shape
                        layer_data.append(file_path)
                    else:
                        layer_data.append(head_scores.to(dtype=torch.float32, device="cpu"))  # Keep as [L, L]
                
                # Store all heads for this layer
                self._samples[seq_idx].append(layer_data)
            
            # Update sequence count
            self._sequence_count += actual_b

    def get_sequence_tensor(self, sequence_idx: int) -> torch.Tensor:
        """Get concatenated tensor for a specific sequence (legacy method, kept for compatibility)."""
        if sequence_idx >= len(self._samples) or len(self._samples[sequence_idx]) == 0:
            return torch.empty(0, dtype=torch.float32)
        
        # Load from disk if saved, otherwise use in-memory tensors
        tensors = []
        for layer_data in self._samples[sequence_idx]:
            # layer_data is a list of head tensors/paths
            for head_item in layer_data:
                if isinstance(head_item, str):
                    # Load from disk
                    import numpy as np
                    arr = np.load(head_item)
                    tensors.append(torch.from_numpy(arr).float())
                else:
                    # In-memory tensor
                    tensors.append(head_item)
        
        if not tensors:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(tensors)
    
    def get_sequence_head_tensors(self, sequence_idx: int) -> list[list[torch.Tensor]]:
        """
        Get tensors organized by layer and head: returns list[list[torch.Tensor]]
        Outer list: layers, inner list: heads
        Each head tensor has shape [L, L] (2D matrix, not flattened)
        """
        if sequence_idx >= len(self._samples) or len(self._samples[sequence_idx]) == 0:
            return []
        
        result = []
        for layer_data in self._samples[sequence_idx]:
            # layer_data is a list of head tensors/paths
            head_tensors = []
            for head_item in layer_data:
                if isinstance(head_item, str):
                    # Load from disk
                    import numpy as np
                    arr = np.load(head_item)
                    head_tensors.append(torch.from_numpy(arr).float())
                else:
                    # In-memory tensor
                    head_tensors.append(head_item)
            result.append(head_tensors)
        
        return result

    def num_sequences(self) -> int:
        """Return number of sequences collected."""
        return len(self._samples)

    def to_tensor(self) -> torch.Tensor:
        """Legacy method: concatenate all sequences. Use get_sequence_tensor for per-sequence access."""
        if not self._samples:
            return torch.empty(0, dtype=torch.float32)
        all_tensors = []
        for seq_idx in range(len(self._samples)):
            seq_tensor = self.get_sequence_tensor(seq_idx)
            if seq_tensor.numel() > 0:
                all_tensors.append(seq_tensor)
        if not all_tensors:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(all_tensors)

    def num_values(self) -> int:
        total = 0
        for seq_samples in self._samples:
            for layer_data in seq_samples:
                # layer_data is a list of head tensors/paths
                for head_item in layer_data:
                    if isinstance(head_item, str):
                        # Load from disk to count
                        import numpy as np
                        arr = np.load(head_item)
                        total += arr.size
                    else:
                        total += head_item.numel()
        return total

    def summary(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "num_values": self.num_values(),
        }


def compute_qk_kl_divergence(
    reference_monitor: QKDotProductMonitor | None,
    target_monitor: QKDotProductMonitor | None,
    bins: int = 512,
    epsilon: float = 1e-8,
    max_sequences: int = 10,
) -> float | None:
    """
    Compute KL(reference || target) using softmax-normalized attention scores.
    
    For each head's [L, L] attention score matrix:
    1. Apply softmax to each row (creating probability distributions over keys)
    2. Compute KL divergence for each row: KL(P||Q) = sum(P * log(P / Q))
       where P is reference (baseline) and Q is target
    3. Average KL across all rows for that head
    4. Average KL across all heads (and sequences and layers)
    
    Also prints statistics (max, avg, min) for the first sequence/layer/head.
    
    Args:
        reference_monitor: Monitor with reference (Phase 0) data
        target_monitor: Monitor with target (Phase 1 or 2) data
        bins: Not used (kept for compatibility)
        epsilon: Small value to avoid log(0)
        max_sequences: Maximum number of sequences to use (default 10)
    
    Returns:
        Average KL divergence across all heads, or None if insufficient data
    """
    if reference_monitor is None or target_monitor is None:
        return None

    num_ref_sequences = reference_monitor.num_sequences()
    num_target_sequences = target_monitor.num_sequences()
    
    if num_ref_sequences == 0 or num_target_sequences == 0:
        return None
    
    # Use minimum of available sequences and max_sequences
    num_sequences = min(num_ref_sequences, num_target_sequences, max_sequences)
    
    if num_sequences == 0:
        return None

    # New approach: compute KL per head, then average
    # Structure: reference_heads[seq_idx][layer_idx][head_idx] = tensor[L*L]
    all_head_kl_values = []
    
    # Statistics for the first sequence, first layer, first head (for reporting)
    stats_printed = False
    
    for seq_idx in range(num_sequences):
        reference_heads = reference_monitor.get_sequence_head_tensors(seq_idx)  # list[list[tensor]]
        target_heads = target_monitor.get_sequence_head_tensors(seq_idx)  # list[list[tensor]]
        
        if not reference_heads or not target_heads:
            continue
        
        # Ensure same number of layers
        num_layers = min(len(reference_heads), len(target_heads))
        if num_layers == 0:
            continue
        
        # For each layer, compute KL per head
        for layer_idx in range(num_layers):
            ref_layer_heads = reference_heads[layer_idx]  # list[tensor], one per head
            tgt_layer_heads = target_heads[layer_idx]  # list[tensor], one per head
            
            if not ref_layer_heads or not tgt_layer_heads:
                continue
            
            # Ensure same number of heads
            num_heads = min(len(ref_layer_heads), len(tgt_layer_heads))
            if num_heads == 0:
                continue
            
            # Compute KL for each head separately
            for head_idx in range(num_heads):
                ref_head = ref_layer_heads[head_idx]  # tensor[L, L]
                tgt_head = tgt_layer_heads[head_idx]  # tensor[L, L]
                
                if ref_head.numel() == 0 or tgt_head.numel() == 0:
                    continue
                
                # Ensure both are 2D [L, L] matrices
                if ref_head.ndim != 2 or tgt_head.ndim != 2:
                    # If flattened, reshape back to [L, L]
                    L = int(math.sqrt(ref_head.numel()))
                    ref_head = ref_head.reshape(L, L)
                    tgt_head = tgt_head.reshape(L, L)
                
                # Print statistics for the first head (once)
                if not stats_printed:
                    ref_max = ref_head.max().item()
                    ref_min = ref_head.min().item()
                    ref_avg = ref_head.mean().item()
                    tgt_max = tgt_head.max().item()
                    tgt_min = tgt_head.min().item()
                    tgt_avg = tgt_head.mean().item()
                    print(f"QK Dot Product Statistics (seq_{seq_idx}, layer_{layer_idx}, head_{head_idx}):")
                    print(f"  Reference matrix: max={ref_max:.6f}, avg={ref_avg:.6f}, min={ref_min:.6f}")
                    print(f"  Target matrix:    max={tgt_max:.6f}, avg={tgt_avg:.6f}, min={tgt_min:.6f}")
                    stats_printed = True
                
                L = ref_head.shape[0]
                head_row_kl_values = []
                
                # Apply softmax to each row, then compute KL divergence
                for row_idx in range(L):
                    ref_row = ref_head[row_idx]  # [L]
                    tgt_row = tgt_head[row_idx]  # [L]
                    
                    if ref_row.numel() == 0 or tgt_row.numel() == 0:
                        continue
                    
                    # Apply softmax to each row (probability distribution over keys)
                    ref_row_softmax = torch.nn.functional.softmax(ref_row, dim=-1)  # [L]
                    tgt_row_softmax = torch.nn.functional.softmax(tgt_row, dim=-1)  # [L]
                    
                    # Compute KL divergence: KL(P||Q) = sum(P * log(P / Q))
                    # where P is reference (baseline) and Q is target
                    kl = torch.sum(ref_row_softmax * torch.log((ref_row_softmax + epsilon) / (tgt_row_softmax + epsilon)))
                    head_row_kl_values.append(kl.item())
                
                # Average KL across all rows for this head
                if head_row_kl_values:
                    head_kl = sum(head_row_kl_values) / len(head_row_kl_values)
                    all_head_kl_values.append(head_kl)
    
    if len(all_head_kl_values) == 0:
        return None
    
    # Return average KL divergence across all heads (and sequences and layers)
    return sum(all_head_kl_values) / len(all_head_kl_values)
    
@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module, 
    pad_token_id: int | None, 
    testloader: DataLoader[dict[str, torch.Tensor]], 
    message: str = "Evaluating perplexity",
    qk_monitor: QKDotProductMonitor | None = None,
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    sync_gpus()

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    if qk_monitor is not None:
        qk_monitor.reset()
        _ScaledDotProductAttentionCapture.enable(qk_monitor)

    logging.info(message)
    try:
        for batch_idx, batch in enumerate(tqdm(testloader, desc=message)):
            logging.debug(f"Evaluating batch {len(nlls)}")
            
            # Only start batch tracking if we still need to collect QK data
            # But always continue perplexity evaluation for full test set
            if qk_monitor is not None and qk_monitor._sequence_count < qk_monitor.max_sequences:
                # Mark start of new batch to track sequences separately
                batch_size = batch["input_ids"].shape[0] if isinstance(batch, dict) and "input_ids" in batch else 1
                qk_monitor.start_batch(batch_size)
            
            batch = map_tensors(batch, model.model.embed_tokens.weight.device)
            logits = model(**batch, use_cache=False).logits

            # shift outputs and labels autoregressively.
            logits = logits[:, :-1, :]
            shift_labels = batch["input_ids"][:, 1:]

            # CrossEntropyLoss demands data dimension is dimension 1.
            nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

            mask = shift_labels != loss_fn.ignore_index
            nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
            nlls.append(nll_means)
            
            # Disable QK monitoring after collecting enough sequences, but continue perplexity evaluation
            if qk_monitor is not None and qk_monitor._sequence_count >= qk_monitor.max_sequences:
                _ScaledDotProductAttentionCapture.disable(qk_monitor)
    finally:
        if qk_monitor is not None:
            _ScaledDotProductAttentionCapture.disable(qk_monitor)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()

def insert_qkv_hooks(model):
    query_hooks = []
    key_hooks = []
    value_hooks = []
    q_a_proj_hooks = []
    kv_a_proj_with_mqa_hooks = []
    query_outputs = {}
    key_outputs = {}
    value_outputs = {}
    q_a_proj_outputs = {}
    kv_a_proj_with_mqa_outputs = {}

    def query_hook_fn(module, input, output, index):
        if index not in query_outputs:
            query_outputs[index] = []
        query_outputs[index].append(output.to('cpu'))

    def key_hook_fn(module, input, output, index):
        if index not in key_outputs:
            key_outputs[index] = []
        key_outputs[index].append(output.to('cpu'))
        
    def value_hook_fn(module, input, output, index):
        if index not in value_outputs:
            value_outputs[index] = []
        value_outputs[index].append(output.to('cpu'))

    def q_a_proj_hook_fn(module, input, output, index):
        if index not in q_a_proj_outputs:
            q_a_proj_outputs[index] = []
        q_a_proj_outputs[index].append(output.to('cpu'))

    def kv_a_proj_with_mqa_hook_fn(module, input, output, index):
        if index not in kv_a_proj_with_mqa_outputs:
            kv_a_proj_with_mqa_outputs[index] = []
        kv_a_proj_with_mqa_outputs[index].append(output.to('cpu'))

    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "q_proj"):
            query_hook = layer.self_attn.q_proj.register_forward_hook(lambda module, input, output, idx=idx: query_hook_fn(module, input, output, idx))
            query_hooks.append(query_hook)
        if hasattr(layer.self_attn, "k_proj"):
            key_hook = layer.self_attn.k_proj.register_forward_hook(lambda module, input, output, idx=idx: key_hook_fn(module, input, output, idx))
            key_hooks.append(key_hook)
        if hasattr(layer.self_attn, "v_proj"):
            value_hook = layer.self_attn.v_proj.register_forward_hook(lambda module, input, output, idx=idx: value_hook_fn(module, input, output, idx))
            value_hooks.append(value_hook)
        if hasattr(layer.self_attn, "q_a_proj"):
            q_a_proj_hook = layer.self_attn.q_a_proj.register_forward_hook(lambda module, input, output, idx=idx: q_a_proj_hook_fn(module, input, output, idx))
            q_a_proj_hooks.append(q_a_proj_hook)
        if hasattr(layer.self_attn, "kv_a_proj_with_mqa"):
            kv_a_proj_with_mqa_hook = layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(lambda module, input, output, idx=idx: kv_a_proj_with_mqa_hook_fn(module, input, output, idx))
            kv_a_proj_with_mqa_hooks.append(kv_a_proj_with_mqa_hook)
    
    return query_hooks, key_hooks, value_hooks, q_a_proj_hooks, kv_a_proj_with_mqa_hooks, query_outputs, key_outputs, value_outputs, q_a_proj_outputs, kv_a_proj_with_mqa_outputs

@torch.no_grad()
def get_qkv_calibrate_outputs(
    model: torch.nn.Module, 
    trainloader: DataLoader[dict[str, torch.Tensor]], 
    message: str = "Calibrating QKV"
):
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    """

    start_time = time.time()

    model.eval()
    query_hooks, key_hooks, value_hooks, q_a_proj_hooks, kv_a_proj_with_mqa_hooks, query_outputs, key_outputs, value_outputs, q_a_proj_outputs, kv_a_proj_with_mqa_outputs = insert_qkv_hooks(model)
    ignore_masks = []
    logging.info(message)
    for batch in tqdm(trainloader, desc=message):
        batch = map_tensors(batch, model.model.embed_tokens.weight.device)
        ignore_masks.append(batch["attention_mask"].to('cpu'))
        model(**batch, use_cache=False)

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    for hook in query_hooks:
        hook.remove()
    for hook in key_hooks:
        hook.remove()
    for hook in value_hooks:
        hook.remove()
    for hook in q_a_proj_hooks:
        hook.remove()
    for hook in kv_a_proj_with_mqa_hooks:
        hook.remove()

    for value in query_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    for value in key_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    for value in value_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    for value in q_a_proj_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    for value in kv_a_proj_with_mqa_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    qkv_outputs = {
        "query": query_outputs,
        "key": key_outputs,
        "value": value_outputs,
        "q_a_proj": q_a_proj_outputs,
        "kv_a_proj": kv_a_proj_with_mqa_outputs,
    }
    return qkv_outputs

@torch.no_grad()
def pca_calc(X: list[torch.Tensor], device: str) -> torch.Tensor:
    H = None
    for idx, X_batch in enumerate(X):

        X_batch = X_batch.double().to(device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eigen_vec = X_eig[1][:, index]
    return eigen_vec

def use_original_norm_weights(self_attn, q_norm_weight, k_norm_weight):
    """
    Use original Qwen3 model's RMS norm weights instead of computing from calibration data.
    
    Since the original q_norm and k_norm have shape [head_dim] while q_a_layernorm and 
    kv_a_layernorm have shapes [q_lora_rank] and [kv_lora_rank] respectively, we use the 
    mean value of the original norm weights as a scalar and set all elements of the new 
    norm weights to that value.
    
    Args:
        self_attn: The attention module (LoraQKV instance) containing q_a_layernorm and
                   kv_a_layernorm modules
        q_norm_weight: Original q_norm.weight tensor from Qwen3 attention, shape [head_dim]
        k_norm_weight: Original k_norm.weight tensor from Qwen3 attention, shape [head_dim]
    """
    if q_norm_weight is not None and hasattr(self_attn, "q_a_layernorm"):
        # Use mean of original q_norm weights as scalar value
        q_norm_scalar = q_norm_weight.mean().item()
        self_attn.q_a_layernorm.weight.data.fill_(q_norm_scalar)
        self_attn.q_a_layernorm.weight.data = self_attn.q_a_layernorm.weight.data.to(
            self_attn.q_a_proj.weight.device
        ).to(self_attn.dtype)
    
    if k_norm_weight is not None and hasattr(self_attn, "kv_a_layernorm"):
        # Use mean of original k_norm weights as scalar value
        k_norm_scalar = k_norm_weight.mean().item()
        self_attn.kv_a_layernorm.weight.data.fill_(k_norm_scalar)
        self_attn.kv_a_layernorm.weight.data = self_attn.kv_a_layernorm.weight.data.to(
            self_attn.kv_a_proj_with_mqa.weight.device
        ).to(self_attn.dtype)

def statistics_qkv_rmsnorm(self_attn, q_a_outputs, kv_a_outputs):
    """
    Compute and set RMS normalization statistics for q_a_layernorm and kv_a_layernorm
    based on calibration outputs from q_a_proj and kv_a_proj_with_mqa.
    
    This function computes the RMS (Root Mean Square) normalization scale factor from
    calibration data and sets the layernorm weights to a constant value based on this
    statistic. This helps stabilize training by normalizing activations.
    
    Args:
        self_attn: The attention module (LoraQKV instance) containing q_a_layernorm and
                   kv_a_layernorm modules
        q_a_outputs: List of tensors from q_a_proj calibration outputs, or None if
                     q_a_proj doesn't exist (when q_lora_rank is None)
        kv_a_outputs: List of tensors from kv_a_proj_with_mqa calibration outputs
    
    Concrete Example (Qwen3-4B):
        Input shapes:
        - q_a_outputs: List of [batch_size, seq_len, q_hidden_dim] tensors from q_a_proj
          Example: [tensor([4, 128, 512]), 
                    tensor([4, 128, 512]), 
                    ...]
        - kv_a_outputs: List of [batch_size, seq_len, kv_hidden_dim] tensors from kv_a_proj_with_mqa
          Example: [tensor([4, 128, 576]), 
                    tensor([4, 128, 576]), 
                    ...]
        
        Process for q_a_layernorm (if q_a_outputs is not None):
        1. Concatenate all batches: torch.cat(q_a_outputs)
           Result: [total_tokens, q_hidden_dim] 
           where total_tokens = n_batches * batch_size * seq_len)
           Example: total_tokens = 2 * 4 * 128 = 1024
        
        2. Compute RMS: rsqrt(mean(x^2) + eps)
           - q_a_proj.pow(2): [1024, 512] → element-wise square
           - .mean(-1): [1024, 512] → [1024] (mean over last dim, per token)
           - + eps: [1024] → [1024] (add epsilon, e.g., 1e-6)
           - rsqrt: [1024] → [1024] (1/sqrt, per token)
           - .mean(): [1024] → scalar (mean over all tokens)
           Example result: scalar value like 0.95
        
        3. Set layernorm weight: full_like(weight.shape, computed_value)
           - q_a_layernorm.weight: [512] → all elements set to computed scalar
           Example: [512] filled with 0.95
        
        Process for kv_a_layernorm (always executed):
        Same as above but with kv_a_outputs:
        - Concatenate: [total_tokens, 576]
        - Compute RMS: scalar value
        - Set kv_a_layernorm.weight: [576] filled with computed scalar
    """
    if q_a_outputs is not None:
        self_attn.q_a_layernorm.weight.data.to(self_attn.q_a_proj.weight.device).to(self_attn.dtype)
        q_a_proj = torch.cat(q_a_outputs)
        q_a_rmsnorm = torch.rsqrt(q_a_proj.pow(2).mean(-1) + self_attn.q_a_layernorm.eps).mean()
        self_attn.q_a_layernorm.weight.data = torch.full_like(self_attn.q_a_layernorm.weight.data, q_a_rmsnorm)

    self_attn.kv_a_layernorm.weight.data.to(self_attn.kv_a_proj_with_mqa.weight.device).to(self_attn.dtype)
    kv_a_proj = torch.cat(kv_a_outputs)
    kv_a_rmsnorm = torch.rsqrt(kv_a_proj.pow(2).mean(-1) + self_attn.kv_a_layernorm.eps).mean()
    self_attn.kv_a_layernorm.weight.data = torch.full_like(self_attn.kv_a_layernorm.weight.data, kv_a_rmsnorm)
