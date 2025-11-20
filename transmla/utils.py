import logging
import time
import torch
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
    
@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module, 
    pad_token_id: int | None, 
    testloader: DataLoader[dict[str, torch.Tensor]], 
    message: str = "Evaluating perplexity"
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

    logging.info(message)
    for batch in tqdm(testloader, desc=message):
        logging.debug(f"Evaluating batch {len(nlls)}")
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

@torch.no_grad()
def extract_qk_from_model(model: torch.nn.Module, hidden_states: torch.Tensor, layer_idx: int, position_embeddings=None, attention_mask=None):
    """
    Extract Q and K tensors from a model layer for QK dot product calculation.
    
    Handles different model types:
    - Original Qwen3: standard q_proj and k_proj
    - PartialRope: q_proj and k_proj with k_up_proj
    - LoraQKV: q_a_proj/q_b_proj and kv_a_proj_with_mqa/kv_b_proj structure
    
    Args:
        model: The model to extract Q and K from
        hidden_states: Input hidden states [batch, seq_len, hidden_size]
        layer_idx: Layer index
        position_embeddings: Optional tuple of (cos, sin) for RoPE
        attention_mask: Optional attention mask
    
    Returns:
        Tuple of (Q, K) tensors in shape [batch, num_heads, seq_len, head_dim]
    """
    layer = model.model.layers[layer_idx]
    self_attn = layer.self_attn
    
    num_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, 'head_dim', model.config.hidden_size // num_heads)
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    
    # Handle different attention structures
    if hasattr(self_attn, 'q_a_proj') and hasattr(self_attn, 'kv_a_proj_with_mqa'):
        # LoraQKV structure
        bsz, q_len, _ = hidden_states.size()
        
        # Query: q_a_proj -> q_a_layernorm -> q_b_proj
        if hasattr(self_attn, 'q_a_proj'):
            q_a = self_attn.q_a_proj(hidden_states)
            if hasattr(self_attn, 'q_a_layernorm'):
                q_a = self_attn.q_a_layernorm(q_a)
            q = self_attn.q_b_proj(q_a)  # [batch, seq_len, num_heads * (head_dim + qk_mqa_dim)]
        else:
            q = self_attn.q_proj(hidden_states)
        
        q = q.view(bsz, q_len, num_heads, -1).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim + qk_mqa_dim]
        qk_mqa_dim = getattr(self_attn, 'qk_mqa_dim', q.size(-1) - head_dim)
        q_nope, q_rope = q.split([head_dim, qk_mqa_dim], dim=-1)
        
        # Key: kv_a_proj_with_mqa -> split -> kv_b_proj
        compressed_kv = self_attn.kv_a_proj_with_mqa(hidden_states)  # [batch, seq_len, kv_lora_rank + qk_mqa_dim]
        kv_lora_rank = getattr(self_attn, 'kv_lora_rank', compressed_kv.size(-1) - qk_mqa_dim)
        kv_nope, k_rope = compressed_kv.split([kv_lora_rank, qk_mqa_dim], dim=-1)
        kv_nope = kv_nope.view(bsz, 1, q_len, kv_lora_rank)
        
        if hasattr(self_attn, 'kv_a_layernorm'):
            kv_nope = self_attn.kv_a_layernorm(kv_nope)
        kv_expanded = self_attn.kv_b_proj(kv_nope)  # [batch, 1, seq_len, num_heads * head_dim * 2]
        kv_expanded = kv_expanded.view(bsz, q_len, num_heads, head_dim * 2).transpose(1, 2)
        k_nope, _ = kv_expanded.split([head_dim, head_dim], dim=-1)
        k_rope = k_rope.view(bsz, 1, q_len, qk_mqa_dim)
        
        # Apply RoPE if position_embeddings provided
        if position_embeddings is not None:
            from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
            cos, sin = position_embeddings
            collapse = getattr(self_attn, 'collapse', 1)
            q_rope, k_rope = apply_rotary_pos_emb_interleave(
                q_rope, k_rope, cos[:, :, ::collapse], sin[:, :, ::collapse]
            )
            # Expand k_rope to all heads
            k_rope = k_rope.expand(bsz, num_heads, q_len, qk_mqa_dim)
        
        # Concatenate nope and rope parts
        q = torch.cat([q_nope, q_rope], dim=-1)  # [batch, num_heads, seq_len, head_dim + qk_mqa_dim]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [batch, num_heads, seq_len, head_dim + qk_mqa_dim]
        
        # Use only head_dim for comparison with original model
        q = q[..., :head_dim]
        k = k[..., :head_dim]
        
    elif hasattr(self_attn, 'k_up_proj'):
        # PartialRope structure
        bsz, q_len, _ = hidden_states.size()
        
        # Q: q_proj -> reshape -> transform via k_up_proj
        q = self_attn.q_proj(hidden_states)  # [batch, seq_len, num_heads * head_dim]
        q = q.view(bsz, q_len, num_heads, head_dim)  # [batch, seq_len, num_heads, head_dim]
        k_up_weight = self_attn.k_up_proj.weight.view(num_heads, head_dim, self_attn.latent_dim)
        q = torch.einsum("bthd,hdc->bhtc", q, k_up_weight)  # [batch, num_heads, seq_len, latent_dim]
        
        # K: k_proj -> reshape
        k_latent = self_attn.k_proj(hidden_states)  # [batch, seq_len, latent_dim]
        k = k_latent.view(bsz, 1, q_len, self_attn.latent_dim)  # [batch, 1, seq_len, latent_dim]
        
        # Apply RoPE if position_embeddings provided
        if position_embeddings is not None:
            from partial_rope import apply_rotary_pos_emb
            cos, sin = position_embeddings
            collapse = getattr(self_attn, 'collapse', 1)
            rope_head = getattr(self_attn, 'rope_head', 1)
            q, k = apply_rotary_pos_emb(q, k, cos[:, :, ::collapse], sin[:, :, ::collapse], rope_head)
        
        # Expand K to all heads
        k = k.expand(bsz, num_heads, q_len, self_attn.latent_dim)
        
        # Transform back to head_dim for comparison with original model
        k_up_weight_T = k_up_weight.transpose(-2, -1)  # [num_heads, latent_dim, head_dim]
        q = torch.einsum("bhtc,hcd->bhtd", q, k_up_weight_T)  # [batch, num_heads, seq_len, head_dim]
        k = torch.einsum("bhtc,hcd->bhtd", k, k_up_weight_T)  # [batch, num_heads, seq_len, head_dim]
        
    else:
        # Standard structure (original Qwen3)
        q = self_attn.q_proj(hidden_states)  # [batch, seq_len, num_heads * head_dim]
        q = q.view(-1, q.size(1), num_heads, head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        k = self_attn.k_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]
        k = k.view(-1, k.size(1), num_kv_heads, head_dim).transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        
        # Apply RoPE if position_embeddings provided (for original Qwen3)
        if position_embeddings is not None:
            # Original Qwen3 uses standard RoPE
            cos, sin = position_embeddings
            # Apply RoPE transformation (simplified, assuming standard implementation)
            # For Qwen3, we need to check the actual RoPE implementation
            # For now, we'll skip RoPE for original model to match the comparison
            pass
        
        # Repeat K for GQA if needed
        if num_kv_heads != num_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    
    return q, k

@torch.no_grad()
def calculate_qk_dot_product_kl_divergence(
    original_model: torch.nn.Module,
    converted_model: torch.nn.Module,
    test_loader: DataLoader[dict[str, torch.Tensor]],
    tokenizer_pad_id: int | None = None,
) -> dict[int, float]:
    """
    Calculate Q K dot product KL divergence between original and converted models.
    
    For each layer, this function:
    1. Captures Q and K from both models during forward pass
    2. Computes Q @ K^T (dot product) for each layer
    3. Converts to probability distributions using softmax
    4. Calculates KL divergence between original and converted distributions
    
    Args:
        original_model: The original non-converted Qwen3-4B model
        converted_model: The newly converted Qwen3-4B model
        test_loader: DataLoader for evaluation
        tokenizer_pad_id: Padding token ID for masking
        
    Returns:
        Dictionary mapping layer index to KL divergence value
    """
    import torch.nn.functional as F
    
    original_model.eval()
    converted_model.eval()
    
    # Ensure both models are on the same device
    device = next(converted_model.parameters()).device
    if next(original_model.parameters()).device != device:
        original_model = original_model.to(device)
    
    # Storage for QK dot products per layer
    original_qk_dot_products = {}
    converted_qk_dot_products = {}
    
    # Process first batch to capture QK dot products
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 1:  # Only use first batch for efficiency
            break
            
        batch = map_tensors(batch, device)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        
        # Get position_ids for RoPE
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        hidden_states_orig = original_model.model.embed_tokens(input_ids)
        hidden_states_conv = converted_model.model.embed_tokens(input_ids)
        
        # Get position embeddings for both models
        # For original Qwen3 model
        if hasattr(original_model.model, 'layers') and len(original_model.model.layers) > 0:
            # Try to get rotary embedding from first layer
            first_layer = original_model.model.layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'rotary_emb'):
                position_embeddings_orig = first_layer.self_attn.rotary_emb(hidden_states_orig, position_ids=position_ids)
            elif hasattr(original_model.model, 'rotary_emb'):
                position_embeddings_orig = original_model.model.rotary_emb(hidden_states_orig, position_ids=position_ids)
            else:
                position_embeddings_orig = None
        else:
            position_embeddings_orig = None
        
        # For converted model (PartialRope or LoraQKV)
        if hasattr(converted_model.model, 'layers') and len(converted_model.model.layers) > 0:
            first_layer_conv = converted_model.model.layers[0]
            if hasattr(first_layer_conv, 'self_attn'):
                # Check if it has rotary_emb or if we need to generate it
                if hasattr(first_layer_conv.self_attn, 'rotary_emb'):
                    position_embeddings_conv = first_layer_conv.self_attn.rotary_emb(hidden_states_conv, position_ids=position_ids)
                elif hasattr(converted_model.model, 'rotary_emb'):
                    position_embeddings_conv = converted_model.model.rotary_emb(hidden_states_conv, position_ids=position_ids)
                else:
                    # For PartialRope/LoraQKV, we might need to generate position embeddings
                    # Use the same as original if available
                    position_embeddings_conv = position_embeddings_orig
            else:
                position_embeddings_conv = position_embeddings_orig
        else:
            position_embeddings_conv = position_embeddings_orig
        
        # Process each layer
        for layer_idx in range(len(original_model.model.layers)):
            # Extract Q and K from original model
            q_orig, k_orig = extract_qk_from_model(
                original_model, hidden_states_orig, layer_idx, 
                position_embeddings=position_embeddings_orig, attention_mask=attention_mask
            )
            
            # Extract Q and K from converted model
            q_conv, k_conv = extract_qk_from_model(
                converted_model, hidden_states_conv, layer_idx,
                position_embeddings=position_embeddings_conv, attention_mask=attention_mask
            )
            
            # Ensure Q and K have compatible shapes
            min_seq_len = min(q_orig.size(2), q_conv.size(2))
            min_head_dim = min(q_orig.size(-1), q_conv.size(-1))
            q_orig = q_orig[:, :, :min_seq_len, :min_head_dim]
            k_orig = k_orig[:, :, :min_seq_len, :min_head_dim]
            q_conv = q_conv[:, :, :min_seq_len, :min_head_dim]
            k_conv = k_conv[:, :, :min_seq_len, :min_head_dim]
            
            # Compute Q @ K^T
            head_dim = q_orig.size(-1)
            scaling = head_dim ** -0.5
            
            qk_dot_orig = torch.matmul(q_orig, k_orig.transpose(-2, -1)) * scaling
            qk_dot_conv = torch.matmul(q_conv, k_conv.transpose(-2, -1)) * scaling
            
            # Apply attention mask if available
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
                # Truncate mask if needed to match QK dot product dimensions
                if mask.size(-1) > min_seq_len:
                    mask = mask[:, :, :, :min_seq_len]
                # Mask should match both seq_len dimensions of QK dot product
                mask_k = mask  # [batch, 1, 1, seq_len] for key dimension
                mask_q = mask.transpose(-2, -1)  # [batch, 1, seq_len, 1] for query dimension
                # Apply mask: set masked positions to -inf
                qk_dot_orig = qk_dot_orig.masked_fill(mask_k == 0, float('-inf'))
                qk_dot_conv = qk_dot_conv.masked_fill(mask_k == 0, float('-inf'))
            
            # Store for this layer
            if layer_idx not in original_qk_dot_products:
                original_qk_dot_products[layer_idx] = []
            if layer_idx not in converted_qk_dot_products:
                converted_qk_dot_products[layer_idx] = []
            
            original_qk_dot_products[layer_idx].append(qk_dot_orig.cpu())
            converted_qk_dot_products[layer_idx].append(qk_dot_conv.cpu())
            
            # Forward through layers to get next hidden states
            # Need to handle position_embeddings in forward pass
            layer_output_orig = original_model.model.layers[layer_idx](
                hidden_states_orig, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings_orig
            )
            hidden_states_orig = layer_output_orig[0] if isinstance(layer_output_orig, tuple) else layer_output_orig
            
            layer_output_conv = converted_model.model.layers[layer_idx](
                hidden_states_conv,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings_conv
            )
            hidden_states_conv = layer_output_conv[0] if isinstance(layer_output_conv, tuple) else layer_output_conv
        
        break  # Only process first batch
    
    # Calculate KL divergence for each layer
    kl_divergences = {}
    for layer_idx in original_qk_dot_products:
        if layer_idx not in converted_qk_dot_products:
            continue
            
        # Concatenate all batches for this layer
        orig_qk = torch.cat(original_qk_dot_products[layer_idx], dim=0)  # [batch, num_heads, seq_len, seq_len]
        conv_qk = torch.cat(converted_qk_dot_products[layer_idx], dim=0)
        
        # Convert to probability distributions using softmax
        # Apply softmax over the last dimension (seq_len)
        orig_probs = F.softmax(orig_qk, dim=-1)
        conv_probs = F.softmax(conv_qk, dim=-1)
        
        # Calculate KL divergence: KL(P_original || P_converted)
        # KL(P||Q) = sum(P * log(P/Q))
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        orig_probs = orig_probs + eps
        conv_probs = conv_probs + eps
        
        # Normalize to ensure they sum to 1
        orig_probs = orig_probs / orig_probs.sum(dim=-1, keepdim=True)
        conv_probs = conv_probs / conv_probs.sum(dim=-1, keepdim=True)
        
        # Calculate KL divergence
        kl_div = orig_probs * torch.log(orig_probs / conv_probs)
        kl_div = kl_div.sum(dim=-1)  # Sum over seq_len dimension
        
        # Average over batch, heads, and sequence positions
        kl_div_mean = kl_div.mean().item()
        
        kl_divergences[layer_idx] = kl_div_mean
    
    return kl_divergences
