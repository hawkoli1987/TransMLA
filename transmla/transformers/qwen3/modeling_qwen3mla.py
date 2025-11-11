from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model,
    Qwen3DecoderLayer,
    Qwen3PreTrainedModel,
    Qwen3ForCausalLM,
)

from .configuration_qwen3mla import Qwen3MLAConfig
from .mla import MLAAttention, eager_attention_forward


class Qwen3MLADecoderLayer(Qwen3DecoderLayer):

    def __init__(self, config: Qwen3MLAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MLAAttention(config, layer_idx)


class Qwen3MLAPreTrainedModel(Qwen3PreTrainedModel):

    config_class = Qwen3MLAConfig
    _no_split_modules = ["Qwen3MLADecoderLayer"]


class Qwen3MLAModel(Qwen3MLAPreTrainedModel, Qwen3Model):

    def __init__(self, config: Qwen3MLAConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [Qwen3MLADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Qwen3MLAForCausalLM(Qwen3MLAPreTrainedModel, Qwen3ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3MLAModel(config)


__all__ = [
    "Qwen3MLAForCausalLM",
    "Qwen3MLAModel",
    "Qwen3MLAPreTrainedModel",
]
