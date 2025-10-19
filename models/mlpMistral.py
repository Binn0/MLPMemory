from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import MistralPreTrainedModel
from transformers.modeling_utils import Conv1D, PreTrainedModel
from transformers.activations import ACT2FN
from torch import nn
import torch

# NOTE:
# The following MLP components are adapted from
# `transformers.models.mistral.modeling_mistral` in the Hugging Face Transformers library.
# The structure and logic are consistent with the original Mistral feed-forward (MLP) design.
#
# If you wish to use the MLP architecture from another model (e.g., LLaMA, GPT2, Qwen, etc.),
# you can refer to the corresponding implementation under `transformers.models.<model_name>.modeling_<model_name>`.

class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class MistralMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MistralMLP(config)

    def forward(self, x, **kwargs):
        residual = x
        x = self.input_layernorm(x)
        x = self.mlp(x)
        return residual + x

class MistralMLPModel(MistralPreTrainedModel):
    def __init__(self, config, input_dim=4096, output_dim=4096):
        super().__init__(config)

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        if self.input_dim != self.output_dim:
            self.embedding_transform = nn.Linear(self.input_dim, self.output_dim)

        self.layers = nn.ModuleList([
            MistralMLPBlock(config) for _ in range(config.num_hidden_layers)
        ])

        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, inputs_embeds=None, position_ids=None, labels=None):
        if self.input_dim != self.output_dim:
            inputs_embeds = self.embedding_transform(inputs_embeds)

        hidden_states = inputs_embeds

        for block in self.layers:
            hidden_states = block(hidden_states)

        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)  # Shape: (bsz, vocab_size)

        return logits
