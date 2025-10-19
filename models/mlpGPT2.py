from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Config, GPT2Model
from transformers.modeling_utils import Conv1D, PreTrainedModel
from transformers.activations import ACT2FN
from torch import nn
import torch

# NOTE:
# The following MLP components are adapted from
# `transformers.models.gpt2.modeling_gpt2` in the Hugging Face Transformers library.
# The structure and logic are consistent with the original GPT2 feed-forward (MLP) design.
#
# If you wish to use the MLP architecture from another model (e.g., Mistral, Llama, Qwen, etc.),
# you can refer to the corresponding implementation under `transformers.models.<model_name>.modeling_<model_name>`.

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.n_embd
        intermediate_size = 4 * embed_dim

        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x, **kwargs):
        residual = x
        x = self.ln_1(x.float())
        x = self.mlp(x)
        return residual + x

class GPT2MLPModel(GPT2PreTrainedModel):
    def __init__(self, config, input_dim=1024, output_dim=1024):
        super().__init__(config)

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        if self.input_dim != self.output_dim:
            self.embedding_transform = nn.Linear(self.input_dim, self.output_dim)

        self.h = nn.ModuleList([
            GPT2MLPBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, inputs_embeds=None, position_ids=None, labels=None):
        if self.input_dim != self.output_dim:
            inputs_embeds = self.embedding_transform(inputs_embeds)

        hidden_states = inputs_embeds

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        logits = self.lm_head(hidden_states)  # Shape: (bsz, vocab_size)

        return logits
