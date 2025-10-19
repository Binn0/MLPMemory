from typing import Optional, Tuple
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from transformers import (
    GenerationMixin,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoConfig,
    StoppingCriteriaList,
    GenerationConfig,
)
from transformers.utils import ModelOutput


@dataclass
class MLPMemoryOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None         # Combined log-probabilities (eval mode)
    knn_logits: Optional[torch.FloatTensor] = None     # Raw KNN logits (train mode)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class _ActivationCapturer:
    """Captures the input or output of a specific layer via forward hook."""
    def __init__(self, capture_input: bool = True):
        self.capture_input = capture_input
        self.captured = None

    def __call__(self, module, inputs, outputs):
        self.captured = inputs[0] if self.capture_input else outputs
        return outputs


class MLPMemory(PreTrainedModel, GenerationMixin):
    """
    Unified MLP-Memory model.

    Modes:
    - **Training mode (`is_train=True`)**:
        - Only trains the `knn_generator`.
        - The `base_lm` is frozen.
        - Returns only `knn_logits`.

    - **Evaluation / Inference mode (`is_train=False`)**:
        - Fuses the base LM and KNN generator outputs via interpolation.
        - Returns fused log-probabilities and (optional) loss.
    """

    MODEL_TO_LAYER_STACK = {
        "gpt2": lambda base: (base.base_model.h,          lambda layer: layer.mlp),
        "llama": lambda base: (base.base_model.layers,    lambda layer: layer.mlp),
        "mistral": lambda base: (base.base_model.layers,  lambda layer: layer.mlp),
    }

    def __init__(
        self,
        base_lm,
        knn_generator,
        lmbda: float = 0.25,
        knn_temp: float = 1.0,
        layer_position: int = -1,
        is_train: bool = False,
    ):
        super().__init__(base_lm.config)

        self.base_lm = base_lm
        self.knn_generator = knn_generator
        self.lmbda = float(lmbda)
        self.knn_temp = float(knn_temp)
        self.layer_position = int(layer_position)
        self.is_train = bool(is_train)

        # Register forward hook to capture MLP layer activations
        self._capturer = _ActivationCapturer(capture_input=True)
        self._hook_handle = None
        self._setup_activation_capture()

        # Freeze parameters according to mode
        self._apply_train_mode_freeze()

    # ================== Internal Helpers =======================
    def _apply_train_mode_freeze(self):
        """Freeze/unfreeze modules based on whether we are in training mode."""
        if self.is_train:
            for p in self.base_lm.parameters():
                p.requires_grad = False
            for p in self.knn_generator.parameters():
                p.requires_grad = True
        else:
            for p in self.base_lm.parameters():
                p.requires_grad = False
            for p in self.knn_generator.parameters():
                p.requires_grad = False

    def _get_layer_stack_and_accessor(self):
        """Return the layer list and accessor for the current model type."""
        mtype = self.base_lm.config.model_type
        if mtype not in self.MODEL_TO_LAYER_STACK:
            raise KeyError(f"Unsupported model type: {mtype}")
        return self.MODEL_TO_LAYER_STACK[mtype](self.base_lm)

    def _resolve_layer_by_position(self, position):
        """Resolve the MLP layer at the specified position (supports negative index)."""
        layers, get_mlp = self._get_layer_stack_and_accessor()
        L = len(layers)
        idx = position if position >= 0 else (L + position)
        return get_mlp(layers[idx])

    def _setup_activation_capture(self):
        """Attach a forward hook to capture the hidden activation at a specific layer."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
        layer = self._resolve_layer_by_position(self.layer_position)
        self._hook_handle = layer.register_forward_hook(self._capturer)

    # ================== Forward Pass ===========================
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        Forward pass.

        **Training mode** (`is_train=True`):
        - Only the KNN generator is trained.
        - Base LM is frozen and evaluated under `torch.no_grad()`.
        - Returns only `knn_logits`.

        **Evaluation mode** (`is_train=False`):
        - Combines base LM and KNN generator logits using interpolation:
              logp_joint = logaddexp(
                  log((1 - λ) * p_base),
                  log(λ * p_knn)
              )
        - Optionally computes loss if labels are provided.
        """
        # 1. Base LM Forward
        self.base_lm.eval()
        with torch.no_grad():
            if self.is_train:
                base_outputs = self.base_lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    **kwargs,
                )
            else:
                base_outputs = self.base_lm(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  past_key_values=past_key_values,
                  use_cache=use_cache,
                  output_hidden_states=True,
                  **kwargs,
                )

        hidden_states = self._capturer.captured

        if hidden_states is None:
            raise RuntimeError("Failed to capture hidden states. Check `layer_position` value.")

        # 2. Flatten Hidden States
        if self.is_train:
            nonpad_mask = torch.cat([
                labels[:, 1:] != -100,
                torch.zeros([labels.shape[0], 1], dtype=torch.bool).to(hidden_states.device)
            ], axis=-1)
            h_flat = hidden_states[nonpad_mask] # [B*T, D]
        else:
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            B, T, D = hidden_states.shape

            h_flat = hidden_states.reshape(-1, D)

        # 3. KNN Generator Forward
        knn_logits = self.knn_generator(
            inputs_embeds=h_flat,
            position_ids=None,
            labels=None,
        )  # (B*T, V)

        # 4. Train / Eval Branch
        if self.is_train:
            # Training mode — only return KNN output
            return MLPMemoryOutput(knn_logits=knn_logits)

        knn_logits = knn_logits.view(B, T, -1)

        # Get base model logits
        logits_base = base_outputs.logits # batch_siz(B), seq_len(T), vocab_size(V)

        if self.knn_temp != 1.0:
            knn_logits = knn_logits / self.knn_temp

        # 5. Interpolation and Loss
        logp_base = F.log_softmax(logits_base, dim=-1)
        logp_knn = F.log_softmax(knn_logits, dim=-1)

        logp_joint = torch.logaddexp(
            logp_base + torch.log(torch.tensor(1.0 - self.lmbda, device=logp_base.device)),
            logp_knn + torch.log(torch.tensor(self.lmbda, device=logp_base.device)),
        )

        # Optional loss (useful for perplexity evaluation)
        loss = None
        if labels is not None:
            shift_logprobs = logp_joint[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.NLLLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logprobs.view(-1, shift_logprobs.size(-1)),
                shift_labels.view(-1)
            )

        return MLPMemoryOutput(
            loss=loss,
            logits=logp_joint,
            past_key_values=base_outputs.past_key_values,
            hidden_states=base_outputs.hidden_states if hasattr(base_outputs, 'hidden_states') else None,
            attentions=base_outputs.attentions if hasattr(base_outputs, 'attentions') else None,
        )

    # ================== Generation ============================
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 20,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        do_sample: bool = False,   # must be False (greedy) for now
        use_cache: bool = True,
        generation_config: Optional[GenerationConfig] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        """
        Generation with hidden states as KNN input.
        Matches MLPMemory's generation loop structure.
        """
        if do_sample:
            raise ValueError("MLPMemory.generate only supports greedy decoding (do_sample=False).")

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Initial forward
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )

        next_token_logits = outputs.logits[:, -1, :]  # (B, V)
        base_past = outputs.past_key_values

        # Greedy select
        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # (B, 1)
        generated = torch.cat([input_ids, next_tokens], dim=-1)  # (B, T+1)

        # --- main loop -------------------------------------------------- #
        num_new_token = 0
        while True:
            if stopping_criteria is not None and False not in stopping_criteria(generated, None):
                break
            if num_new_token >= max_new_tokens:
                break

            outputs = self.forward(
                input_ids=next_tokens,
                attention_mask=None,  # past manages causal masking
                past_key_values=base_past,
                use_cache=use_cache,
                **kwargs,
            )

            next_token_logits = outputs.logits[:, -1, :]
            base_past = outputs.past_key_values

            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_tokens], dim=-1)
            num_new_token += 1

        return generated

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.knn_generator.save_pretrained(os.path.join(save_dir, "knn_generator"), safe_serialization=False)
        cfg = {
            "layer_position": self.layer_position,
            "is_train": self.is_train,
        }
        with open(os.path.join(save_dir, "mlp_memory_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def __del__(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
