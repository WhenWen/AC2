# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by the AC2 authors (2026) to implement AC2; see src/verl/README.md for the list of changed files.

from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    # Per-token distillation outputs computed inside the fused forward (OPSD nitrobrew path);
    # None unless the engine passed nb_teacher_hidden. See _maybe_nitrobrew_fused_aux.
    fused_linear_aux: Optional[object] = None


def _maybe_nitrobrew_fused_aux(self, hidden_states: torch.Tensor, loss_kwargs: dict):
    """OPSD fused distillation: full-vocab forward KL computed from hidden states on BOTH sides
    (student = hidden @ lm_head.T, teacher = nb_teacher_hidden @ nb_teacher_unembed.T), chunked
    over the vocab — the (N, V) student logits never materialize. Runs here, inside the model
    forward, because this is where FSDP has the lm_head parameters gathered.

    Activated by the engine passing nb_* kwargs (transformer_impl.prepare_model_inputs)."""
    nb_teacher_hidden = loss_kwargs.get("nb_teacher_hidden")
    if nb_teacher_hidden is None:
        return None
    from verl.trainer.distillation.fsdp.nitrobrew_loss import compute_fused_nitrobrew_aux

    return compute_fused_nitrobrew_aux(
        hidden_states=hidden_states,
        lm_head_weight=self.lm_head.weight,
        teacher_hidden=nb_teacher_hidden,
        teacher_unembed=loss_kwargs["nb_teacher_unembed"],
        response_index=loss_kwargs.get("nb_response_index"),
        token_clip=loss_kwargs.get("nb_token_clip"),
        temperature=loss_kwargs.get("nb_kd_temperature", 1.0),
        log_prob_min_clamp=loss_kwargs.get("nb_log_prob_min_clamp"),
    )


def forward_base_model(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> CausalLMOutputWithPast:
    r"""
    Copy paste LLaMa's forward
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/llama.py

    This function should be generic enough for all pure text models.
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    return outputs


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    nb_hidden_only: bool = False,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend has to return_dict")

    if nb_hidden_only:
        # OPSD compute_ref_hidden: decoder-only pass — skip ALL lm_head work (no FusedLinearForPPO
        # over the full sequence). The caller captures the final hidden via a hook on the
        # decoder's norm; outputs.hidden_states is also populated when output_hidden_states=True.
        #
        # The final hidden is ALSO returned directly (a reference, zero cost): FSDP1 registers
        # its pre-backward hooks on the tensors the forward RETURNS, so a training caller whose
        # loss flows only from a hook-captured intermediate never re-arms the FSDP state
        # machine and every post-backward hook then asserts
        #   "expected FORWARD_BACKWARD but current state is IDLE".
        # A backward routed through this returned tensor keeps FSDP's bookkeeping intact.
        return CausalLMOutputForPPO(
            past_key_values=outputs.past_key_values,
            hidden_states=(hidden_states,) if outputs.hidden_states is None else outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Loss calculations.
    # When the engine has already prepared globally-rolled labels (e.g. the FSDP
    # path under Ulysses SP, see issue #6068), it passes them as `shift_labels`
    # so we don't redo `torch.roll` on a sequence-parallel-sliced shard.
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        fused_linear_aux=_maybe_nitrobrew_fused_aux(self, hidden_states, loss_kwargs),
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    nb_hidden_only: bool = False,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend has to return_dict")

    if nb_hidden_only:
        # OPSD compute_ref_hidden: decoder-only pass — skip ALL lm_head work. See the torch
        # backend for details.
        return CausalLMOutputForPPO(
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Loss calculations. See `forward_with_torch_backend` for why `shift_labels`
    # takes precedence over local `torch.roll` (issue #6068).
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        fused_linear_aux=_maybe_nitrobrew_fused_aux(self, hidden_states, loss_kwargs),
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
