# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0
#
# This file includes third-party code:
#
# - `DomynEdgeStateDictAdapter` is adapted from torchtitan
#   (https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model/state_dict_adapter.py),
#   Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#   Licensed under the BSD 3-Clause License (https://github.com/pytorch/torchtitan/blob/main/LICENSE).
#   Modified for DomynEdge.

"""This script is adapted from torchtitan/models/llama3/model/state_dict_adapter.py.

We can use this script to adapt the checkpoint from HF to the format that we can load into the torchtitan model
and vice versa. This can enable us to do a parity test with the HF implementation and make sure that our results are
aligned with the HF implementation.

"""

import re
from typing import Any

from torchtitan.models.utils import MoEStateDictAdapter

from .model import DomynEdgeModel


class DomynEdgeStateDictAdapter(MoEStateDictAdapter):
    """Convert DomynEdge state dicts between the HF and torchtitan key layouts."""

    def __init__(self, model_config: DomynEdgeModel.Config, hf_assets_path: str | None):
        """Set up the HF -> torchtitan key mapping.

        Args:
            model_config: DomynEdge model config.
            hf_assets_path: Path to the HF assets, forwarded to the base adapter.
        """
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            "model.layers.{}.attention.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.attention.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.attention.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.attention.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.attention.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.attention.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # MLP module for non-MoE
            "model.layers.{}.feedforward.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.feedforward.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.feedforward.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Transformer layer
            "model.layers.{}.pre_attention_norm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_norm.weight": "layers.{}.post_attn_norm.weight",
            "model.layers.{}.pre_feedforward_norm.weight": "layers.{}.ffn_norm.weight",
            "model.layers.{}.post_feedforward_norm.weight": "layers.{}.post_ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a torchtitan state dict to HF keys.

        Keys without an HF mapping are dropped, as is ``lm_head.weight`` when weight tying is enabled.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_match = re.search(r"\d+", key)
                assert layer_match is not None
                layer_num = layer_match.group(0)
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                if key not in to_hf_map:
                    continue
                if self.model_config.enable_weight_tying and key == "lm_head.weight":
                    continue
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert an HF state dict to torchtitan keys.

        With weight tying enabled and no ``lm_head.weight`` present, it is filled from
        ``model.embed_tokens.weight``.
        """
        state_dict = {}

        if self.model_config.enable_weight_tying and "lm_head.weight" not in hf_state_dict:
            assert "model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_match = re.search(r"\d+", key)
                assert layer_match is not None
                layer_num = layer_match.group(0)
                new_key = self.from_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value

        return state_dict
