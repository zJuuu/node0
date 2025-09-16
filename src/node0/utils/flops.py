# Copyright 2025 Pluralis Research
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

import torch

from node0.models.arguments import ModelArguments


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    """Compute number of model parameters."""
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding and hasattr(model, "tok_embeddings"):
        num_params -= model.tok_embeddings.weight.numel()

    if exclude_embedding and hasattr(model, "fixed_tok_embeddings"):
        num_params -= model.fixed_tok_embeddings.weight.numel()

    return num_params


def get_num_flop_per_token_fwd(num_params: int, model_config: ModelArguments, seq_len: int) -> int:
    """Compute number of flop per token in forward pass."""
    layers, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.hidden_dim // model_config.n_heads,
        seq_len,
    )

    flop_per_token = 2 * num_params + 4 * layers * h * q * t

    return flop_per_token


def get_num_flop_per_token_bwd(num_params: int, model_config: ModelArguments, seq_len: int) -> int:
    """Compute number of flop per token in backward pass."""
    layers, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.hidden_dim // model_config.n_heads,
        seq_len,
    )

    flop_per_token = 4 * num_params + 8 * layers * h * q * t

    return flop_per_token
