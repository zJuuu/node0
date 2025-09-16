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

from typing import Self

from pydantic import BaseModel, model_validator


class ModelArguments(BaseModel):
    # Model parameters
    hidden_dim: int
    n_heads: int
    num_hidden_layers: int
    n_layers: int = 0
    stage: int | None = None

    # Attention projection parameters
    attn_proj: bool = False

    # QK norm
    qk_norm: bool = False
    norm_reorder: bool = False
    trainable_rmsnorm: bool = True

    # Compression parameters
    compression_rate: int | None = None
    use_compression: bool = False
    ss_component: str | None

    @model_validator(mode="after")
    def set_n_layers(self) -> Self:
        self.n_layers = self.num_hidden_layers
        return self
