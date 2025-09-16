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

from node0.models.arguments import ModelArguments


class LlamaArguments(ModelArguments):
    hidden_dim: int = 4096
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 50265  # Using AutoTokenizer.from_pretrained("facebook/opt-2.7b")
    # vocab_size: int = 50280  # Using AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 512
    depth_init: bool = True
    constant_init: bool = False
    norm_type: str = "rmsnorm"
