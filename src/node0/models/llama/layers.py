# This file contains code originally from Hivemind under MIT License
# Original: Copyright 2020 Learning@home authors and collaborators
# Modified by: Pluralis Research 2025
#
# Original code: MIT License (see THIRD_PARTY_LICENSES)
# Modifications: Apache 2.0 License (see LICENSE)
#
# Licensed under the Apache License, Version 2.0 (the "License") for modifications only;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from abc import abstractmethod

import torch
import torch.nn.functional as F

from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import nn


def build_norm(norm_type: str, dim: int, eps: float = 1e-6, trainable = True):
    """
    Builds the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to build.
            Supported types: layernorm, np_layernorm, rmsnorm, fused_rmsnorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The built normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps, trainable=trainable)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6, trainable = True):
        super().__init__()
        self.eps = eps
        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight if self.trainable else output

    def reset_parameters(self):
        if self.trainable:
            torch.nn.init.ones_(self.weight)  # type: ignore


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args, layer_id):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        self.qk_norm = model_args.qk_norm
        self.layer_id = layer_id
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.hidden_dim // model_args.n_heads

        self.wq = nn.Linear(model_args.hidden_dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.hidden_dim, bias=False)

        self.hidden_dim = model_args.hidden_dim
        self.attn_proj = model_args.attn_proj

        # QK-norm   
        if self.qk_norm:
            self.q_norm = build_norm(
                model_args.norm_type, 
                self.head_dim, 
                eps=model_args.norm_eps,
                trainable=model_args.trainable_rmsnorm
                )
            self.k_norm = build_norm(
                model_args.norm_type, 
                self.head_dim, 
                eps=model_args.norm_eps,
                trainable=model_args.trainable_rmsnorm                
                )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        if self.attn_proj:
            rng = torch.Generator()
            rng.manual_seed(1337 + self.layer_id)

            Rk = torch.randn(
                (self.n_kv_heads * self.head_dim, int(self.n_kv_heads * self.head_dim / 4)),
                dtype=torch.float32,
                generator=rng,
            )
            self.Rk, _ = torch.linalg.qr(Rk, mode="reduced")

            Rq = torch.randn((self.hidden_dim, int(self.hidden_dim / 4)), dtype=torch.float32, generator=rng)
            self.Rq, _ = torch.linalg.qr(Rq, mode="reduced")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.attn_proj:
            self.Rk = self.Rk.to(xq.device)
            self.Rq = self.Rq.to(xq.device)
            xq = xq @ self.Rq[None, :, :] @ self.Rq[None, :, :].transpose(1, 2)
            xk = xk @ self.Rk[None, :, :] @ self.Rk[None, :, :].transpose(1, 2)
            xv = xv @ self.Rk[None, :, :] @ self.Rk[None, :, :].transpose(1, 2)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Normalize across the head dimension (last dimension)
        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.hidden_dim
        self.norm_reorder = model_args.norm_reorder
        self.trainable_rmsnorm = model_args.trainable_rmsnorm        
        self.attention = Attention(model_args, layer_id)
        self.feed_forward = FeedForward(
            dim=model_args.hidden_dim,
            hidden_dim=4 * model_args.hidden_dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = build_norm(
            model_args.norm_type, 
            dim=model_args.hidden_dim, 
            eps=model_args.norm_eps,
            trainable=self.trainable_rmsnorm,
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, 
            dim=model_args.hidden_dim, 
            eps=model_args.norm_eps,
            trainable=self.trainable_rmsnorm,
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        elif model_args.constant_init:
            self.weight_init_std = 0.02
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        if self.norm_reorder:
            pre_attention = self.attention_norm(self.attention(x, freqs_cis))
        else:
            pre_attention = self.attention(self.attention_norm(x), freqs_cis)

        h = x + pre_attention

        ffn_in = h
        if self.norm_reorder:
            ffn_out = self.ffn_norm(
                self.feed_forward(ffn_in)
            )
        else:
            ffn_out = self.feed_forward(
                self.ffn_norm(ffn_in)
            )
        out = h + ffn_out
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


head_sample_input = lambda batch_size, sequence_length, hid_dim: (
    torch.randint(low=0, high=1000, size=(batch_size, sequence_length), dtype=torch.long),
)

body_sample_input = lambda batch_size, sequence_length, hid_dim: (torch.empty((batch_size, sequence_length, hid_dim)),)

tail_sample_input = lambda batch_size, sequence_length, hid_dim: (
    torch.empty((batch_size, sequence_length, hid_dim)),
    torch.randint(0, 1000, (batch_size, sequence_length), dtype=torch.long),
)


class BaseExpert(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        self.n_layers = model_args.n_layers

        # Precompute frequency tensor for positional embeddings
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # Initialize common compression components
        if model_args.use_compression:
            self.forward_compression_list = [True] * model_args.n_layers

            self.compression_length = int(model_args.hidden_dim // model_args.compression_rate)
            self.fixed_tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.hidden_dim)
            self.rcv = nn.Parameter(torch.empty(model_args.hidden_dim, self.compression_length), requires_grad=False)

        # Layers dictionary
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            layer_id_global = model_args.stage * model_args.n_layers + layer_id
            self.layers[str(layer_id)] = TransformerBlock(layer_id_global, model_args)

    def initialize_layers(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        Initialize weights for the base expert model.
        All models store precomputed freqs_cis and transformer blocks
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()

        # Initialize fixed token embeddings
        if self.model_args.use_compression and self.fixed_tok_embeddings is not None:
            nn.init.normal_(self.fixed_tok_embeddings.weight)
            self.fixed_tok_embeddings.weight.requires_grad = False

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.hidden_dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def compress_output(self, output):
        rcv = self.rcv.unsqueeze(0).clone()

        # Extract output and token indices
        x = output[:, :, :-1]
        idx = output[:, :, -1:]
        tokens = idx.to(torch.int).squeeze(2)

        # Get fixed embeddings
        fixed_embed = self.fixed_tok_embeddings(tokens)

        # Compress: compressed ≈ rcv.T @ (output - fixed_embeddings)
        compressed_output = (rcv.transpose(2, 1) @ (x - fixed_embed).transpose(2, 1)).transpose(2, 1)

        # Concatenate token indices back to the compressed output
        return torch.cat([compressed_output, output[:, :, -1:]], dim=-1)

    def decompress_input(self, input):
        rcv = self.rcv.unsqueeze(0).clone()

        # Extract compressed representation and token indices
        x = input[:, :, :-1].transpose(2, 1)
        idx = input[:, :, -1:]
        tokens = idx.to(torch.int).squeeze(2).clone()

        # Get fixed embeddings
        fixed_embed = self.fixed_tok_embeddings(tokens)

        # Decompress: h ≈ rcv @ compressed + fixed_embeddings
        decompressed_output = (rcv @ x + fixed_embed.transpose(2, 1)).transpose(2, 1)

        # Concatenate the tokens to the decompressed output
        return torch.cat([decompressed_output, input[:, :, -1:]], dim=-1)

    def load_comp(self, ss_comps):
        # Load RCV
        self.rcv.data = ss_comps["rcv"].clone()

        # Copy embedding weights
        self.fixed_tok_embeddings.weight.data = ss_comps["fixed_tok_weight"].clone()
        self.fixed_tok_embeddings.weight.requires_grad = False

    def ss_regularize(self):
        # Regularize attention and feedforward weights
        self.rcv.data = self.rcv.data.contiguous()
        with torch.no_grad():
            for i, layer in enumerate(self.layers.values()):
                if self.forward_compression_list[i]:
                    layer.attention.wo.weight.data = self.rcv @ (self.rcv.T @ layer.attention.wo.weight.data)
                    layer.attention.wo.weight.data = layer.attention.wo.weight.data.contiguous()
                    layer.feed_forward.w2.weight.data = self.rcv @ (self.rcv.T @ layer.feed_forward.w2.weight.data)
                    layer.feed_forward.w2.weight.data = layer.feed_forward.w2.weight.data.contiguous()

    @abstractmethod
    def forward(self, input_ids):
        """
        Abstract forward method that must be implemented by each child class.
        This ensures each expert handles its specific input processing.
        """
        raise NotImplementedError("Each expert must implement its own forward method")


@register_expert_class("lm_head", head_sample_input)
class HeadExpert(BaseExpert):
    def __init__(self, model_args):
        super().__init__(model_args)

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.hidden_dim)

        self.init_weights()

    def init_weights(self):
        # Initialize base layers
        self.initialize_layers()

        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

    def ss_regularize(self):
        # For head expert call weight regularization and also regularize tok_embeddings
        super().ss_regularize()
        with torch.no_grad():
            self.tok_embeddings.weight.data = (self.rcv @ (self.rcv.T @ self.tok_embeddings.weight.data.T)).T
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.contiguous()

    def forward(self, input_ids):
        hidden_states = self.tok_embeddings(input_ids)

        if self.model_args.use_compression:
            hidden_states += self.fixed_tok_embeddings(input_ids)

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis)

        if self.model_args.use_compression:
            hidden_states = torch.cat([hidden_states, input_ids.unsqueeze(2)], dim=-1)
            hidden_states = self.compress_output(hidden_states)

        return hidden_states


@register_expert_class("lm_body", body_sample_input)
class BodyExpert(BaseExpert):
    def __init__(self, model_args):
        super().__init__(model_args)

        self.initialize_layers()

    def forward(self, hidden_states):
        if self.model_args.use_compression:
            hidden_states = self.decompress_input(hidden_states)
            # Extract token indices for later use
            token_indices = hidden_states[:, :, -1:]
            hidden_states = hidden_states[:, :, :-1]

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis)

        if self.model_args.use_compression:
            # Reattach token indices and compress
            hidden_states = torch.cat([hidden_states, token_indices], dim=-1)
            hidden_states = self.compress_output(hidden_states)

        return hidden_states


@register_expert_class("lm_tail", tail_sample_input)
class TailExpert(BaseExpert):
    def __init__(self, model_args):
        model_args.attn_proj = False  # Attention projection is false on tail
        super().__init__(model_args)

        # The last layers are not compressed
        if model_args.use_compression:
            self.forward_compression_list = [True] * (model_args.n_layers - 2) + [False, False]

        self.norm = build_norm(model_args.norm_type, dim=model_args.hidden_dim, eps=model_args.norm_eps)

        self.output = nn.Linear(model_args.hidden_dim, model_args.vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        # Initialize base layers
        self.initialize_layers()

        self.norm.reset_parameters()
        final_out_std = self.model_args.hidden_dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def forward(self, hidden_states, labels):
        if self.model_args.use_compression:
            hidden_states = self.decompress_input(hidden_states)
            hidden_states = hidden_states[:, :, :-1]  # Remove tokens

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis)

        hidden_states = self.norm(hidden_states)
        lm_logits = self.output(hidden_states)

        lm_logits = lm_logits.contiguous()
        labels = labels.contiguous()

        loss = F.cross_entropy(lm_logits.permute(0, 2, 1), labels, reduction="none")

        return loss
