
# From xFormers:
#
# Copyright (c) Facebook, Inc. and its affiliates
#
#
# ===
#
# BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional, Set, Type, Union

import torch

# from /xformers/ops/common.py
def get_xformers_operator(name: str):
    def no_such_operator(*args, **kwargs):
        raise RuntimeError(
            f"No such operator xformers::{name} - did you forget to build xformers with `python setup.py develop`?"
        )

    try:
        return getattr(torch.ops.xformers, name)
    except (RuntimeError, AttributeError):
        return no_such_operator

try:
    from .. import _C_flashattention  # type: ignore[attr-defined]

    has_flashattention = True
except ImportError:
    has_flashattention = False


class AttentionMask:
    """Base class for custom masks that can be applied \
        in :attr:`xformers.ops.memory_efficient_attention`.

    When using an :attr:`xformers.ops.AttentionMask`
    instead of a :attr:`torch.Tensor`, the mask matrix does
    not need to be materialized, and can be
    hardcoded into some kernels for better performance.

    See also :attr:`xformers.ops.LowerTriangularMask`
    """

    def to_tensor(self) -> torch.Tensor:
        """Materializes the mask tensor

        Returns:
            torch.Tensor
        """
        raise NotImplementedError()


class LowerTriangularMask(AttentionMask):
    """A lower triangular mask that can be used for causal attention"""

    def __init__(self, *tensor_args, **tensor_kwargs) -> None:
        """Creates a Lower triangular mask.
        It is not requires to specify any parameter, as they are only \
            used when calling :attr:`LowerTriangularMask.to_tensor`

        The mask will not be materialized by default, and hence does not use \
            any additional memory, but acts as an option for the MHA kernel.
        """
        self._tensor: Optional[torch.Tensor] = None
        self._tensor_kwargs = tensor_kwargs
        self._tensor_args = tensor_args

    def to_tensor(self) -> torch.Tensor:
        """Materializes the mask tensor

        Returns:
            torch.Tensor
        """
        if self._tensor is None:
            # Work around for "triu_tril_cuda_template" not implemented for 'BFloat16'
            dtype = self._tensor_kwargs.pop("dtype", torch.float)
            create_as = dtype if dtype is not torch.bfloat16 else torch.float32
            self._tensor = torch.full(  # type: ignore
                *self._tensor_args,
                **self._tensor_kwargs,
                dtype=create_as,
                fill_value=float("-inf"),
            )
            self._tensor = torch.triu(self._tensor, diagonal=1).to(dtype)  # type: ignore
        return self._tensor


class AttentionOpBase(torch.autograd.Function):
    """Base class for any attention operator in xFormers

    See:

    - :attr:`xformers.ops.MemoryEfficientAttentionOp`

    - :attr:`xformers.ops.MemoryEfficientAttentionCutlassOp`

    - :attr:`xformers.ops.MemoryEfficientAttentionFlashAttentionOp`

    - :attr:`xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp`
    """

    FORWARD_OPERATOR: Any
    FORWARD_ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 3e-4,
        torch.half: 4e-3,
        torch.bfloat16: 2e-2,
    }
    FORWARD_ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 2e-5,
        torch.half: 4e-4,
        torch.bfloat16: 5e-3,
    }
    SUPPORTED_DEVICES: Set[str]
    SUPPORTED_DTYPES: Set[torch.dtype]
    SUPPORTED_MAX_K: float
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None)}
    SUPPORTS_DROPOUT: bool
    SUPPORTS_CUSTOM_SCALE: bool = False
    SUPPORTS_DIFFERENT_VALUE_EMBED: bool = False
    NAME: str

    _TEST_BATCH_SIZES: List[int] = [1, 300]
    _TEST_K: List[int] = [32, 128]

    @classmethod
    def info(cls):
        if cls.FORWARD_OPERATOR.__name__ == "no_such_operator":
            return "not built"
        return "available"

    @classmethod
    def forward_no_grad(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]],
        p: float,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p, scale):
        raise NotImplementedError()

    @classmethod
    def backward(cls, ctx, grad):
        raise NotImplementedError()

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        device_type = d.device if isinstance(d.device, str) else d.device.type
        if device_type not in cls.SUPPORTED_DEVICES:
            return False
        if d.dtype not in cls.SUPPORTED_DTYPES:
            return False
        if not cls.SUPPORTS_DIFFERENT_VALUE_EMBED and d.k != d.kv:
            return False
        if max(d.k, d.kv) > cls.SUPPORTED_MAX_K:
            return False
        if d.attn_bias_type not in cls.SUPPORTED_ATTN_BIAS_TYPES:
            return False
        if d.has_dropout and not cls.SUPPORTS_DROPOUT:
            return False
        if d.has_custom_scale and not cls.SUPPORTS_CUSTOM_SCALE:
            return False
        # bfloat16 is only supported on A100+
        # ... although the kernels can still run and give the
        # correct result
        if d.dtype is torch.bfloat16 and (
            not device_type.startswith("cuda")
            or torch.cuda.get_device_capability(d.device)[0] < 8
        ):
            return False
        return True


AttentionOp = Type[AttentionOpBase]


class MemoryEfficientAttentionOp(AttentionOpBase):
    """An operator optimized for very small values of K (``K <= 32``) \
        and f32 pre-Ampere as it does not use TensorCores.
    Only supports contiguous inputs in BMK format, so an extra reshape \
        or contiguous call might be done.

    :Deprecated:

        This operator is deprecated and should not be used in new code
    """

    FORWARD_OPERATOR = get_xformers_operator("efficient_attention")
    SUPPORTED_DEVICES = {"cuda", "cpu"}
    SUPPORTED_DTYPES = {torch.float}
    SUPPORTED_MAX_K: float = 32
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), torch.Tensor}
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = False
    NAME = "small_k"

    # as this kernel is a bit slow, this should make tests run faster
    _TEST_BATCH_SIZES = [1, 3]
    _TEST_K = [2, 3, 8, 16, 32]

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if not super(MemoryEfficientAttentionOp, cls).supports(d):
            return False
        buffer_size = 8
        for pack in [1, 2, 4]:
            if (d.k % pack) == 0 and (d.k // pack) <= buffer_size:
                return True
        return False

    @classmethod
    def bmhk2bmk_contiguous(cls, tensor) -> torch.Tensor:
        return (
            tensor.permute((0, 2, 1, 3))
            .contiguous()
            .view([tensor.shape[0] * tensor.shape[2], tensor.shape[1], tensor.shape[3]])
            .contiguous()
        )

    @classmethod
    def bmk2bmhk(cls, tensor, num_heads: int) -> torch.Tensor:
        return tensor.reshape(
            [-1, num_heads, tensor.shape[1], tensor.shape[2]]
        ).permute((0, 2, 1, 3))

    @classmethod
    def forward_no_grad(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]],
        p: float,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        if scale is not None:
            raise NotImplementedError("Unsupport custom scale")
        num_heads = query.shape[2]
        query = cls.bmhk2bmk_contiguous(query)
        key = cls.bmhk2bmk_contiguous(key)
        value = cls.bmhk2bmk_contiguous(value)
        output = cls._forward_no_grad_bmk(query, key, value, attn_bias=attn_bias, p=p)
        return cls.bmk2bmhk(output, num_heads)

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p, scale):
        if scale is not None:
            raise NotImplementedError("Unsupport custom scale")
        num_heads = query.shape[2]
        query = cls.bmhk2bmk_contiguous(query)
        key = cls.bmhk2bmk_contiguous(key)
        value = cls.bmhk2bmk_contiguous(value)
        output = cls._forward_bmk(ctx, query, key, value, attn_bias=attn_bias, p=p)
        return cls.bmk2bmhk(output, num_heads)

    @classmethod
    def backward(cls, ctx, grad):
        num_heads = grad.shape[2]
        grad = cls.bmhk2bmk_contiguous(grad)
        gq, gk, gv, _, _ = cls._backward_bmk(ctx, grad)
        gq = cls.bmk2bmhk(gq, num_heads)
        gk = cls.bmk2bmhk(gk, num_heads)
        gv = cls.bmk2bmhk(gv, num_heads)
        return gq, gk, gv, None, None, None

    @classmethod
    def _forward_no_grad_bmk(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]],
        p: float,
    ) -> torch.Tensor:
        return cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            compute_logsumexp=False,
            attn_bias=attn_bias,
            p=p,
        )[0]

    @classmethod
    def _forward_bmk(cls, ctx, query, key, value, attn_bias, p):
        out, lse, rng_seed, rng_offset = cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            compute_logsumexp=True,
            attn_bias=attn_bias,
            p=p,
        )
        ctx.save_for_backward(query, key, value, lse, attn_bias, out)
        ctx.p = p
        ctx.rng_seed = rng_seed
        ctx.rng_offset = rng_offset
        return out

    @staticmethod
    def _backward_bmk(ctx, grad):
        query, key, value, lse, attn_bias, out = ctx.saved_tensors
        p = ctx.p
        rng_seed = ctx.rng_seed
        rng_offset = ctx.rng_offset
        grad_q, grad_k, grad_v = torch.ops.xformers.efficient_attention_backward(
            grad, query, key, value, lse, out, attn_bias, p, rng_seed, rng_offset
        )
        return grad_q, grad_k, grad_v, None, None


class MemoryEfficientAttentionCutlassOp(AttentionOpBase):
    """xFormers' MHA kernel based on CUTLASS.
    Supports a large number of settings (including without TensorCores, f32 ...)
    and GPUs as old as P100 (Sm60)
    """

    FORWARD_OPERATOR = get_xformers_operator("efficient_attention_forward_cutlass")
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_DTYPES = {torch.float, torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = math.inf
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), LowerTriangularMask}
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = True
    NAME = "cutlass"

    _TEST_K: List[int] = [
        32,  # 64x64 kernel
        128,  # 64x128 kernel
        256,  # 64x128 with accumulation in gmem
    ]

    @classmethod
    def forward_no_grad(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]],
        p: float,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        if attn_bias is not None and not isinstance(attn_bias, LowerTriangularMask):
            raise NotImplementedError("Unsupported attn_bias type")
        return cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=-1,
            compute_logsumexp=False,
            causal=isinstance(attn_bias, LowerTriangularMask),
            scale=scale,
        )[0]

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p, scale):
        if attn_bias is not None and not isinstance(attn_bias, LowerTriangularMask):
            raise NotImplementedError("Unsupported attn_bias type")
        causal = isinstance(attn_bias, LowerTriangularMask)
        out, lse = cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=-1,
            compute_logsumexp=True,
            causal=causal,
            scale=scale,
        )
        ctx.save_for_backward(query, key, value, lse, out)
        ctx.p = p
        ctx.causal = causal
        ctx.scale = scale
        return out

    @classmethod
    def uses_tensorcores(cls, d: "AttentionOpDispatch", is_half: bool) -> bool:
        sm_major = torch.cuda.get_device_capability(d.device)[0]
        if sm_major >= 8:
            return True
        if sm_major >= 7:
            return is_half
        return False

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if not super(MemoryEfficientAttentionCutlassOp, cls).supports(d):
            return False
        cap = torch.cuda.get_device_capability(d.device)
        sm = cap[0] * 10 + cap[1]
        bits_per_scalar = {torch.float: 32, torch.half: 16, torch.bfloat16: 16}[d.dtype]
        uses_tensorcores = cls.uses_tensorcores(d, bits_per_scalar == 16)
        matmul_alignment_mn = 1
        if sm >= 80:
            matmul_alignment_mn = 4
        if uses_tensorcores:
            matmul_alignment_mn = max(matmul_alignment_mn, 128 // bits_per_scalar)
        if (d.k % matmul_alignment_mn != 0) or (d.kv % matmul_alignment_mn != 0):
            return False
        # Sm86 does not have enough shared-memory
        # See https://github.com/facebookresearch/xformers/issues/517
        if (
            d.requires_grad
            and sm >= 80
            and sm != 80
            and d.dtype is torch.float
            and max(d.kv, d.k) > 64
        ):
            return False
        return True

    @classmethod
    def backward(cls, ctx, grad):
        query, key, value, lse, out = ctx.saved_tensors

        dtype = query.dtype
        (
            grad_q,
            grad_k,
            grad_v,
        ) = torch.ops.xformers.efficient_attention_backward_cutlass(
            grad.to(dtype),
            query,
            key,
            value,
            lse,
            out.to(dtype),
            causal=ctx.causal,
            scale=ctx.scale,
        )
        return grad_q, grad_k, grad_v, None, None, None


class MemoryEfficientAttentionFlashAttentionOp(AttentionOpBase):
    """Operator that computes memory-efficient attention using \
        `Flash-Attention <https://github.com/HazyResearch/flash-attention>`_ \
        implementation.


    This is a wrapper to make FlashAttention compatible with xformers's API
    Most of this code was taken from:
    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_interface.py
    """

    FORWARD_OPERATOR = None
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_DTYPES = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 128
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), LowerTriangularMask}
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    NAME = "flshatt"

    @classmethod
    def info(cls):
        if not has_flashattention:
            return "not built"
        return "available - requires GPU with compute capability 7.5+"

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if not has_flashattention:
            return False
        if not super(MemoryEfficientAttentionFlashAttentionOp, cls).supports(d):
            return False
        # We know `d.device` is cuda now
        # d=128 is only supported on A100 for bw
        device_capability = torch.cuda.get_device_capability(d.device)
        is_sm80 = device_capability[0] == 8 and device_capability[1] == 0
        if d.k not in [16, 32, 64, 128]:
            return False
        if d.requires_grad and d.k == 128 and not is_sm80:
            return False
        return device_capability >= (7, 5)

    @classmethod
    def forward_no_grad(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]],
        p: float,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        return cls.forward(
            ctx=None,
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=p,
            scale=scale,
        )

    @classmethod
    def prepare_inputs(
        cls, ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        batch = query.shape[0]
        seqlen_q = query.shape[1]
        seqlen_k = key.shape[1]
        num_heads = query.shape[2]
        head_dim_q = query.shape[3]
        head_dim_v = value.shape[3]
        ctx.max_seqlen_q = seqlen_q
        ctx.max_seqlen_k = seqlen_k

        cu_seqlens_k = torch.arange(
            0,
            (batch + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=query.device,
        )
        if seqlen_q == seqlen_k:
            cu_seqlens_q = cu_seqlens_k
        else:
            cu_seqlens_q = torch.arange(
                0,
                (batch + 1) * seqlen_q,
                step=seqlen_q,
                dtype=torch.int32,
                device=query.device,
            )

        # Initially we have `query.shape = [batch, seqlen, head_dim_q]`
        # We want format `[batch * seqlen, num_heads, head_dim_q]`
        ctx.query_api_input_shape = query.shape
        ctx.key_api_input_shape = key.shape
        ctx.value_api_input_shape = value.shape
        query = query.reshape([batch * seqlen_q, num_heads, head_dim_q])
        key = key.reshape([batch * seqlen_k, num_heads, head_dim_q])
        value = value.reshape([batch * seqlen_k, num_heads, head_dim_v])
        return query, key, value, cu_seqlens_k, cu_seqlens_q

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p, scale):
        if attn_bias is not None and not isinstance(attn_bias, LowerTriangularMask):
            raise NotImplementedError("Unsupported attn_bias type")
        causal = isinstance(attn_bias, LowerTriangularMask)
        return_softmax = False
        ctx_flash = ctx if ctx is not None else SimpleNamespace()
        query, key, value, cu_seqlens_k, cu_seqlens_q = cls.prepare_inputs(
            ctx_flash, query, key, value
        )

        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if p > 0 else None
        softmax_scale = query.shape[-1] ** (-0.5) if scale is None else scale
        out, softmax_lse, S_dmask = cls._flash_attn_forward(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx_flash.max_seqlen_q,
            ctx_flash.max_seqlen_k,
            p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax,
        )
        if ctx is not None:
            ctx.save_for_backward(
                query,
                key,
                value,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                rng_state,
            )
            ctx.dropout_p = p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.kernel_output_shape = out.shape
        return out

    @classmethod
    def backward(cls, ctx, grad):
        return cls._backward(ctx, grad, ctx.saved_tensors)

    @classmethod
    def _backward(cls, ctx, grad, saved_tensors):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            rng_state,
        ) = saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        # Create dq,dk,dv
        # If Q/K/V come from a single QKV tensor, let's put the gradient in the
        # right strides, so we can avoid a `cat`
        if (
            q.shape[0] == k.shape[0]
            and q.shape[2] == v.shape[2]
            and q.storage().data_ptr() == k.storage().data_ptr()
            and q.storage().data_ptr() == v.storage().data_ptr()
        ):
            # Create one big contiguous chunk
            # This is because q, k and v usually come from a single
            # output of a linear layer that is chunked.
            # Creating the gradients with the right layout saves us
            # a `torch.cat` call in the backward pass
            chunk = torch.empty(
                (q.shape[0], 3, q.shape[1], q.shape[2]), dtype=q.dtype, device=q.device
            )
            dq = chunk.select(1, 0)
            dk = chunk.select(1, 1)
            dv = chunk.select(1, 2)
        else:
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        assert grad.dtype in cls.SUPPORTED_DTYPES
        cls._flash_attn_backward(
            grad.reshape(ctx.kernel_output_shape),
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        dq = dq.reshape(ctx.query_api_input_shape)
        dk = dk.reshape(ctx.key_api_input_shape)
        dv = dv.reshape(ctx.value_api_input_shape)
        return dq, dk, dv, None, None, None

    @staticmethod
    def _flash_attn_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
    ):
        out, softmax_lse, *rest = _C_flashattention.fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            return_softmax,
            None,
        )
        S_dmask = rest[0] if return_softmax else None
        return out, softmax_lse, S_dmask

    @staticmethod
    def _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
    ):
        softmax_d = _C_flashattention.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            None,
        )
        return dq, dk, dv, softmax_d


class MemoryEfficientAttentionCutlassFwdFlashBwOp(MemoryEfficientAttentionCutlassOp):
    """An operator that uses :attr:`xformers.ops.MemoryEfficientAttentionCutlassOp` for the forward pass \
        and :attr:`xformers.ops.MemoryEfficientAttentionFlashAttentionOp` for the backward.
    """

    FW_OP = MemoryEfficientAttentionCutlassOp
    BW_OP = MemoryEfficientAttentionFlashAttentionOp
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTED_DTYPES = BW_OP.SUPPORTED_DTYPES.intersection(FW_OP.SUPPORTED_DTYPES)

    NAME = "fctls_bflsh"

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if d.requires_grad and not cls.BW_OP.supports(d):
            return False
        return cls.FW_OP.supports(replace(d, requires_grad=False))

    @classmethod
    def backward(cls, ctx, grad):
        query, key, value, lse, out = ctx.saved_tensors
        ctx_flash = SimpleNamespace()

        ctx_flash.causal = ctx.causal
        ctx_flash.dropout_p = 0.0
        query, key, value, cu_seqlens_k, cu_seqlens_q = cls.BW_OP.prepare_inputs(
            ctx_flash, query, key, value
        )
        ctx_flash.kernel_output_shape = (query.shape[0], query.shape[1], value.shape[2])
        ctx_flash.softmax_scale = (
            query.shape[-1] ** (-0.5) if ctx.scale is None else ctx.scale
        )
        rng_state = None

        out = out.reshape(ctx_flash.kernel_output_shape)
        grad = grad.reshape(ctx_flash.kernel_output_shape)
        return cls.BW_OP._backward(
            ctx_flash,
            grad,
            [query, key, value, out, lse, cu_seqlens_q, cu_seqlens_k, rng_state],
        )


@dataclass
class AttentionOpDispatch:
    """Dispatcher to automatically select
    the best operator to run memory-efficient attention.
    """

    dtype: torch.dtype
    device: Union[torch.device, str]
    k: int
    has_dropout: bool
    attn_bias_type: Any
    kv_len: int
    q_len: int
    kv: int = -1
    batch_size: int = -1
    num_heads: int = 1
    has_custom_scale: bool = False
    requires_grad: bool = True

    def __post_init__(self):
        if self.kv == -1:
            self.kv = self.k

    def _is_cutlass_fwd_faster_than_flash(self) -> bool:
        # Very small batch sizes - if batch size specified
        if self.batch_size > 0:
            threads_flash = self.batch_size * self.num_heads
            threads_cutlass = threads_flash * (self.q_len // 64)
            if threads_flash < 60 and (threads_cutlass // 2) >= threads_flash:
                return True
        # Large values of K
        return max(self.k, self.kv) == 128

    @property
    def op(self) -> AttentionOp:
        """Computes the best operator

        Raises:
            NotImplementedError: if not operator was found

        Returns:
            AttentionOp: The best operator for the configuration
        """
        priority_list_ops: List[AttentionOp] = [
            MemoryEfficientAttentionFlashAttentionOp,
            MemoryEfficientAttentionCutlassOp,
            MemoryEfficientAttentionOp,
        ]
        if self.requires_grad and self._is_cutlass_fwd_faster_than_flash():
            priority_list_ops.insert(0, MemoryEfficientAttentionCutlassFwdFlashBwOp)
        for op in priority_list_ops:
            if op.supports(self):
                return op
        raise NotImplementedError(f"No operator found for this attention: {self}")

    @classmethod
    def from_arguments(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]] = None,
        p: float = 0.0,
        scale: Optional[float] = None,
    ) -> "AttentionOpDispatch":
        """Creates an :attr:`xformers.ops.AttentionOpDispatch` from :attr:`xformers.ops.memory_efficient_attention`'s
        arguments

        Args:
            query (torch.Tensor)
            key (torch.Tensor)
            value (torch.Tensor)
            attn_bias (Optional[Union[torch.Tensor, xformers.ops.AttentionMask]], optional): Defaults to None.
            p (float, optional): Defaults to 0.0.
            scale (float, optional): Custom scale. Default to None (use q.shape[-1]**-0.5).

        Returns:
            AttentionOpDispatch
        """
        B, H = query.shape[0], 1
        if query.ndim == 4:
            H = query.shape[2]
        return AttentionOpDispatch(
            dtype=query.dtype,
            device=query.device,
            k=query.shape[-1],
            kv=value.shape[-1],
            has_dropout=p > 0.0,
            has_custom_scale=scale is not None,
            attn_bias_type=type(attn_bias),
            kv_len=value.shape[1],
            q_len=query.shape[1],
            batch_size=B,
            num_heads=H,
            requires_grad=any(x.requires_grad for x in [query, key, value]),
        )


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionMask]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[AttentionOp] = None,
) -> torch.Tensor:
    """Implements the memory-efficient attention mechanism following
    `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_.

    :Inputs shape:

    - Input tensors must be in format ``[B, M, H, K]``, where B is the batch size, M \
        the sequence length, H the number of heads, and K the embeding size per head

    - If inputs have dimension 3, it is assumed that the dimensions are ``[B, M, K]`` and ``H=1``

    - Inputs can be non-contiguous - we only require the last dimension's stride to be 1


    :Equivalent pytorch code:

    .. code-block:: python

        scale = 1 / query.shape[-1] ** 0.5
        query = query * scale
        attn = query @ key.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(-1)
        attn = F.dropout(attn, p)
        return attn @ value

    :Examples:

    .. code-block:: python

        import xformers.ops as xops

        # Compute regular attention
        y = xops.memory_efficient_attention(q, k, v)

        # With a dropout of 0.2
        y = xops.memory_efficient_attention(q, k, v, p=0.2)

        # Causal attention
        y = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=xops.LowerTriangularMask()
        )

    :Supported hardware:

        NVIDIA GPUs with compute capability above 6.0 (P100+), datatype ``f16``, ``bf16`` and ``f32``.

    Raises:
        NotImplementedError: if there is no operator available to compute the MHA

    :parameter query: Tensor of shape ``[B, Mq, H, K]``
    :parameter key: Tensor of shape ``[B, Mkv, H, K]``
    :parameter value: Tensor of shape ``[B, Mkv, H, Kv]``
    :parameter attn_bias: Bias to apply to the attention matrix - defaults to no masking. \
        For causal attention, use :attr:`xformers.ops.LowerTriangularMask`. \
        This can also be a :attr:`torch.Tensor` for an arbitrary mask.
    :parameter p: Dropout probability. Disabled if set to ``0.0``
    :parameter scale: The scale to query_state weights. If set to ``None``, the default \
        scale (q.shape[-1]**-0.5) will be used.
    :parameter op: The operator to use - see :attr:`xformers.ops.AttentionOpBase`. \
        If set to ``None`` (recommended), xFormers \
        will dispatch to the best available operator, depending on the inputs \
        and options.
    :return: multi-head attention Tensor with shape ``[B, Mq, H, Kv]``
    """

    if query.ndim not in [3, 4]:
        raise ValueError(
            f"Invalid shape for query: {query.shape}. "
            "Expected shape [batch, seqlen, num_heads, K], or [batch, seqlen, K]."
        )
    output_shape = tuple(query.shape[:-1]) + (value.shape[-1],)
    # Convert from legacy format
    if query.ndim == 3:
        query = query.unsqueeze(2)
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    if op is None:
        op = AttentionOpDispatch.from_arguments(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=p,
            scale=scale,
        ).op

    # fast-path that doesn't require computing the logsumexp for backward computation
    if all(x.requires_grad is False for x in [query, key, value]):
        return op.forward_no_grad(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=p,
            scale=scale,
        ).reshape(output_shape)
    return op.apply(query, key, value, attn_bias, p, scale).reshape(output_shape)
