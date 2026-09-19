"""
qat_int4.py  —  High-Performance INT4 (W4A16) Group-Wise Quantization-Aware Training
=====================================================================================
Custom PyTorch module providing:
  1. Group-Wise Symmetric 4-Bit Fake-Quantization with Straight-Through Estimator (STE).
  2. FakeQuantLinearW4A16 layer replacing standard nn.Linear.
  3. Bit-packing (two 4-bit weights per uint8 byte) for ultra-compact 55MB disk storage.
  4. Automatic QAT conversion helper for ModernBertModel.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. Straight-Through Estimator (STE) Autograd Function
# ──────────────────────────────────────────────────────────────────────────────

class FakeQuantW4A16Function(torch.autograd.Function):
    """
    Symmetric Group-Wise 4-bit Weight Fake-Quantization:
      - Signed 4-bit range: [-8, 7]
      - Straight-Through Estimator (STE) for backpropagation
      - Clipped gradients outside range [-8.5, 7.5]
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, group_size: int = 64) -> torch.Tensor:
        out_features, in_features = weight.shape
        num_groups = in_features // group_size

        # Reshape to [out_features * num_groups, group_size]
        w_groups = weight.view(-1, group_size)

        # Dynamic per-group absolute maximum scale aligned to weight dtype
        max_val = torch.clamp(w_groups.abs().amax(dim=-1, keepdim=True), min=1e-7)
        scales = (max_val / 7.0).to(weight.dtype)

        # Quantize to 4-bit integer grid [-8, 7]
        q = torch.clamp(torch.round(w_groups / scales), -8.0, 7.0)

        # Save for backward pass
        ctx.save_for_backward(w_groups, scales)

        # Dequantize back to float for computation
        w_dequant = (q * scales).view(out_features, in_features)
        return w_dequant

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        w_groups, scales = ctx.saved_tensors
        # Normalized weight position relative to quantization bins
        norm_w = w_groups / scales
        # STE with gradient saturation mask
        grad_mask = (norm_w >= -8.5) & (norm_w <= 7.5)
        grad_weight = grad_output.view_as(w_groups) * grad_mask.float()
        return grad_weight.view_as(grad_output), None


def fake_quant_w4a16(weight: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Applies group-wise 4-bit fake quantization using STE."""
    return FakeQuantW4A16Function.apply(weight, group_size)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Fake-Quantized Linear Layer (W4A16)
# ──────────────────────────────────────────────────────────────────────────────

class FakeQuantLinearW4A16(nn.Module):
    """
    Linear layer that simulates 4-bit group-wise weight quantization during training.
    Operates seamlessly in FP16/BF16/FP32 autocast while training weights to snap
    into INT4 boundaries.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 64,
    ):
        super().__init__()
        assert in_features % group_size == 0, (
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.qat_enabled = True

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 64) -> "FakeQuantLinearW4A16":
        """Converts an existing nn.Linear into a FakeQuantLinearW4A16, copying weights."""
        has_bias = linear.bias is not None
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            group_size=group_size,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if has_bias:
                layer.bias.copy_(linear.bias)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qat_enabled:
            w_effective = fake_quant_w4a16(self.weight, self.group_size)
        else:
            w_effective = self.weight
        return F.linear(x, w_effective.type_as(x), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}, qat_enabled={self.qat_enabled}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. 4-Bit Packing & Unpacking Utilities (Disk & Memory Serialization)
# ──────────────────────────────────────────────────────────────────────────────

def pack_weights_int4(weight: torch.Tensor, group_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs a float weight tensor into true 4-bit storage:
      - 2 signed 4-bit weights packed per uint8 byte.
      - Returns: (packed_weights [uint8], scales [float16])
    """
    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    w_groups = weight.detach().view(-1, group_size)

    # Calculate per-group scales in float16 for storage
    max_val = torch.clamp(w_groups.abs().amax(dim=-1, keepdim=True), min=1e-7)
    scales = (max_val / 7.0).view(out_features, num_groups).to(torch.float16)

    # Quantize to signed integer [-8, 7] using matching scale values
    scales_expanded = scales.view(-1, 1).to(weight.dtype).expand(-1, group_size)
    q = torch.clamp(torch.round(w_groups / scales_expanded), -8.0, 7.0).to(torch.int8)

    # Two's complement 4-bit representation in lower 4 bits (0x0F)
    q_flat = q.view(out_features, in_features)
    low_bits = q_flat[:, 0::2] & 0x0F
    high_bits = (q_flat[:, 1::2] & 0x0F) << 4

    packed = (low_bits | high_bits).to(torch.uint8)
    return packed, scales


def unpack_weights_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 64,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Unpacks a 4-bit packed tensor back to high precision on the fly:
      - packed: [out_features, in_features // 2] (uint8)
      - scales: [out_features, in_features // group_size] (float16)
    """
    out_features, half_in = packed.shape
    in_features = half_in * 2
    num_groups = in_features // group_size

    # Extract 4-bit nibbles
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)

    # Sign extension from 4-bit to 8-bit (if bit 3 is 1, subtract 16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    # Interleave back to original sequence
    q = torch.empty((out_features, in_features), dtype=torch.int8, device=packed.device)
    q[:, 0::2] = low
    q[:, 1::2] = high

    # Multiply by group scales
    w_groups = q.view(-1, group_size).to(dtype)
    s_expanded = scales.view(-1, 1).expand(-1, group_size).to(dtype)
    w_dequant = (w_groups * s_expanded).view(out_features, in_features)
    return w_dequant


# ──────────────────────────────────────────────────────────────────────────────
# 4. Model-Wide QAT Insertion for ModernBert
# ──────────────────────────────────────────────────────────────────────────────

def apply_qat_to_modernbert(model: nn.Module, group_size: int = 64) -> nn.Module:
    """
    Traverses ModernBertModel and replaces all compute-intensive linear projection layers
    with FakeQuantLinearW4A16:
      - Attention: Wq, Wk, Wv, Wo
      - SwiGLU MLP: w1, w2, w3
    Leaves RoPE frequencies, RMSNorm, and Softmax in high precision.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Only quantize if in_features is divisible by group_size (768 and 2048 both match)
            if module.in_features % group_size == 0:
                qat_layer = FakeQuantLinearW4A16.from_linear(module, group_size=group_size)
                setattr(model, name, qat_layer)
        else:
            # Recursive descent into TransformerBlock, ModernAttention, ModernMLP
            apply_qat_to_modernbert(module, group_size=group_size)
    return model


def set_qat_enabled(model: nn.Module, enabled: bool = True):
    """Recursively toggles QAT fake-quantization mode across all layers."""
    for m in model.modules():
        if isinstance(m, FakeQuantLinearW4A16):
            m.qat_enabled = enabled
