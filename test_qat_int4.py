"""
test_qat_int4.py  —  Unit Verification for INT4 W4A16 QAT Engine
==================================================================
Tests:
  1. Forward pass numerical quantization.
  2. Backward gradient flow via Straight-Through Estimator (STE).
  3. Bit-packing (two 4-bit weights per uint8 byte) and unpacking fidelity.
  4. Automatic QAT layer replacement on ModernBertModel.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
from qat_int4 import (
    FakeQuantLinearW4A16,
    fake_quant_w4a16,
    pack_weights_int4,
    unpack_weights_int4,
    apply_qat_to_modernbert,
)
from bert import ModernBertModel, BertConfig


def test_forward_and_backward():
    print("Testing FakeQuantLinearW4A16 forward and backward STE...")
    layer = FakeQuantLinearW4A16(768, 768, group_size=64)
    x = torch.randn(2, 16, 768, requires_grad=True)

    out = layer(x)
    assert out.shape == (2, 16, 768), f"Unexpected out shape: {out.shape}"

    loss = out.sum()
    loss.backward()

    assert layer.weight.grad is not None, "Weight gradient is None!"
    assert not torch.all(layer.weight.grad == 0), "Weight gradient is all zeros!"
    assert x.grad is not None, "Input gradient is None!"
    print("  [OK] Forward and backward gradient flow verified successfully.")


def test_bit_packing():
    print("Testing 4-bit packing and unpacking fidelity...")
    w = torch.randn(768, 768, dtype=torch.float16)
    packed, scales = pack_weights_int4(w, group_size=64)

    # Check shapes and dtypes
    assert packed.dtype == torch.uint8, f"Packed dtype should be uint8, got {packed.dtype}"
    assert packed.shape == (768, 384), f"Packed shape should be (768, 384), got {packed.shape}"
    assert scales.shape == (768, 12), f"Scales shape should be (768, 12), got {scales.shape}"

    # Verify unpacked tensor matches fake_quant output exactly
    w_unpacked = unpack_weights_int4(packed, scales, group_size=64, dtype=torch.float16)
    w_fake = fake_quant_w4a16(w, group_size=64)

    max_diff = (w_unpacked - w_fake).abs().max().item()
    assert max_diff == 0.0, f"Mismatch between unpacked and fake-quantized weights! Max diff: {max_diff}"
    print(f"  [OK] 4-bit packing verified! Exact bit-level match (diff: {max_diff}), storage halved by 4x.")


def test_model_qat_conversion():
    print("Testing apply_qat_to_modernbert on 12-layer architecture...")
    config = BertConfig(n_layer=2, n_embd=768, intermediate_size=2048)
    model = ModernBertModel(config)

    # Count original linear layers
    orig_linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(orig_linears) > 0, "No linear layers found in model!"

    apply_qat_to_modernbert(model, group_size=64)

    qat_layers = [m for m in model.modules() if isinstance(m, FakeQuantLinearW4A16)]
    # All internal transformer projections (14 layers for 2 blocks) are quantized, while lm_head stays in FP16
    expected_qat = len(orig_linears) - 1
    assert len(qat_layers) == expected_qat, (
        f"Expected {expected_qat} QAT layers, found {len(qat_layers)}"
    )
    assert isinstance(model.lm_head, nn.Linear) and not isinstance(model.lm_head, FakeQuantLinearW4A16), (
        "lm_head should be preserved as high-precision nn.Linear!"
    )

    # Test forward pass through QAT model
    input_ids = torch.randint(0, 1000, (2, 32))
    logits, _ = model(input_ids)
    assert logits.shape == (2, 32, config.vocab_size), f"Unexpected logits shape: {logits.shape}"
    print("  [OK] Full model QAT conversion verified successfully.")


if __name__ == "__main__":
    test_forward_and_backward()
    test_bit_packing()
    test_model_qat_conversion()
    print("\nALL QAT INT4 UNIT TESTS PASSED! [SUCCESS]\n")
