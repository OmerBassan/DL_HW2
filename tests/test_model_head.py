"""Regression tests for the cosine scoring head (AttentionalSiameseNetwork.score).

The old head summed 25,088 squared feature differences into one distance
(10,778-62,837 at init) and fed it to Linear(1,1): logits were ~1e4, sigmoid
returned exactly 1.0 for every pair (eval AUC 0.5), and fp16 overflowed.
"""
import pytest
import torch

from src.models.siamese_main import INIT_BIAS, MAX_SCALE, AttentionalSiameseNetwork


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return AttentionalSiameseNetwork(freeze_backbone=True).eval()


def _maps(seed, b=4):
    g = torch.Generator().manual_seed(seed)
    f1 = torch.rand(b, 512, 7, 7, generator=g)  # non-negative, like post-ReLU features
    f2 = torch.rand(b, 512, 7, 7, generator=g)
    mask = torch.rand(b, 1, 7, 7, generator=g)
    return f1, f2, mask


def test_logit_does_not_grow_with_feature_magnitude(model):
    """The saturation bug: the old score scaled with the square of the features."""
    f1, f2, mask = _maps(0)
    with torch.no_grad():
        base = model.score(f1, f2, mask)
        big = model.score(f1 * 1000, f2 * 1000, mask)
        big_mask = model.score(f1, f2, mask * 50)
    assert torch.allclose(base, big, atol=1e-4)
    assert torch.allclose(base, big_mask, atol=1e-4)


def test_logit_is_bounded_even_at_extreme_scale(model):
    f1, f2, mask = _maps(1)
    with torch.no_grad():
        model.log_scale.fill_(50.0)  # exp(50) would be ~5e21 without the clamp
        try:
            logits = model.score(f1 * 1e6, f2, mask)
        finally:
            model.log_scale.fill_(torch.log(torch.tensor(10.0)).item())
    assert torch.isfinite(logits).all()
    assert logits.abs().max() <= MAX_SCALE * (1 + abs(INIT_BIAS)) + 1e-3


def test_identical_maps_score_higher_than_different_maps(model):
    """Sign is fixed by construction: scale > 0, so more similar -> higher logit.
    The old Linear(1,1) started with a random-sign weight and had to learn it."""
    f1, f2, mask = _maps(2)
    with torch.no_grad():
        same = model.score(f1, f1.clone(), mask)
        diff = model.score(f1, f2, mask)
    assert (same > diff).all()
    assert torch.allclose(same, torch.full_like(same, (1 - INIT_BIAS) * 10.0), atol=1e-4)


def test_all_zero_mask_gives_finite_logit(model):
    f1, f2, _ = _maps(3)
    with torch.no_grad():
        logits = model.score(f1, f2, torch.zeros(4, 1, 7, 7))
    assert torch.isfinite(logits).all()


def test_forward_outputs_are_not_saturated(model):
    """Mirror of the vacuous-determinism trap: outputs must have spread."""
    g = torch.Generator().manual_seed(4)
    a, b = torch.randn(6, 3, 224, 224, generator=g), torch.randn(6, 3, 224, 224, generator=g)
    with torch.inference_mode():
        logits, attn = model(a, b)
    probs = torch.sigmoid(logits)
    assert logits.shape == (6, 1) and attn.shape == (6, 1, 7, 7)
    assert torch.isfinite(logits).all()
    assert probs.unique().numel() > 1
    assert ((probs > 1e-4) & (probs < 1 - 1e-4)).all()


def test_head_parameters_stay_trainable_when_backbone_frozen(model):
    assert model.log_scale.requires_grad and model.bias.requires_grad
    assert model.attn_conv.weight.requires_grad
    assert not next(model.feature_backbone.parameters()).requires_grad
