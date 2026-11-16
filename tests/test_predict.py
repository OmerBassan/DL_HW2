"""Regression test: predict.py must apply sigmoid to the model's logit.

The model returns logits (BCEWithLogitsLoss). Comparing a raw logit to 0.5
labels every pair with 0 < logit <= 0.5 (probability 0.50-0.62) as DIFFERENT.
"""
import math

import pytest
import torch

from predict import upsample_attention, verdict
from src.models.siamese_main import AttentionalSiameseNetwork


@pytest.mark.parametrize(
    "logit, expected",
    [
        (0.3, "SAME PERSON"),        # sigmoid=0.574; raw-logit compare said DIFFERENT
        (0.5, "SAME PERSON"),        # sigmoid=0.622; raw-logit compare said DIFFERENT
        (-0.3, "DIFFERENT PEOPLE"),
        (0.0, "DIFFERENT PEOPLE"),   # sigmoid=0.5 exactly, strict '>' -> DIFFERENT
        (5.0, "SAME PERSON"),
        (-5.0, "DIFFERENT PEOPLE"),
    ],
)
def test_verdict_label_uses_sigmoid(logit, expected):
    _, result = verdict(logit)
    assert result == expected


@pytest.mark.parametrize("logit", [-5.0, -0.3, 0.0, 0.3, 5.0])
def test_verdict_reports_probability_not_logit(logit):
    prob, _ = verdict(logit)
    assert 0.0 <= prob <= 1.0
    assert prob == pytest.approx(1.0 / (1.0 + math.exp(-logit)))


def test_verdict_respects_custom_threshold():
    # sigmoid(0.3)=0.574: SAME at 0.5, DIFFERENT at a val-tuned 0.6
    assert verdict(0.3, threshold=0.5)[1] == "SAME PERSON"
    assert verdict(0.3, threshold=0.6)[1] == "DIFFERENT PEOPLE"


def test_upsample_attention_accepts_model_output_shape():
    """Regression: predict.py crashed with 'Input and output must have the same
    number of spatial dimensions' because it unsqueezed an already-4D map."""
    # Shape contract comes from the model: attention_mask is [B, 1, 7, 7].
    attn_map = torch.rand(1, 1, 7, 7)
    out = upsample_attention(attn_map, size=(224, 224))
    assert out.shape == (224, 224)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_model_attention_shape_matches_upsample_contract():
    """Pins the producer side of the contract, so a model change that alters
    the attention shape fails here instead of at visualisation time."""
    torch.manual_seed(0)
    model = AttentionalSiameseNetwork().eval()
    with torch.inference_mode():
        logits, attn = model(torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 1)
    assert attn.shape == (1, 1, 7, 7)
