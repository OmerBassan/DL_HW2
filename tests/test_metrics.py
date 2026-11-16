import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.metrics import accuracy_at, best_threshold, predict_probs, roc_auc, wilson_ci


# ------------------------------------------------------------------ Wilson CI
@pytest.mark.parametrize(
    "k, n, lo, hi",
    [  # reference values: statsmodels proportion_confint(method="wilson"), 0.14.2
        (850, 1000, 0.8265313415731996, 0.8707899274869118),
        (0, 50, 0.0, 0.07134759913335874),
        (50, 50, 0.9286524008666412, 1.0),
    ],
)
def test_wilson_matches_statsmodels(k, n, lo, hi):
    got = wilson_ci(k, n)
    assert got == pytest.approx((lo, hi), abs=1e-9)


def test_wilson_stays_in_unit_interval_and_contains_point_estimate():
    for k in (0, 1, 499, 999, 1000):
        lo, hi = wilson_ci(k, 1000)
        assert 0.0 <= lo <= k / 1000 <= hi <= 1.0


def test_wilson_rejects_invalid_counts():
    with pytest.raises(ValueError):
        wilson_ci(5, 0)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


# --------------------------------------------------------- AUC and threshold
def test_auc_perfect_random_and_inverted():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert roc_auc(y, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) == 1.0
    assert roc_auc(y, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]) == 0.0
    assert roc_auc(y, [0.5] * 6) == 0.5


def test_auc_single_class_fails_loudly():
    with pytest.raises(ValueError):
        roc_auc([1, 1, 1], [0.2, 0.5, 0.9])


def test_best_threshold_separable_lands_between_classes():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.10, 0.20, 0.30, 0.70, 0.80, 0.90]
    t, acc = best_threshold(y, p)
    assert acc == 1.0
    assert 0.30 < t < 0.70
    assert accuracy_at(y, p, t) == 1.0


def test_best_threshold_finds_shifted_operating_point():
    """A model whose scores are all squeezed above 0.5: 0.5 gives 50%,
    the tuned threshold recovers 100%. This is why we tune on val."""
    y = [0, 0, 1, 1]
    p = [0.60, 0.65, 0.80, 0.90]
    assert accuracy_at(y, p, 0.5) == 0.5
    t, acc = best_threshold(y, p)
    assert acc == 1.0 and 0.65 < t < 0.80


def test_best_threshold_is_deterministic_under_ties():
    y = [0, 1, 0, 1]
    p = [0.4, 0.4, 0.6, 0.6]  # every threshold gives 50%
    assert best_threshold(y, p) == best_threshold(list(y), list(p))


# ------------------------------------------------------------- predict_probs
class _TinySiamese(nn.Module):
    """Same output contract as AttentionalSiameseNetwork: ([B,1] logits, attn)."""

    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(4, 1)

    def forward(self, a, b):
        return self.fc(self.drop(a - b)), None


def _loader():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(10, 4, generator=g), torch.randn(10, 4, generator=g)
    y = torch.tensor([0, 1] * 5, dtype=torch.float32)
    return DataLoader(TensorDataset(a, b, y), batch_size=3, shuffle=False)


def test_predict_probs_uses_eval_mode_and_is_repeatable():
    torch.manual_seed(0)
    model = _TinySiamese().train()  # deliberately left in train mode
    p1, y1, _ = predict_probs(model, _loader(), "cpu")
    p2, y2, _ = predict_probs(model, _loader(), "cpu")
    assert not model.training
    np.testing.assert_array_equal(p1, p2)  # dropout active would make these differ
    np.testing.assert_array_equal(y1, [0, 1] * 5)
    assert p1.shape == (10,) and ((p1 >= 0) & (p1 <= 1)).all()


def test_predict_probs_mean_loss_is_sample_weighted():
    torch.manual_seed(0)
    model = _TinySiamese()
    crit = nn.BCEWithLogitsLoss()
    p, y, loss = predict_probs(model, _loader(), "cpu", criterion=crit)
    expected = nn.functional.binary_cross_entropy(torch.tensor(p), torch.tensor(y, dtype=torch.float64)).item()
    assert loss == pytest.approx(expected, rel=1e-5)  # last batch has 1 item, not 3
