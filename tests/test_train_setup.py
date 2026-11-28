"""Regression tests for the optimizer/scheduler setup in train.py.

Colab crashed at the unfreeze epoch with
  TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
Locally (torch 2.5.1) the same argument only emitted a deprecation warning, so
these tests turn warnings into errors to fail here the way newer PyTorch fails.
"""
import warnings

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import train
from src.models.siamese_main import AttentionalSiameseNetwork


@pytest.fixture(scope="module")
def frozen_model():
    torch.manual_seed(0)
    return AttentionalSiameseNetwork(freeze_backbone=True)


def test_build_optimizer_uses_no_deprecated_or_removed_arguments(frozen_model):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # deprecated today == removed on a newer torch
        optimizer, scheduler = train.build_optimizer(frozen_model)
    scheduler.step(1.0)  # the call train.py makes each epoch
    head, backbone = optimizer.param_groups
    assert head["lr"] == train.HEAD_LEARNING_RATE
    assert backbone["lr"] == train.LEARNING_RATE


def test_head_learning_rate_is_large_enough_to_move_the_scale():
    """Adam moves a weight ~lr per step. At the backbone lr the cosine scale
    could change by only ~0.1 in log space over the whole run."""
    steps = 143 * train.NUM_EPOCHS  # 9,108 train images / batch 64, measured
    assert train.HEAD_LEARNING_RATE * steps >= 1.0
    assert train.HEAD_LEARNING_RATE > train.LEARNING_RATE


def test_head_and_backbone_groups_partition_all_parameters(frozen_model):
    head = {id(p) for p in frozen_model.head_parameters()}
    backbone = {id(p) for p in frozen_model.backbone_parameters()}
    everything = {id(p) for p in frozen_model.parameters()}
    assert head.isdisjoint(backbone)
    assert head | backbone == everything


def test_frozen_backbone_batchnorm_statistics_do_not_change():
    """requires_grad=False does not stop BatchNorm running-stat updates;
    model.train() used to overwrite the ImageNet statistics in epochs 0-4."""
    torch.manual_seed(0)
    model = AttentionalSiameseNetwork(freeze_backbone=True).train()
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    before = [bn.running_mean.clone() for bn in bns]
    with torch.no_grad():
        model(torch.randn(2, 3, 64, 64) * 5 + 3, torch.randn(2, 3, 64, 64))
    assert model.training  # the model itself is still in train mode (dropout active)
    assert all(torch.equal(b, bn.running_mean) for b, bn in zip(before, bns))

    model.unfreeze_backbone()  # after unfreeze, BatchNorm adapts again
    assert all(bn.training for bn in bns)
    with torch.no_grad():
        model(torch.randn(2, 3, 64, 64) * 5 + 3, torch.randn(2, 3, 64, 64))
    assert not torch.equal(before[0], bns[0].running_mean)


def test_eval_mode_still_applies_to_backbones_after_unfreeze():
    model = AttentionalSiameseNetwork(freeze_backbone=True)
    model.unfreeze_backbone()
    model.eval()
    assert not any(m.training for m in model.modules())


class _NaNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, a, b):
        return (a.sum(1, keepdim=True) * self.w) * float("nan"), None


class _NullWriter:
    def add_scalar(self, *a, **k):
        pass


def test_training_stops_on_non_finite_loss():
    data = TensorDataset(torch.randn(4, 3), torch.randn(4, 3), torch.ones(4))
    model = _NaNModel()
    with pytest.raises(FloatingPointError, match="epoch 1, batch 0"):
        train.train_one_epoch(model, DataLoader(data, batch_size=2), nn.BCEWithLogitsLoss(),
                              torch.optim.SGD(model.parameters(), lr=0.1), 0, _NullWriter())


def test_unfreeze_path_rebuilds_optimizer_over_all_parameters():
    """Exercises the epoch == UNFREEZE_EPOCH branch, which a 1-epoch smoke run never reaches."""
    model = AttentionalSiameseNetwork(freeze_backbone=True)
    frozen_opt, _ = train.build_optimizer(model)
    n_trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.unfreeze_backbone()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        opt, sched = train.build_optimizer(model)
    sched.step(1.0)

    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_in_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_trainable_frozen < n_all  # backbones really were frozen before
    assert n_trainable == n_all        # and are trainable after
    assert n_in_opt == n_all           # and the new optimizer will update them
