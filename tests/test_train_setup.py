"""Regression tests for the optimizer/scheduler setup in train.py.

Colab crashed at the unfreeze epoch with
  TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
Locally (torch 2.5.1) the same argument only emitted a deprecation warning, so
these tests turn warnings into errors to fail here the way newer PyTorch fails.
"""
import warnings

import torch.nn as nn

import train
from src.models.siamese_main import AttentionalSiameseNetwork


def test_build_optimizer_uses_no_deprecated_or_removed_arguments():
    model = nn.Linear(2, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # deprecated today == removed on a newer torch
        optimizer, scheduler = train.build_optimizer(model)
    scheduler.step(1.0)  # the call train.py makes each epoch
    assert optimizer.param_groups[0]["lr"] == train.LEARNING_RATE


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
