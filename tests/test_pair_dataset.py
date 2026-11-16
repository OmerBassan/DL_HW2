"""Fixed-pair dataset determinism: the eval set must not change between runs."""
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.lfw_dataset import LFWDataset, PairListDataset
from src.data.transforms import get_transforms

PEOPLE = {"Ann": 3, "Bob": 2, "Cid": 1, "Dee": 2}
PAIRS = [("Ann", 1, "Ann", 3, 1), ("Ann", 2, "Bob", 1, 0), ("Bob", 1, "Bob", 2, 1),
         ("Cid", 1, "Dee", 2, 0), ("Dee", 1, "Dee", 2, 1), ("Bob", 2, "Cid", 1, 0)]


@pytest.fixture
def root(tmp_path):
    rng = np.random.default_rng(0)
    for person, n in PEOPLE.items():
        (tmp_path / person).mkdir()
        for k in range(1, n + 1):
            arr = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_path / person / f"{person}_{k:04d}.jpg")
    return str(tmp_path)


def _ds(root):
    return PairListDataset(root, PAIRS, transform=get_transforms(split="test", input_size=(32, 32)))


def test_item_idx_is_always_pair_idx(root):
    a, b = _ds(root), _ds(root)
    assert len(a) == len(PAIRS)
    for i in range(len(PAIRS)):
        x1, x2, y = a[i]
        z1, z2, w = b[i]
        assert torch.equal(x1, z1) and torch.equal(x2, z2)
        assert y.item() == PAIRS[i][4]


def test_identity_of_a_pair_matches_its_files(root):
    """Pair 0 is Ann_1 vs Ann_3: the tensors must differ from Ann_1 vs Ann_1."""
    ds = _ds(root)
    x1, x2, _ = ds[0]
    assert not torch.equal(x1, x2)


def test_multiworker_loader_yields_same_batches_as_direct_indexing(root):
    ds = _ds(root)
    direct_labels = torch.stack([ds[i][2] for i in range(len(ds))])
    direct_img1 = torch.stack([ds[i][0] for i in range(len(ds))])
    for _ in range(2):  # two separate "runs"
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
        img1 = torch.cat([b[0] for b in loader])
        labels = torch.cat([b[2] for b in loader])
        assert torch.equal(labels, direct_labels)
        assert torch.equal(img1, direct_img1)


def test_missing_image_raises_instead_of_substituting(root):
    with pytest.raises(FileNotFoundError):
        PairListDataset(root, PAIRS + [("Ann", 9, "Bob", 1, 0)])


def test_training_dataset_respects_explicit_people(root):
    ds = LFWDataset(root, transform=None, people=["Ann", "Bob"])
    assert ds.people == ["Ann", "Bob"]
    assert set(ds.person_images) == {"Ann", "Bob"}
