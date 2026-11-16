"""Split integrity (identity-disjoint, balanced, valid) and idempotency tests."""
import filecmp
import os

import pytest
import yaml

from src.data import splits as S

# person -> n_images for a synthetic LFW tree
FAKE = {f"P{i:03d}": (1 if i % 3 == 0 else 2 + i % 4) for i in range(120)}
TEST_MATCH = [("P001", 1, 2), ("P002", 1, 3), ("P004", 2, 3)]
TEST_MISMATCH = [("P005", 1, "P006", 1), ("P007", 2, "P009", 1), ("P010", 1, "P011", 1)]


@pytest.fixture
def fake_lfw(tmp_path):
    root = tmp_path / "lfw"
    for person, n in FAKE.items():
        (root / person).mkdir(parents=True)
        for k in range(1, n + 1):
            (root / person / f"{person}_{k:04d}.jpg").write_bytes(b"")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "matchpairsDevTest.csv").write_text(
        "name,imagenum1,imagenum2\n" + "".join(f"{a},{b},{c}\n" for a, b, c in TEST_MATCH))
    (meta / "mismatchpairsDevTest.csv").write_text(
        "name,imagenum1,name,imagenum2\n" + "".join(f"{a},{b},{c},{d}\n" for a, b, c, d in TEST_MISMATCH))
    return str(root), str(meta)


def _build(fake_lfw, seed=42, n=40):
    root, meta = fake_lfw
    return S.build_splits(root, meta, val_ratio=0.25, n_val_pairs=n, seed=seed)


def test_identities_are_disjoint_across_train_val_test(fake_lfw):
    sp = _build(fake_lfw)
    tr, va, te = set(sp["train_people"]), set(sp["val_people"]), set(sp["test_people"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    # every pair only references people of its own split
    assert S.pair_identities(sp["val_pairs"]) <= va
    assert S.pair_identities(sp["test_pairs"]) == te
    # nobody is lost: all non-test people land in train or val
    assert tr | va == set(FAKE) - te


def test_val_pairs_are_balanced_unique_and_valid(fake_lfw):
    sp = _build(fake_lfw, n=40)
    vp = sp["val_pairs"]
    assert len(vp) == 40 and sum(p[4] for p in vp) == 20
    assert len(set(vp)) == len(vp), "duplicate pairs"
    for n1, k1, n2, k2, label in vp:
        assert 1 <= k1 <= FAKE[n1] and 1 <= k2 <= FAKE[n2]
        if label == 1:
            assert n1 == n2 and k1 != k2
        else:
            assert n1 != n2


def test_test_pairs_are_the_official_list_unchanged(fake_lfw):
    sp = _build(fake_lfw)
    expected = [(a, b, a, c, 1) for a, b, c in TEST_MATCH] + [(a, b, c, d, 0) for a, b, c, d in TEST_MISMATCH]
    assert sp["test_pairs"] == expected


def test_split_is_byte_identical_across_runs(fake_lfw, tmp_path):
    S.write_splits(str(tmp_path / "run1"), _build(fake_lfw))
    S.write_splits(str(tmp_path / "run2"), _build(fake_lfw))
    names = ["train_people.txt", "val_people.txt", "val_pairs.csv", "test_pairs.csv"]
    match, mismatch, errors = filecmp.cmpfiles(tmp_path / "run1", tmp_path / "run2", names, shallow=False)
    assert match == names, (mismatch, errors)


def test_seed_actually_controls_the_split(fake_lfw):
    assert _build(fake_lfw, seed=1)["val_people"] != _build(fake_lfw, seed=2)["val_people"]


def test_pairs_roundtrip_through_csv(fake_lfw, tmp_path):
    vp = _build(fake_lfw)["val_pairs"]
    path = str(tmp_path / "p.csv")
    S.write_pairs(path, vp)
    assert S.read_pairs(path) == vp


def test_missing_test_identity_fails_loudly(fake_lfw):
    root, meta = fake_lfw
    with open(os.path.join(meta, "matchpairsDevTest.csv"), "a") as f:
        f.write("Not_In_Dataset,1,2\n")
    with pytest.raises(FileNotFoundError):
        S.build_splits(root, meta, val_ratio=0.25, n_val_pairs=40, seed=42)


def test_too_few_matched_pairs_fails_loudly():
    with pytest.raises(ValueError):
        S.make_balanced_pairs({"A": 2, "B": 1}, n_pairs=4, seed=0)  # only 1 matched pair exists


# ---- integration against the real dataset (skipped if the data is absent) ----
with open("config.yaml") as _f:
    CFG = yaml.safe_load(_f)
REAL = os.path.isdir(CFG["dataset_path"]) and os.path.isdir(CFG["splits_dir"])


@pytest.mark.skipif(not REAL, reason="LFW images or committed splits not present")
def test_committed_split_files_match_regeneration(tmp_path):
    """If someone edits the sampling code, the committed val set would silently
    diverge from what the code produces. This catches that."""
    sp = S.build_splits(CFG["dataset_path"], CFG["lfw_meta_dir"], CFG["val_ratio"], CFG["n_val_pairs"], CFG["seed"])
    S.write_splits(str(tmp_path), sp)
    names = ["train_people.txt", "val_people.txt", "val_pairs.csv", "test_pairs.csv"]
    match, mismatch, errors = filecmp.cmpfiles(CFG["splits_dir"], tmp_path, names, shallow=False)
    assert match == names, (mismatch, errors)
