"""Identity-disjoint train / val / test splits for LFW face verification.

Protocol (decided once, materialised to disk so it can be audited):
  * TEST  = official LFW View 1 DevTest pairs (matchpairsDevTest.csv +
            mismatchpairsDevTest.csv): 500 matched + 500 mismatched pairs.
  * TRAIN / VAL = every person NOT appearing in the test pairs, split BY PERSON
            with a seeded RNG (val_ratio of people go to val).
  * VAL pairs = a fixed, balanced list generated once with a seeded RNG.

Why materialise instead of regenerating in code each run: any later edit to the
sampling code would silently change the val set and invalidate comparisons
between runs. Files on disk are versioned and diffable.

Pair file format (all pair CSVs we write): name1,num1,name2,num2,label
  label 1 = same person, 0 = different. Image path = root/name/name_NNNN.jpg.

Run:  python -m src.data.splits     (writes config['splits_dir'])
"""
import csv
import os
import random
import tempfile

import yaml

PAIR_HEADER = ["name1", "num1", "name2", "num2", "label"]


# --------------------------------------------------------------------------- io
def image_path(root_dir, name, num):
    return os.path.join(root_dir, name, f"{name}_{int(num):04d}.jpg")


def scan_people(root_dir):
    """Return {person: n_images}, sorted by name for determinism."""
    people = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    counts = {p: len([f for f in os.listdir(os.path.join(root_dir, p)) if f.endswith(".jpg")]) for p in people}
    return {p: n for p, n in counts.items() if n > 0}


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [[c for c in row if c != ""] for row in reader if row]


def load_view1_test_pairs(meta_dir):
    """Official LFW View 1 DevTest pairs as (name1, num1, name2, num2, label)."""
    pairs = []
    for name, n1, n2 in _read_rows(os.path.join(meta_dir, "matchpairsDevTest.csv")):
        pairs.append((name, int(n1), name, int(n2), 1))
    for a, n1, b, n2 in _read_rows(os.path.join(meta_dir, "mismatchpairsDevTest.csv")):
        pairs.append((a, int(n1), b, int(n2), 0))
    return pairs


def read_pairs(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == PAIR_HEADER, f"{path}: header {reader.fieldnames} != {PAIR_HEADER}"
        return [(r["name1"], int(r["num1"]), r["name2"], int(r["num2"]), int(r["label"])) for r in reader]


def read_people(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _atomic_write(path, write_fn):
    """Write via temp file + rename so a crash never leaves a half-written split."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            write_fn(f)
        os.replace(tmp, path)
    except BaseException:
        os.remove(tmp)
        raise


def write_pairs(path, pairs):
    def _w(f):
        w = csv.writer(f, lineterminator="\n")
        w.writerow(PAIR_HEADER)
        w.writerows(pairs)
    _atomic_write(path, _w)


def write_people(path, people):
    _atomic_write(path, lambda f: f.write("".join(p + "\n" for p in people)))


# ---------------------------------------------------------------------- logic
def pair_identities(pairs):
    return {p[0] for p in pairs} | {p[2] for p in pairs}


def split_people(people, excluded, val_ratio, seed):
    """Split people (minus `excluded`) by identity into sorted (train, val) lists."""
    assert 0.0 < val_ratio < 1.0, val_ratio
    pool = sorted(set(people) - set(excluded))
    rng = random.Random(seed)
    rng.shuffle(pool)
    n_val = int(round(len(pool) * val_ratio))
    return sorted(pool[n_val:]), sorted(pool[:n_val])


def make_balanced_pairs(counts, n_pairs, seed):
    """Fixed balanced pair list: n_pairs//2 matched + n_pairs//2 mismatched, no duplicates.

    Matched pairs come from people with >= 2 images; mismatched pairs from any
    two distinct people. Persons are sampled uniformly, then images uniformly.
    """
    assert n_pairs % 2 == 0, "n_pairs must be even for a balanced set"
    half = n_pairs // 2
    rng = random.Random(seed)
    people = sorted(counts)
    multi = [p for p in people if counts[p] >= 2]
    max_pos = sum(counts[p] * (counts[p] - 1) // 2 for p in multi)
    if len(people) < 2 or max_pos < half:
        raise ValueError(f"cannot draw {half} unique matched pairs from {len(multi)} multi-image people")

    pos, neg = set(), set()
    while len(pos) < half:
        p = rng.choice(multi)
        a, b = sorted(rng.sample(range(1, counts[p] + 1), 2))
        pos.add((p, a, p, b, 1))
    while len(neg) < half:
        p, q = sorted(rng.sample(people, 2))
        neg.add((p, rng.randint(1, counts[p]), q, rng.randint(1, counts[q]), 0))
    # sets are unordered: sort for a deterministic file, then interleave labels
    pos, neg = sorted(pos), sorted(neg)
    return [x for pair in zip(pos, neg) for x in pair]


def build_splits(root_dir, meta_dir, val_ratio, n_val_pairs, seed):
    counts = scan_people(root_dir)
    test_pairs = load_view1_test_pairs(meta_dir)
    test_ids = pair_identities(test_pairs)
    missing = [p for p in test_ids if p not in counts]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test identities missing under {root_dir}, e.g. {missing[:3]}")
    train_people, val_people = split_people(counts, test_ids, val_ratio, seed)
    val_pairs = make_balanced_pairs({p: counts[p] for p in val_people}, n_val_pairs, seed)
    return {
        "train_people": train_people,
        "val_people": val_people,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
        "test_people": sorted(test_ids),
    }


def write_splits(out_dir, splits):
    os.makedirs(out_dir, exist_ok=True)
    write_people(os.path.join(out_dir, "train_people.txt"), splits["train_people"])
    write_people(os.path.join(out_dir, "val_people.txt"), splits["val_people"])
    write_pairs(os.path.join(out_dir, "val_pairs.csv"), splits["val_pairs"])
    write_pairs(os.path.join(out_dir, "test_pairs.csv"), splits["test_pairs"])


def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    splits = build_splits(
        root_dir=cfg["dataset_path"],
        meta_dir=cfg["lfw_meta_dir"],
        val_ratio=cfg["val_ratio"],
        n_val_pairs=cfg["n_val_pairs"],
        seed=cfg["seed"],
    )
    write_splits(cfg["splits_dir"], splits)
    tr, va, te = (set(splits[k]) for k in ("train_people", "val_people", "test_people"))
    vp, tp = splits["val_pairs"], splits["test_pairs"]
    print(f"people  train={len(tr)}  val={len(va)}  test={len(te)}")
    print(f"overlap train&val={len(tr & va)}  train&test={len(tr & te)}  val&test={len(va & te)}")
    print(f"val pairs={len(vp)} (pos={sum(p[4] for p in vp)})  test pairs={len(tp)} (pos={sum(p[4] for p in tp)})")
    print(f"written to {cfg['splits_dir']}")


if __name__ == "__main__":
    main()
