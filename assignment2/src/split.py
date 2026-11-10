import json

import numpy as np

from data import read_jsonl, read_labels, write_tsv

labels = read_labels("data/labels.txt")
rows = read_jsonl("data/paragraphs.jsonl")
docs = sorted({r["doc"] for r in rows})
row_doc = np.array([r["doc"] for r in rows])

y = np.zeros((len(rows), len(labels) + 1))
for i, r in enumerate(rows):
    for lab in r["labels"]:
        y[i, labels.index(lab)] = 1
    if not r["labels"]:
        y[i, -1] = 1


def make_split(seed):
    perm = np.random.RandomState(seed).permutation(docs)
    return {"train": sorted(perm[30:]), "val": sorted(perm[15:30]), "test": sorted(perm[:15])}


def imbalance(split):
    p = [y[np.isin(row_doc, split[part])].mean(0) for part in ("train", "val", "test")]
    return np.abs(p[0] - p[1]).sum() + np.abs(p[0] - p[2]).sum() + np.abs(p[1] - p[2]).sum()


scores = [imbalance(make_split(s)) for s in range(1000)]
seed = int(np.argmin(scores))
split = make_split(seed)
print(f"best seed {seed}, imbalance {scores[seed]:.3f} (median {np.median(scores):.3f})")

info = {"seed": seed, "docs": split, "stats": {}}
for part in ("train", "val", "test"):
    part_rows = [r for r in rows if r["doc"] in split[part]]
    write_tsv(part_rows, f"data/gold_{part}.tsv")
    info["stats"][part] = {"docs": len(split[part]), "paragraphs": len(part_rows),
                           "none": sum(not r["labels"] for r in part_rows)}
    print(part, info["stats"][part])
json.dump(info, open("data/splits.json", "w"), indent=2)
