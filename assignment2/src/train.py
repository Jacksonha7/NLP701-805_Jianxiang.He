import argparse
import json
import pickle

from data import read_jsonl
from model import LABELS, ROOT, FramingModel, load_config, load_split, official_f1, tune_thresholds


def train(cfg, train_rows, val_rows):
    model = FramingModel(cfg).fit(train_rows)
    gold = [r["labels"] for r in val_rows]
    model.thresholds = tune_thresholds(model.predict_proba(val_rows), gold,
                                       cfg.get("threshold", "fixed"), cfg.get("t", 0.5))
    return model, official_f1(gold, model.predict(val_rows))


ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
args = ap.parse_args()

cfg = load_config(args.exp)
train_rows, val_rows = load_split("train"), load_split("val")
if cfg.get("extra_data"):
    train_rows += read_jsonl(ROOT / "data" / "extra_semeval23.jsonl")

if cfg.get("C_grid"):
    val_micro = {C: train({**cfg, "C": C}, train_rows, val_rows)[1][0] for C in cfg["C_grid"]}
    print("val micro per C:", val_micro)
    cfg["C"] = max(cfg["C_grid"], key=lambda C: (val_micro[C], -abs(C - 1)))

model, (micro, macro) = train(cfg, train_rows, val_rows)
print(f"{args.exp}: val micro={micro:.4f} macro={macro:.4f} C={cfg.get('C')} "
      f"thresholds={model.thresholds.tolist()}")

out = ROOT / "outputs" / args.exp
out.mkdir(parents=True, exist_ok=True)
pickle.dump(model, open(out / "model.pkl", "wb"))
log = {"config": cfg, "thresholds": dict(zip(LABELS, model.thresholds.tolist())),
       "val_micro": micro, "val_macro": macro}
json.dump(log, open(out / "train_log.json", "w"), indent=2)
