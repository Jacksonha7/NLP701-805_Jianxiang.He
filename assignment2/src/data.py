import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORER = ROOT / "scorer" / "scorer-subtask-1.py"
NO_FRAME = {"-", "No framing"}


def read_labels(path):
    return [line.strip() for line in open(path) if line.strip()]


def read_doc(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def load_paragraphs(data_dir):
    data_dir = Path(data_dir)
    rows, docs, n_mismatch = [], {}, 0
    for line in open(data_dir / "subtask-1-annotations-with-paragraphs.txt", encoding="utf-8"):
        if not line.strip():
            continue
        cols = line.rstrip("\n").split("\t")
        doc, start, end, text = cols[0].strip(), int(cols[1]), int(cols[2]), cols[-1].strip()
        labels = [x.strip() for x in cols[3:-1] if x.strip() and x.strip() not in NO_FRAME]
        if doc not in docs:
            docs[doc] = read_doc(data_dir / "raw-documents" / doc)
        if docs[doc][start:end].strip() != text:
            n_mismatch += 1
        title = docs[doc].splitlines()[0].strip()
        rows.append(dict(doc=doc, start=start, end=end, labels=labels, text=text, title=title))
    print(f"{data_dir}: {len(rows)} paragraphs, {len(docs)} docs, "
          f"{sum(not r['labels'] for r in rows)} without frame, {n_mismatch} offset mismatches")
    return rows


def load_from_offsets(input_file, docs_dir):
    rows, docs = [], {}
    for line in open(input_file, encoding="utf-8"):
        cols = line.strip().split("\t")
        if len(cols) < 3:
            continue
        doc, start, end = cols[0], int(cols[1]), int(cols[2])
        if doc not in docs:
            docs[doc] = read_doc(Path(docs_dir) / doc)
        title = docs[doc].splitlines()[0].strip()
        rows.append(dict(doc=doc, start=start, end=end, labels=[],
                         text=docs[doc][start:end].strip(), title=title))
    print(f"{input_file}: {len(rows)} paragraphs, {len(docs)} docs")
    return rows


def write_tsv(rows, path, label_key="labels"):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write("\t".join([r["doc"], str(r["start"]), str(r["end"])] + list(r[label_key])) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_scorer(gold, pred, labels):
    out = subprocess.run([sys.executable, str(SCORER), "-g", str(gold), "-p", str(pred),
                          "-f", str(labels), "-o"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stdout + out.stderr)
    micro, macro = out.stdout.strip().splitlines()[-1].split("\t")
    return float(micro), float(macro)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/EN")
    ap.add_argument("--out", default="data/")
    ap.add_argument("--labels", default="data/labels.txt")
    ap.add_argument("--extra_dir", default=None, help="SemEval-2023 EN data, only for E4")
    args = ap.parse_args()
    out = Path(args.out)

    rows = load_paragraphs(args.data_dir)
    write_jsonl(rows, out / "paragraphs.jsonl")
    write_tsv(rows, out / "gold_clean.tsv")

    print("clean gold vs itself:", run_scorer(out / "gold_clean.tsv", out / "gold_clean.tsv", args.labels))
    raw_gold = Path(args.data_dir) / "subtask-1-annotations.txt"
    print("raw gold vs perfect pred:", run_scorer(raw_gold, out / "gold_clean.tsv", args.labels))

    if args.extra_dir:
        write_jsonl(load_paragraphs(args.extra_dir), out / "extra_semeval23.jsonl")
