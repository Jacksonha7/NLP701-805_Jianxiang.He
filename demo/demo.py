import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

ENCODERS = {
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),  
    "bge": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
}
STOP = {"a", "an", "the", "is", "are", "in", "on", "at", "of", "and", "or", "with"}


def clean(t):
    return " ".join(w for w in t.lower().split() if w not in STOP)


def line_scores(model, prefix, question, lines):
    q = model.encode(prefix + clean(question), normalize_embeddings=True)
    L = model.encode([clean(t) for t in lines], normalize_embeddings=True, batch_size=64)
    return L @ q


def score_curve(c, s):
    s = np.minimum(np.where(s > 0.5, 0.5 + 2 * (s - 0.5), s), 1.0)
    fps, n = c["native_fps"], c["native_frames"]
    f = np.zeros(n)
    for (st, en, _), v in zip(c["subtitles"], s):
        a, b = max(0, int((st - 2) * fps)), min(n, int((en + 2) * fps))
        if v <= 0.2 or b <= a:
            continue
        x = np.arange(a, b)
        f[a:b] = np.maximum(f[a:b], v * np.exp(-((x - (a + b) // 2) ** 2) / (2 * ((b - a) / 4) ** 2)))
    return np.interp(np.linspace(0, n - 1, c["n_scores"]), np.arange(n), f)


def is_gold(c, st, en):
    return any(min(abs(g - st), abs(g - en)) <= c["tolerance_s"] or st <= g <= en for g in c["gt_secs"])


cases = json.load(open("data/cases.json"))
os.makedirs("results", exist_ok=True)
models = {k: SentenceTransformer(m) for k, (m, _) in ENCODERS.items()}
log = open("results/results.txt", "w")


def out(*a):
    print(*a)
    print(*a, file=log)


for i, c in enumerate(cases, 1):
    lines = [t for _, _, t in c["subtitles"]]
    gold = np.array([is_gold(c, st, en) for st, en, _ in c["subtitles"]])
    out(f"\n===== case {i}: {c['qid']}  ({c['duration'] / 60:.0f} min video, {len(lines)} subtitle lines)")
    out("Q:", c["question"])
    out("answer:", c["options"][c["answer"]], "  at", ", ".join(f"{g:.0f}s" for g in c["gt_secs"]))

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    frames_fig, fax = plt.subplots(2, 9, figsize=(14, 3.6))
    for j, (k, (_, prefix)) in enumerate(ENCODERS.items()):
        s = line_scores(models[k], prefix, c["question"], lines)
        rank = 1 + np.sum(s > s[gold].max())
        frames = c["saved_frames"][k]
        hit = any(abs(t - g) <= c["tolerance_s"] for t in frames for g in c["gt_secs"])
        out(f"\n[{k}]  rank of gold line: {rank}   search result: {'HIT' if hit else 'miss'}")
        for r in np.argsort(-s)[:3]:
            st, _, t = c["subtitles"][r]
            out(f"   {s[r]:.3f}  {st:7.1f}s  {'*' if gold[r] else ' '} {t}")
        out("   8 frames picked (s):", sorted(int(t) for t in frames))
        out("   max diff vs saved scores: %.1e" % np.abs(s - np.array(c["saved_scores"][k])).max())

        # plot 1: subtitle score over time, red = answer moment, x = picked frames
        ax[j].plot(score_curve(c, s), lw=0.8)
        for g in c["gt_secs"]:
            ax[j].axvspan(g - c["tolerance_s"], g + c["tolerance_s"], color="r", alpha=0.3)
        ax[j].plot(frames, [-0.05] * 8, "kx")
        ax[j].set_ylim(-0.1, 1.05)
        ax[j].set_ylabel(k)
        ax[j].set_title(f"{k}: gold line rank {rank}, {'hit' if hit else 'miss'}", fontsize=9, loc="left")

        fax[j, 0].imshow(plt.imread(f"data/frames/{c['qid']}/gt.jpg"))
        fax[j, 0].set_title("ground truth", fontsize=8, color="r")
        for m, t in enumerate(frames):
            fax[j, m + 1].imshow(plt.imread(f"data/frames/{c['qid']}/{k}_{m}.jpg"))
            near = any(abs(t - g) <= c["tolerance_s"] for g in c["gt_secs"])
            fax[j, m + 1].set_title(f"{t:.0f}s", fontsize=8, color="r" if near else "k")
        fax[j, 0].set_ylabel(k)
    ax[1].set_xlabel("time (s)")
    fig.suptitle(f"case {i}: subtitle score over time")
    fig.tight_layout()
    fig.savefig(f"results/case{i}_scores.png", dpi=120)
    for a in fax.flat:
        a.set_xticks([]), a.set_yticks([])
    frames_fig.suptitle(f"case {i}: frames picked by the search (red = near the answer)")
    frames_fig.tight_layout()
    frames_fig.savefig(f"results/case{i}_frames.png", dpi=120)
    plt.close("all")

print("\nsaved to results/")
