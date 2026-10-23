"""Re-run VSI keyframe search on T2A/T2E/T2O with a configurable subtitle branch.
"""
import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from data import DATA_ROOT, load_questions  # noqa: E402
from retrievers import BM25Scorer, DenseScorer, to_unit_range, vsi_filter  # noqa: E402
from VSI_keyframe_search import (  # noqa: E402
    MultimodalTStarFramework, ScoreFusionManager, SubtitleProcessor,
    TextSimilarityCalculator, VideoSearchConfig)

log = logging.getLogger("fused")

DENSE_MODELS = {
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
}


class SubtitleVariantFramework(MultimodalTStarFramework):
    """VSI framework whose subtitle line scores come from a pluggable scorer."""

    def __init__(self, config, yolo_scorer, text_calculator, question, args, scorer):
        # Skip the parent constructor: it reloads the sentence encoder per video.
        self.config = config
        self.grounder = None
        self.yolo_scorer = yolo_scorer
        self.subtitle_processor = SubtitleProcessor()
        self.text_calculator = text_calculator
        self.score_fusion_manager = ScoreFusionManager(config.text_weight)
        self.subtitles = []
        self._initialize_video_info()
        self.results = {}
        self.q, self.args, self.scorer = question, args, scorer

    def _compute_text_similarity_scores(self) -> np.ndarray:
        subs = self.q.subtitles(fix_offset=self.args.fix_offset)
        if self.scorer is None or not subs:
            return np.zeros(self.total_frame_num)
        query = (self.q.quote or self.q.question) if self.args.query == "quote" else self.q.question
        lines = [s.text for s in subs]
        if self.args.scorer == "mpnet":
            sims = self.text_calculator._compute_text_similarities(query, lines)
        elif self.args.scorer == "bm25":
            sims = to_unit_range(self.scorer.score(query, lines))
        else:
            sims = self.scorer.score(query, lines)
        boosted = self.text_calculator._apply_soft_threshold(np.asarray(sims, dtype=float),
                                                             threshold=self.args.theta)
        return self.text_calculator._convert_to_frame_scores(
            [(s.start, s.end) for s in subs], boosted, self.total_frame_num, self.fps)


def adaptive_grid(duration: float, rows: int, cols: int):
    """Largest square grid a clip sampled at 1 fps can fill, capped at rows x cols."""
    n = int(duration)
    if n >= rows * cols:
        return rows, cols
    side = max(1, min(int(math.isqrt(max(n, 1))), rows, cols))
    return side, side


def load_grounding(primary: str, fallbacks):
    """GPT-4o grounding per (video_id, question). The released runs overwrite
    `grounding_objects` with [] when a search fails, so missing entries are
    filled from other released runs, in the order given. Returns
    {key: (grounding, source file name)}."""
    out = {}
    for path in [primary, *fallbacks]:
        if not os.path.exists(path):
            log.warning("grounding file %s not found, skipped", path)
            continue
        for r in json.load(open(path)):
            g = r.get("grounding_objects")
            key = (r["video_id"], r["question"])
            if key not in out and isinstance(g, dict) and g.get("target_objects"):
                out[key] = (g, os.path.basename(path))
    return out


def write_run_config(args, path):
    """Record the exact command, resolved arguments and environment of a run."""
    import hashlib
    import platform
    import subprocess
    repo = os.path.dirname(HERE)
    sources = ["VSI_keyframe_search.py", "TStar/interface_searcher.py", "TStar/interface_yolo.py",
               "nlp805/run_fused_search.py", "nlp805/data.py", "nlp805/retrievers.py"]
    cfg = {
        "command": " ".join([sys.executable, *sys.argv]),
        "args": vars(args),
        "text_encoder": (DENSE_MODELS[args.scorer][0] if args.scorer in DENSE_MODELS else args.scorer),
        "git_commit": subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        # nlp805/ is not tracked by git, so pin the code version by file hash.
        "source_sha256": {p: hashlib.sha256(open(os.path.join(repo, p), "rb").read()).hexdigest()[:16]
                          for p in sources},
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    log.info("command: %s", cfg["command"])
    log.info("config written to %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="/nfs-stor/jianxiang.he/nlp805/fused")
    ap.add_argument("--grounding_json", default="/nfs-stor/jianxiang.he/nlp805/frame_search/"
                    "MVSLS_0_w0.3_rel0.0_LongVideoBench_gpt4_cot_all.json")
    ap.add_argument("--grounding_fallback", nargs="*", default=[
        "/nfs-stor/jianxiang.he/nlp805/frame_search/KFS_UNI_0_F32_rel0.0_PTonly_LongVideoBench_gpt4_cot_vid1.json",
        # Local Qwen3-VL-8B grounding (qwen_grounding.py) for questions no GPT-4o run covers.
        "/nfs-stor/jianxiang.he/nlp805/grounding/qwen3vl8b.json"])
    ap.add_argument("--config_path", default=os.path.join(
        os.path.dirname(HERE), "third_party/YOLO-World/configs/pretrain/"
        "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"))
    ap.add_argument("--checkpoint_path", default="/l/users/jianxiang.he/YOLO-World/weights/s_stage1-d1c1d7d8.pth")
    ap.add_argument("--scorer", default="mpnet", choices=["none", *DENSE_MODELS, "bm25"])
    ap.add_argument("--query", default="full", choices=["full", "quote"])
    ap.add_argument("--fix_offset", action="store_true")
    ap.add_argument("--grid_fix", action="store_true")
    ap.add_argument("--text_weight", type=float, default=0.3)
    ap.add_argument("--theta", type=float, default=0.5, help="soft-threshold knee (VSI default 0.5)")
    ap.add_argument("--groups", type=int, nargs="*", default=None, help="restrict to these duration groups")
    ap.add_argument("--qid_seed", action="store_true",
                    help="seed each question by its qid instead of its list index, so subsets stay paired")
    ap.add_argument("--search_nframes", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these qids")
    ap.add_argument("--tmp_dir", default=None,
                    help="where per-video score histories go (default: <out_dir>/tmp)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    write_run_config(args, os.path.join(args.out_dir, f"{args.name}.config.json"))
    done = {}
    if os.path.exists(out_path):
        # Errored questions are retried on resume.
        done = {r["qid"]: r for r in json.load(open(out_path)) if "error" not in r}

    grounding = load_grounding(args.grounding_json, args.grounding_fallback)
    questions = [q for q in load_questions() if (q.video_id, q.question) in grounding]
    log.info("%d questions have grounding objects", len(questions))
    if args.groups:
        questions = [q for q in questions if q.duration_group in set(args.groups)]
    if args.only:
        questions = [q for q in questions if q.qid in set(args.only)]
    if args.limit:
        questions = questions[: args.limit]

    from TStar.interface_yolo import YoloWorldInterface
    yolo = YoloWorldInterface(config_path=args.config_path, checkpoint_path=args.checkpoint_path,
                              device=args.device)
    text_calc = TextSimilarityCalculator(DENSE_MODELS["mpnet"][0], args.device)
    if args.scorer in ("none", "mpnet"):
        scorer = None if args.scorer == "none" else "mpnet"
    elif args.scorer == "bm25":
        scorer = BM25Scorer()
    else:
        name, prefix = DENSE_MODELS[args.scorer]
        scorer = DenseScorer(name, args.device, query_prefix=prefix)
    text_weight = 0.0 if args.scorer == "none" else args.text_weight

    results = list(done.values())
    for i, q in enumerate(questions):
        if q.qid in done:
            continue
        rows, cols = adaptive_grid(q.duration, 4, 4) if args.grid_fix else (4, 4)
        video_path = os.path.join(DATA_ROOT, "videos", f"{q.video_id}.mp4")
        if not os.path.exists(video_path):
            # TikTok ids look like "@user-<digits>", but the file is stored as "<digits>.mp4".
            video_path = os.path.join(DATA_ROOT, "videos", f"{q.video_id.rsplit('-', 1)[-1]}.mp4")
        config = VideoSearchConfig(
            video_path=video_path,
            subtitle_path="", question=q.question, options="",
            text_weight=text_weight, search_nframes=args.search_nframes,
            grid_rows=rows, grid_cols=cols, device=args.device,
            output_dir=args.tmp_dir or os.path.join(args.out_dir, "tmp"))
        rec = {"qid": q.qid, "video_id": q.video_id, "category": q.category,
               "duration_group": q.duration_group, "grid": [rows, cols],
               "grounding_source": grounding[(q.video_id, q.question)][1]}
        t0 = time.time()
        try:
            np.random.seed(args.seed + (zlib.crc32(q.qid.encode()) % 100003 if args.qid_seed else i))
            fw = SubtitleVariantFramework(config, yolo, text_calc, q, args, scorer)
            res = fw.process_video(grounding[(q.video_id, q.question)][0])
            rec.update(keyframe_timestamps=[float(t) for t in res.keyframe_timestamps],
                       num_iterations=res.num_iterations)
        except Exception as e:  # noqa: BLE001 -- recorded and counted as a miss
            rec.update(keyframe_timestamps=[], error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc(limit=3))
            log.error("%s failed: %s", q.qid, e)
        rec["seconds"] = time.time() - t0
        results.append(rec)
        log.info("[%d/%d] %s %.1fs", i + 1, len(questions), q.qid, rec["seconds"])
        if len(results) % 10 == 0 or i == len(questions) - 1:
            with open(out_path, "w") as f:
                json.dump(results, f)
    with open(out_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
