"""Re-run VSI keyframe search on T2A/T2E/T2O with a configurable subtitle branch.
"""
import argparse
import json
import logging
import os
import sys
import time

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
        subs = self.q.subtitles(fix_offset=False)
        if self.scorer is None or not subs:
            return np.zeros(self.total_frame_num)
        query = self.q.question
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


def load_grounding(path: str):
    """GPT-4o grounding per (video_id, question) from one released VSI run."""
    out = {}
    for r in json.load(open(path)):
        g = r.get("grounding_objects")
        if isinstance(g, dict) and g.get("target_objects"):
            out[(r["video_id"], r["question"])] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="/nfs-stor/jianxiang.he/nlp805/fused")
    ap.add_argument("--grounding_json", default="/nfs-stor/jianxiang.he/nlp805/frame_search/"
                    "MVSLS_0_w0.3_rel0.0_LongVideoBench_gpt4_cot_all.json")
    ap.add_argument("--config_path", default=os.path.join(
        os.path.dirname(HERE), "third_party/YOLO-World/configs/pretrain/"
        "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"))
    ap.add_argument("--checkpoint_path", default="/l/users/jianxiang.he/YOLO-World/weights/s_stage1-d1c1d7d8.pth")
    ap.add_argument("--scorer", default="mpnet", choices=["none", *DENSE_MODELS, "bm25"])
    ap.add_argument("--text_weight", type=float, default=0.3)
    ap.add_argument("--theta", type=float, default=0.5, help="soft-threshold knee (VSI default 0.5)")
    ap.add_argument("--search_nframes", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")

    grounding = load_grounding(args.grounding_json)
    questions = [q for q in load_questions() if (q.video_id, q.question) in grounding]
    log.info("%d questions have grounding objects", len(questions))
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

    results = []
    for i, q in enumerate(questions):
        config = VideoSearchConfig(
            video_path=os.path.join(DATA_ROOT, "videos", f"{q.video_id}.mp4"),
            subtitle_path="", question=q.question, options="",
            text_weight=text_weight, search_nframes=args.search_nframes,
            grid_rows=4, grid_cols=4, device=args.device,
            output_dir=os.path.join(args.out_dir, "tmp"))
        t0 = time.time()
        np.random.seed(args.seed + i)
        fw = SubtitleVariantFramework(config, yolo, text_calc, q, args, scorer)
        res = fw.process_video(grounding[(q.video_id, q.question)])
        results.append({"qid": q.qid, "video_id": q.video_id, "category": q.category,
                        "duration_group": q.duration_group,
                        "keyframe_timestamps": [float(t) for t in res.keyframe_timestamps],
                        "num_iterations": res.num_iterations, "seconds": time.time() - t0})
        log.info("[%d/%d] %s %.1fs", i + 1, len(questions), q.qid, results[-1]["seconds"])
    with open(out_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
