"""
Minimal VSI keyframe search (visual + subtitle fusion).

Condensed from VSI_keyframe_search.py + TStar/interface_searcher.py.
Pipeline for one video, sampled at 1 fps:
    1. Subtitle branch: question-subtitle cosine similarity -> per-second text scores
    2. Visual branch:   sample 16 frames -> 4x4 grid -> object detector -> per-frame confidences
    3. Fusion:          score = (1 - w) * minmax(visual) + w * minmax(text)
    4. Update the sampling distribution from the fused scores; repeat until all
       target objects are verified or the search budget runs out
    5. Return the top-k seconds by score
"""

import json
import os
import re

import cv2
import numpy as np

STOPWORDS = {"a", "an", "the", "is", "are", "in", "on", "at", "of", "and", "or", "with"}


# ---------------------------------------------------------------------------
# Subtitle branch
# ---------------------------------------------------------------------------

def _to_seconds(t):
    """Accepts a number, 'HH:MM:SS.sss' or 'HH:MM:SS,mmm' (SRT)."""
    if isinstance(t, (int, float)):
        return float(t)
    h, m, s = str(t).replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_subtitles(path):
    """Returns [(start_sec, end_sec, text)]. LongVideoBench uses .json, VideoMME uses .srt."""
    if not path or not os.path.exists(path):
        return []

    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        return [
            (_to_seconds(it["start"]), _to_seconds(it["end"]), it.get("line", it.get("text", "")))
            for it in items
            if it.get("start") is not None and it.get("end") is not None
        ]

    if path.endswith(".srt"):
        with open(path, encoding="utf-8") as f:
            blocks = re.split(r"\n\s*\n", f.read().strip())
        subs = []
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue
            m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", lines[1])
            if m:
                text = " ".join(re.sub(r"<.*?>", "", l) for l in lines[2:]).strip()
                subs.append((_to_seconds(m.group(1)), _to_seconds(m.group(2)), text))
        return subs

    raise ValueError(f"Unsupported subtitle format: {path}")


class SubtitleScorer:
    """Maps question-subtitle similarity onto a per-second score curve."""

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name, device=device)

    @staticmethod
    def _filter(text):
        return " ".join(w for w in text.lower().split() if w not in STOPWORDS)

    def score(self, question, subtitles, num_seconds, time_window=2.0):
        scores = np.zeros(num_seconds)
        if not subtitles:
            return scores

        q = self.encoder.encode([self._filter(question)], normalize_embeddings=True)
        s = self.encoder.encode([self._filter(t) for _, _, t in subtitles],
                                batch_size=32, normalize_embeddings=True)
        sims = (s @ q.T).ravel()

        # Soft threshold: stretch similarities above 0.5, cap at 1.0
        sims = np.minimum(np.where(sims > 0.5, 0.5 + (sims - 0.5) * 2, sims), 1.0)

        # Spread each relevant subtitle over [start - 2s, end + 2s] with a Gaussian
        for (start, end, _), sim in zip(subtitles, sims):
            if sim <= 0.2:
                continue
            a = max(0, int(start - time_window))
            b = min(num_seconds, int(end + time_window))
            if b <= a:
                continue
            t = np.arange(a, b)
            center, sigma = (a + b) // 2, (b - a) / 4
            scores[a:b] = np.maximum(scores[a:b], sim * np.exp(-((t - center) ** 2) / (2 * sigma ** 2)))
        return scores


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.full(len(x), 0.5)


def fuse_scores(visual, text, text_weight):
    return (1 - text_weight) * _minmax(visual) + text_weight * _minmax(text)


# ---------------------------------------------------------------------------
# Keyframe searcher (visual branch + iterative sampling)
# ---------------------------------------------------------------------------

class KeyframeSearcher:
    """
    `detector` must provide:
        set_vocabulary(names: list[str])
        detect(image: np.ndarray) -> list[(name, confidence, (x1, y1, x2, y2))]
    `text_scores` is a per-second array (see SubtitleScorer.score); None = all zeros.
    """

    CELL_W, CELL_H = 200, 95  # size of one frame inside the detection grid

    def __init__(self, video_path, target_objects, cue_objects, detector,
                 text_scores=None, text_weight=0.3, search_nframes=8, grid_shape=(4, 4),
                 confidence_threshold=0.7, search_budget=1.0, seed=2025):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        self.raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.raw_fps)  # video length in seconds

        self.detector = detector
        self.detector.set_vocabulary(target_objects + cue_objects)
        self.weights = {o: 1.0 for o in target_objects}
        self.weights.update({o: 0.5 for o in cue_objects})
        self.remaining_targets = list(target_objects)

        self.text_scores = np.zeros(self.n) if text_scores is None else np.asarray(text_scores)[:self.n]
        self.text_weight = text_weight
        self.k = search_nframes
        self.grid_shape = grid_shape
        self.threshold = confidence_threshold
        self.budget = min(1000, self.n * search_budget)
        self.rng = np.random.default_rng(seed)

        self.score = np.zeros(self.n)
        self.unvisited = np.ones(self.n)
        self.P = np.ones(self.n) / self.n
        self.first_round = True

    # --- frame IO ---

    def read_frame(self, sec, size):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * self.raw_fps))
        ok, frame = self.cap.read()
        if not ok:
            frame = np.zeros((size[1], size[0], 3), np.uint8)
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size)

    def make_grid(self, frames):
        rows, cols = self.grid_shape
        blank = np.zeros_like(frames[0])
        frames = frames + [blank] * (rows * cols - len(frames))  # pad short videos
        return np.vstack([np.hstack(frames[r * cols:(r + 1) * cols]) for r in range(rows)])

    # --- detection ---

    def detect_cells(self, image, grid_shape):
        """Runs the detector once; returns per-cell max weighted confidence and object names."""
        rows, cols = grid_shape
        cell_h, cell_w = image.shape[0] / rows, image.shape[1] / cols
        conf = np.zeros(rows * cols)
        objects = [[] for _ in range(rows * cols)]
        for name, c, (x1, y1, x2, y2) in self.detector.detect(image):
            gx = min(int((x1 + x2) / 2 // cell_w), cols - 1)
            gy = min(int((y1 + y2) / 2 // cell_h), rows - 1)
            idx = gy * cols + gx
            conf[idx] = max(conf[idx], c * self.weights.get(name, 0.5))
            objects[idx].append(name)
        return conf, objects

    # --- sampling ---

    def sample(self, num):
        num = min(num, self.n)
        if self.first_round:  # uniform first pass
            self.first_round = False
            return np.arange(0, self.n, self.n // num)[:num]

        p = (self.P + num / self.n) * self.unvisited
        p = p * (p >= np.percentile(p, 75))  # keep top 25% unvisited seconds
        if np.count_nonzero(p) < num:
            p = self.P + num / self.n
        return self.rng.choice(self.n, size=num, replace=False, p=p / p.sum())

    # --- distribution update ---

    def update(self, secs, confs):
        for s, c in zip(secs, confs):
            self.unvisited[s] = 0
            self.score[s] = c

        # Propagate top-25% confidences to +-5 s neighbours with 1/(d+1) decay
        top = np.percentile(confs, 75)
        for s, c in zip(secs, confs):
            if c < top:
                continue
            for off in range(-5, 6):
                j = s + off
                if 0 <= j < self.n:
                    self.score[j] = max(self.score[j], self.score[s] / (abs(off) + 1))

        self.score = fuse_scores(self.score, self.text_scores, self.text_weight)

        # Interpolate visited scores over all seconds -> sigmoid -> normalise
        visited = np.flatnonzero(self.unvisited == 0)
        interp = np.interp(np.arange(self.n), visited, self.score[visited])
        p = 1 / (1 + np.exp(-np.maximum(1 / self.n, interp)))
        self.P = p / p.sum()

    def verify(self, sec, objects):
        """Re-detects a single frame; drops a target once it is confirmed above threshold."""
        for target in list(self.remaining_targets):
            if target not in objects:
                continue
            frame = self.read_frame(sec, (self.CELL_W * 3, self.CELL_H * 3))
            conf, objs = self.detect_cells(frame, (1, 1))
            self.score[sec] = conf[0]
            if target in objs[0] and conf[0] > self.threshold:
                self.remaining_targets.remove(target)
                return

    # --- main loop ---

    def search(self):
        rows, cols = self.grid_shape
        iterations = 0
        while self.remaining_targets and self.budget > 0:
            iterations += 1
            secs = [int(s) for s in self.sample(rows * cols)]
            self.budget -= rows * cols

            frames = [self.read_frame(s, (self.CELL_W, self.CELL_H)) for s in secs]
            conf, objects = self.detect_cells(self.make_grid(frames), self.grid_shape)
            confs = conf[:len(secs)]

            self.update(secs, confs)
            for s, objs in zip(secs, objects):
                self.verify(s, objs)

        top_k = np.argsort(self.score)[-self.k:][::-1]
        self.cap.release()
        return {
            "keyframe_timestamps": [float(s) for s in top_k],
            "num_iterations": iterations,
            "score_distribution": self.score.tolist(),
        }
