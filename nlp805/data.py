"""Load the LongVideoBench text-referred perception subsets (T2A, T2E, T2O).
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DATA_ROOT = "/nfs-stor/jianxiang.he/continual-vsi/data/LongVideoBench"
TEXT_CATEGORIES = ("T2A", "T2E", "T2O")
POSITION_FPS = 24.0
# Search accuracy counts a hit within 200 frames of a ground-truth keyframe.
TOLERANCE_S = 200 / POSITION_FPS

QUOTE_RE = re.compile(r"['\"“‘](.{6,}?)['\"”’](?=[\s?,.]|$)")


@dataclass
class Subtitle:
    start: float
    end: float
    text: str


@dataclass
class Question:
    qid: str
    video_id: str
    category: str
    question: str
    question_wo_ref: str
    duration: float
    duration_group: int
    sub_offset: float
    gt_secs: List[float]
    candidates: List[str] = field(default_factory=list)
    subtitles_raw: List[Subtitle] = field(default_factory=list)

    def subtitles(self, fix_offset: bool) -> List[Subtitle]:
        if not fix_offset or not self.sub_offset:
            return self.subtitles_raw
        o = self.sub_offset
        return [Subtitle(s.start - o, s.end - o, s.text) for s in self.subtitles_raw]

    @property
    def quote(self) -> Optional[str]:
        """Longest quoted span in the question, i.e. the subtitle anchor."""
        spans = QUOTE_RE.findall(self.question)
        return max(spans, key=len) if spans else None


def _to_seconds(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    h, m, s = v.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_subtitles(path: str) -> List[Subtitle]:
    """Both LongVideoBench formats: {start,end,line} and {timestamp,text}."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    subs = []
    for it in items:
        if "start" in it:
            b, e, text = _to_seconds(it["start"]), _to_seconds(it["end"]), it["line"]
        else:
            b, e = it["timestamp"]
            b = float(b)
            e = float(e) if e is not None else b
            text = it["text"]
        subs.append(Subtitle(b, e, " ".join(text.split())))
    return subs


def load_questions(root: str = DATA_ROOT,
                   categories: Tuple[str, ...] = TEXT_CATEGORIES) -> List[Question]:
    with open(os.path.join(root, "lvb_val.json")) as f:
        items = json.load(f)
    out = []
    for it in items:
        if it["question_category"] not in categories:
            continue
        q = Question(
            qid=it["id"],
            video_id=it["video_id"],
            category=it["question_category"],
            question=it["question"],
            question_wo_ref=it["question_wo_referring_query"],
            duration=float(it["duration"]),
            duration_group=int(it["duration_group"]),
            sub_offset=float(it["starting_timestamp_for_subtitles"]),
            gt_secs=[p / POSITION_FPS for p in it["position"]],
            candidates=list(it["candidates"]),
        )
        q.subtitles_raw = load_subtitles(os.path.join(root, "subtitles", it["subtitle_path"]))
        out.append(q)
    return out
