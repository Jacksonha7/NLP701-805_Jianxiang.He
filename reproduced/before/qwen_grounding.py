"""Grounding objects from a local Qwen3-VL-8B for questions the released GPT-4o
runs do not cover (all of the 15 s group, a few others).

Uses T*'s own grounding prompt (with relations, as in the released runs), 8
uniformly sampled frames and T*'s parser, so only the model differs. Output
records have the fields `load_grounding` in run_fused_search.py reads.

Run in an env with transformers >= 4.57 (Qwen3-VL), e.g. ~/.conda/envs/starVLA:
    python nlp805/qwen_grounding.py --out /nfs-stor/jianxiang.he/nlp805/grounding/qwen3vl8b.json
"""
import argparse
import json
import logging
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from data import DATA_ROOT, load_questions  # noqa: E402
from TStar.interface_llm import TStarUniversalGrounder  # noqa: E402
from TStar.utilites import load_video_frames  # noqa: E402

log = logging.getLogger("qwen_grounding")
MODEL = "/nfs-stor/jianxiang.he/ego-oracle-gap/models/Qwen3-VL-8B-Instruct"


def video_path(video_id: str) -> str:
    p = os.path.join(DATA_ROOT, "videos", f"{video_id}.mp4")
    if not os.path.exists(p):
        # TikTok ids look like "@user-<digits>", but the file is stored as "<digits>.mp4".
        p = os.path.join(DATA_ROOT, "videos", f"{video_id.rsplit('-', 1)[-1]}.mp4")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=448 * 448)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    # Prompt builder and parser only; the constructor would demand an API key.
    tstar = TStarUniversalGrounder.__new__(TStarUniversalGrounder)

    done = {}
    if os.path.exists(args.out):
        done = {r["qid"]: r for r in json.load(open(args.out))}
    results = list(done.values())
    questions = load_questions()
    for i, q in enumerate(questions):
        if q.qid in done:
            continue
        options = "\n".join(f"{chr(65 + j)}) {c}" for j, c in enumerate(q.candidates))
        prompt = tstar._build_grounding_prompt(question=q.question, options=options, with_relations=True)
        frames = load_video_frames(video_path=video_path(q.video_id), num_frames=args.num_frames)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                 *({"type": "image"} for _ in frames)]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=frames or None, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        target, cue, rel = tstar._parse_grounding_response(response, expect_relations=True)
        results.append({"qid": q.qid, "video_id": q.video_id, "question": q.question, "n_frames": len(frames),
                        "response": response,
                        "grounding_objects": {"target_objects": target, "cue_objects": cue,
                                              "relations": [list(r) for r in rel]}})
        log.info("[%d/%d] %s target=%s cue=%s", i + 1, len(questions), q.qid, target, cue)
        if len(results) % 20 == 0:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
