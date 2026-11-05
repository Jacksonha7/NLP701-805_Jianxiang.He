# Textual Semantic Matching for Keyframe Search in Long-Video Understanding
## Introduction for Github repository
### Repository Structure

```text
NLP701-805_Jianxiang.He/
├── nlp805/                         #  Code and experiments for this course project
│   ├── data.py                     # Loads LongVideoBench text-referred subsets (T2A / T2E / T2O) and subtitles
│   ├── retrievers.py               # Subtitle-line scorers: VSI dense encoder, MiniLM / MPNet / BGE, BM25
│   ├── run_fused_search.py         # Re-runs VSI keyframe search with a configurable subtitle branch
│   └── experiments/
│       ├── exp1_encoder/           # Exp 1: subtitle encoder ablation (MiniLM / BGE / BM25 vs. MPNet)
│       │   ├── run.sbatch          # Slurm script that launches all variants
│       │   ├── logs/               # Slurm output, per-run logs, exact commands, pip freeze
│       │   ├── outputs/            # Per-run predictions, configs, and per-video score histories 
│       │   └── results/            # Final comparison table
│       └── exp2_query/             # Exp 2: quoted text vs. full question as the subtitle query
│           ├── run.sbatch
│           ├── logs/
│           ├── outputs/
│           └── results/            # eval_quote_vs_full.txt
├── reproduced/                     # Reproduction of VSI keyframe search
│   ├── before/                     # R1: first reproduction (before the mid-term report)
│   │   ├── qwen_grounding.py       # Object grounding with a local Qwen3-VL-8B
│   │   ├── object_grounding/       # Grounding output (qwen3vl8b.json)
│   │   ├── run_multimodal_search.py# VSI search script, adapted to our server
│   │   ├── result/                 # Search outputs for T* and VSI
│   │   └── matching_results/       # Keyframe matching results + command + log (R1_Tstar, R1_VSI)
│   └── new/                        # R2: full re-run with complete commands, configs and logs
│       ├── run_multimodal_search.py
│       ├── qwen3vl8b.json
│       ├── logs_w0.0/              # text_weight = 0.0 (visual only): command, config, env, metrics, logs
│       ├── logs_w1.0/              # text_weight = 1.0
│       ├── R2_*_LongVideoBench_*.json
│       └── matching_results/       # R2_Tstar, R2_VSI
├── account/                        # Development timeline, commit history and correspondence with the instructor
├── jupyternotebook.ipynb           # Assignment 1: Arabic NER with CRFs
├── ANERCorp_CamelLab_test.pred.txt # Assignment 1: predictions on the ANERCorp test set
└── README.md
```

## Why
Visual-only keyframe searchers cannot handle text-related tasks. Although the VSI method that utilizes text information has achieved certain improvements in text-related tasks, it still has not solved all the problems.

## Problem
At present, there are still several issues that need to be studied: 
- VSI has increased the accuracy rate of locating the key frames for text-related problems to 45%, but this figure is still very low. What is the reason for this? Can further improvements be made in text positioning?
- Could we make better use of the text information rather than simply conducting a search similar to "control F"?

## Reproduced repositories
- [Visual-Subtitle-Integration](https://github.com/Jacksonha7/Visual-Subtitle-Integration) (VSI, CVPR 2026 Findings): the main system. I rebuilt the environment and ran keyframe search end-to-end on a real video.

## Overleaf Link
- [Project](https://www.overleaf.com/read/spfjgfpsbsdr#e336e6)
- [Proposal](https://www.overleaf.com/read/spfjgfpsbsdr#e336e6)
- [Assignment 1](https://www.overleaf.com/read/spfjgfpsbsdr#e336e6)
- [Mid-Term Report](https://www.overleaf.com/read/spfjgfpsbsdr#e336e6)
- [Final Project](https://cn.overleaf.com/3693258572dwjbngzzpzyx#c552c4)


## To-do
 
- [x] Reproduce the keyframe_search function of VSI
- [x] Encoder ablation: MiniLM, MPNet, BGE (weeks 8–10)
- [ ] Error analysis of the text-referred questions that are still missed (weeks 11–12)
- [ ] Text weight chosen per question instead of per task type (weeks 13–14, if time allows)
- [ ] Final report (10 pages, ACL format), code repository and demo (weeks 14–16)
