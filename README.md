# Textual Semantic Matching for Keyframe Search in Long-Video Understanding

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


## To-do
 
- [x] Reproduce the keyframe_search function of VSI
- [x] Encoder ablation: MiniLM, MPNet, BGE (weeks 8–10)
- [ ] Error analysis of the text-referred questions that are still missed (weeks 11–12)
- [ ] Text weight chosen per question instead of per task type (weeks 13–14, if time allows)
- [ ] Final report (10 pages, ACL format), code repository and demo (weeks 14–16)
