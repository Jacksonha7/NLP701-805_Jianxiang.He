# Textual Semantic Matching for Keyframe Search in Long-Video Understanding

## 1.Why
Visual-only keyframe searchers cannot handle text-related tasks. Although the VSI method that utilizes text information has achieved certain improvements in text-related tasks, it still has not solved all the problems.

## 2.Problem
At present, there are still several issues that need to be studied: 
- VSI has increased the accuracy rate of locating the key frames for text-related problems to 45%, but this figure is still very low. What is the reason for this? Can further improvements be made in text positioning?
- Could we make better use of the text information rather than simply conducting a search similar to "control F"?

## 3.Reproduced repositories
- [Visual-Subtitle-Integration](https://github.com/Jacksonha7/Visual-Subtitle-Integration) (VSI, CVPR 2026 Findings): the main system. I rebuilt the environment and ran keyframe search end-to-end on a real video.
- [Logic-in-Frames](https://github.com/guoweiyu/Logic-in-Frames) (VSLS, NeurIPS 2025): the visual-only baseline with logical relations between objects.

### 3.1 Reproduction process
**Step1:** 
you need to configure the environment according to the code in the original [Visual-Subtitle-Integration](https://github.com/Jacksonha7/Visual-Subtitle-Integration) code repository and [Logic-in-Frames](https://github.com/guoweiyu/Logic-in-Frames) code respository and download the [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench) dataset.

**Step2：** Use `Keyframe_Matching.py` to evaluate the three files in `reproduced/`.
Each method returns the top 8 keyframes per question. A keyframe counts as a hit
if it lies within 200 frames of a ground-truth position

```bash
python Keyframe_Matching.py --input_json reproduced/Tstar_PTonly_LongVideoBench_gpt4_cot_vid1.json --num_frame 8 --threshold 200
python Keyframe_Matching.py --input_json reproduced/VSI_PTonly_LongVideoBench_gpt4_cot_vid1.json   --num_frame 8 --threshold 200
```

| Method | Modality | Video-level acc. (160 Qs) | Video-level acc. (all 220 Qs) | Frame-level acc. |
|---|---|---|---|---|
| T\* | Visual | 23.12% (37/160) | 16.82% (37/220) | 5.55% (71/1280) |
| **VSI** | Visual + subtitle | **45.00%** (72/160) | **32.73%** (72/220) | **14.61%** (187/1280) |

- **Video-level acc.** is the share of questions where at least one of the 8 keyframes is a hit.
  This is the "overall match rate" that `Keyframe_Matching.py` prints.
- **Frame-level acc.** is the number of hit keyframes divided by the total number of returned keyframes (160 × 8 = 1280).
- **160 vs. 220 questions:** the subset has 220 questions, but in each file 60 of them have no keyframe output.
  `Keyframe_Matching.py` leaves these questions out, so its numbers use 160 questions.

## 4.New features added

## 5.Overleaf Link
- [Project](https://www.overleaf.com/read/spfjgfpsbsdr#e336e6)
- [Proposal](https://www.overleaf.com/read/spfjgfpsbsdr#e336e6)
- [Assignment 1](https://www.overleaf.com/read/spfjgfpsbsdr#e336e6)
- [Mid-Term Report](https://www.overleaf.com/read/spfjgfpsbsdr#e336e6)


## 6.To-do
 
- [x] Reproduce the keyframe_search function of VSI
- [ ] Encoder ablation: MiniLM, MPNet, BGE (weeks 8–10)
- [ ] Error analysis of the text-referred questions that are still missed (weeks 11–12)
- [ ] Text weight chosen per question instead of per task type (weeks 13–14, if time allows)
- [ ] Final report (10 pages, ACL format), code repository and demo (weeks 14–16)
