# Semantic Flip

### Synthetic OOD Generation for Robust Refusal in Embodied Question Answering and Spatial Localization

Dongbin Na<sup>&#42;&dagger;</sup> &nbsp; Chanwoo Kim<sup>&#42;</sup> &nbsp; Giyun Choi &nbsp; Dooyoung Hong<sup>&dagger;</sup><br/>
<sub>RGA Inc. &nbsp;<sup>&#42;</sup>equal contribution &nbsp;<sup>&dagger;</sup>corresponding</sub>

[![Paper](https://img.shields.io/badge/Paper-PDF-B0303F.svg?style=flat-square)](https://ndb796.github.io/SemanticFlip)
[![Project Page](https://img.shields.io/badge/Project-Page-181A1F.svg?style=flat-square)](https://ndb796.github.io/SemanticFlip)
![Python](https://img.shields.io/badge/Python-3.10+-555.svg?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-555.svg?style=flat-square)

Embodied vision-language agents tend to answer confidently even when their visual memory does not actually support the query, which becomes dangerous the moment a wrong answer turns into a wrong physical action. Semantic Flip teaches the agent *when to refuse*. This method synthesizes out-of-distribution (OOD) query–memory pairs from in-distribution data, with no external OOD annotation, and trains a small rejection module on top of a **frozen** VLM. The module drops into an existing pipeline without touching the underlying model.

This repository contains the official source codes for Semantic Flip and the **SpaceReject** benchmark.

## Abstract

> Detecting unanswerable user queries remains essential for the reliable deployment of real-world embodied agents. However, modern vision-language models (VLMs) often generate overly confident answers even when the available visual memory cannot support the query. Such overconfidence carries different costs across tasks: the agent may give misleading information in Embodied Question Answering, or pick an arbitrary coordinate and physically guide the user there in spatial reasoning for navigation. Despite these stakes, only a few prior studies address when and how an embodied VLM should respond with "I do not know."
We propose **Semantic Flip**, a simple and effective framework that synthesizes auxiliary OOD samples for embodied refusal without external OOD annotation. The key idea is to independently transform the query and the video memory to construct auxiliary OOD pairs that lack sufficient visual grounding. These synthesized pairs train a lightweight rejection module on top of a frozen pretrained VLM, and the module attaches to any existing VLM-based pipeline without retraining the underlying model. Across two complementary benchmarks, Semantic Flip consistently outperforms strong prompting baselines. We also introduce **SpaceReject**, a new refusal benchmark for spatial localization with deliberately unanswerable queries over long video memory, where Semantic Flip reaches an F1 of 0.9559. All experiments use only open-source models, so the full pipeline is reproducible.

## Method

Semantic Flip builds two complementary OOD distributions from answerable training pairs by flipping exactly one side of an otherwise answerable pair:

- **Q-Flip** rewrites the question into an ungroundable variant while keeping the video memory unchanged.
- **V-Flip** keeps the question but erases its referent from the memory through a parse &rarr; detect &rarr; inpaint pipeline (spaCy, Grounding-DINO, LaMa).

A frozen VLM encoder produces one joint embedding for the in-distribution, Q-Flip, and V-Flip samples, and only a small 3-layer MLP rejection gate is trained on top. Because the backbone stays frozen, the gate reuses the forward pass the agent already runs and adds essentially no extra inference cost.

## Results

With a frozen 7B encoder and a small head, Semantic Flip reaches F1 0.7110 on AbstainEQA and 0.9559 on SpaceReject, above strong prompting baselines. It also generalizes to abstention categories never seen during synthesis: for example, 0.89 OOD recall on *Information Unavailability*, a category Q-Flip does not target.

#### AbstainEQA (HM3D-380)

| Method | F1 | Bal. Acc | Recall | Spec. |
|:--|:--:|:--:|:--:|:--:|
| Qwen2.5-VL-32B prompting (Coarse / Fine / CoT) | – | – | – | – |
| **Semantic Flip (ours)** | **0.7110** | **0.6684** | 0.8158 | 0.5211 |

#### SpaceReject (spatial localization over long video memory)

| Method | BalAcc | F1 | Recall | Spec. |
|:--|:--:|:--:|:--:|:--:|
| C2 (Tool), Qwen3-8B prompting | 0.8778 | 0.8874 | 0.9630 | 0.7926 |
| Semantic Flip, Q-Flip only | 0.9504 | 0.9494 | 0.9363 | 0.9644 |
| **Semantic Flip, Q-Flip + V-Flip** | **0.9563** | **0.9559** | 0.9467 | 0.9659 |

## Installation

```bash
git clone https://github.com/ndb796/SemanticFlip.git
cd SemanticFlip
conda create -n semflip python=3.10 -y
conda activate semflip
pip install -r requirements.txt
```

A single 48 GB GPU (e.g. RTX A6000) is enough for the whole pipeline; the 32B baseline uses the AWQ-quantized checkpoint. All checkpoints (`Qwen2.5-VL-7B/32B-AWQ`, `Qwen2.5-7B`, `grounding-dino-tiny`) are public and are pulled from the Hugging Face Hub on first use. If your environment needs authenticated downloads, export a token first:

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

## Reproducing the experiments

Open the unified notebook and run it top to bottom:

```bash
jupyter lab notebooks/reproduce.ipynb
```

It rebuilds the deterministic split, synthesizes the Q-Flip and V-Flip pools, extracts frozen `Qwen2.5-VL-7B` features, trains the 5-seed ensemble gate, reproduces the headline F1 of 0.7110, runs the appendix ablations, and writes ready-to-paste LaTeX table fragments. A full run takes about 6–8 hours on one A6000; every stage is resume-safe, so you can switch a stage off once its cache exists (see the `RUN_*` flags in the first cell).

## Datasets and models

- **AbstainEQA (HM3D-380).** Built on HM3D episodes; the notebook rebuilds the split deterministically (`SEED = 42`).
- **SpaceReject / SpaceRejectExtra.** A new spatial-localization refusal benchmark over long video memory. The queries, videos, annotations, and trained models will be released on **Hugging Face**; see the [project page](https://ndb796.github.io/SemanticFlip) for the link (coming soon).
- **Backbones.** `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`, `Qwen/Qwen2.5-7B-Instruct`, and `IDEA-Research/grounding-dino-tiny`.

The released default synthesizes one rewrite per (question, strategy), which reproduces the headline F1 of 0.7110. `QFLIP_SAMPLES_PER_PAIR` in the first cell controls the synthesis pool size if you want to scale it up.

## Repository structure

```
SemanticFlip/
├── notebooks/
├── src/
│   ├── synthesis/      # Q-Flip (query rewriting) + V-Flip (inpainting) generators
│   ├── gate/           # ensemble MLP rejection gate
│   └── data/           # HM3D / AbstainEQA loaders
├── baselines/          # prompt-based and learned baselines
├── configs/            # experiment configs
├── assets/             # figures
├── docs/               # project page
└── requirements.txt
```

Some modules (the standalone synthesis library, the baseline implementations, and the SpaceReject pipeline) are still being cleaned up and currently ship as placeholders. The reproduction notebook is self-contained and does not depend on them; the remaining source and the SpaceReject data and code will be added incrementally.

## Citation

```bibtex
@article{na2026semanticflip,
  title   = {Semantic Flip: Synthetic OOD Generation for Robust Refusal
             in Embodied Question Answering and Spatial Localization},
  author  = {Na, Dongbin and Kim, Chanwoo and Choi, Giyun and Hong, Dooyoung},
  year    = {2026}
}
```

## Acknowledgements

Built with open-source models from the Qwen team and IDEA-Research, and the [LaMa](https://github.com/advimman/lama) inpainting model.

## Contact

Correspondence to Dongbin Na (`dongbinna@postech.ac.kr`) and Dooyoung Hong (`dooyoung@rgarobot.com`).
