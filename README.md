# A Two-Stage Fusion Framework for Few-Shot Species Identification from Melting Curve Data

> 🎉 **Our work has been accepted by ACM-BCB 2026.**

This repository contains the official implementation of our paper **"A Two-Stage Fusion Framework for Few-Shot Species Identification from Melting Curve Data"**.

## Overview

We propose a two-stage multimodal fusion framework for species identification from High-Resolution Melting (HRM) curve data. The framework targets the few-shot classification setting across 65 species and combines two complementary stages:

1. **Stage 1 — Text-based species representation via BERT.** Each melting curve is converted into a structured textual prompt that describes its characteristics (peak counts, temperature ranges, statistical properties, etc.). A fine-tuned BERT model learns discriminative species representations from these descriptions.

2. **Stage 2 — Multimodal fusion with cross-attention.** A patch-based time-series encoder processes the raw melting curve, while the BERT model from Stage 1 provides textual embeddings. The two modalities are fused through a cross-attention mechanism with learnable queries, optionally combined with supervised contrastive learning for tighter class separation.

## Architecture

| Component | Description |
|---|---|
| **Time-series encoder** | Patch embedding + cross-attention with learnable queries |
| **Text encoder** | Pre-trained BERT (fine-tuned in Stage 1) with `[CLS]` pooling |
| **Fusion** | BERT text embedding concatenated as an additional patch token for cross-modal attention |
| **Loss** | Cross-entropy + optional supervised contrastive loss |

## Project Structure

```
├── run.py                  # Main entry: trains the Stage 2 multimodal fusion model
├── bert_classification.py  # Stage 1: BERT fine-tuning on species text prompts
├── model.py                # Multimodal fusion model (patch encoder + BERT + cross-attention)
├── experiment.py           # Training / evaluation loop for the fusion model
├── dataset.py              # Time-series dataset with optional text prompts
├── utils.py                # Utility functions (e.g., seed setting)
└── layers/
    ├── Embed.py            # Patch embedding for time series
    └── Encoder_Layer.py    # Cross-attention encoder layer
```

## Requirements

- Python >= 3.9
- PyTorch >= 1.12
- Transformers (HuggingFace)
- scikit-learn
- pandas, numpy, scipy
- tqdm

Install dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy scipy tqdm
```

## Usage

### Stage 1 — Fine-tune BERT on text prompts

```bash
python bert_classification.py
```

This trains a BERT classifier on species-descriptive text prompts across 10 random seeds and saves the resulting checkpoints into `bert_checkpoints/`. These checkpoints are loaded by the Stage 2 model.

### Stage 2 — Train the multimodal fusion model

```bash
python run.py --use_text --d_model 128 --train_epochs 3000 --learning_rate 1e-6
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--d_model` | 128 | Hidden dimension |
| `--use_text` | True | Enable BERT text modality |
| `--use_gate` | False | Enable head-specific gating (Qwen-style) |
| `--contrastive_weight` | 0 | Weight for supervised contrastive loss |
| `--temperature` | 0.07 | Temperature for the contrastive loss |
| `--train_epochs` | 3000 | Number of training epochs |
| `--learning_rate` | 1e-6 | Learning rate |
| `--batch_size` | 32 | Batch size |
| `--seq_len` | 3478 | Melting curve sequence length |

## Data

The dataset consists of HRM melting curve measurements for 65 species. Each sample contains:

- **Melting curve** — raw fluorescence intensity values across a temperature gradient.
- **Text prompt** — a structured natural-language description of the curve's characteristics, generated from extracted features.
- **Species label** — one of 65 species classes.

Data splits are generated with 10 random seeds for robust evaluation.

> **Note:** The dataset is not included in this repository. Please contact the authors for access.

## Citation

If you find this work useful, please cite our ACM-BCB 2026 paper:

```bibtex
@inproceedings{zhang2026twostage,
  title     = {A Two-Stage Fusion Framework for Few-Shot Species Identification from Melting Curve Data},
  author    = {Zhang, Chengqian and Lee, Kyumin and Tong, Zhongyi},
  booktitle = {Proceedings of the 17th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM-BCB)},
  year      = {2026}
}
```
