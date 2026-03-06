# A Two-Stage Fusion Framework for Few-Shot Species Identification from Melting Curve Data

This repository contains the official implementation of our paper: **"A Two-Stage Fusion Framework for Few-Shot Species Identification from Melting Curve Data"**.

## Overview

We propose a two-stage multimodal fusion framework for species identification using High-Resolution Melting (HRM) curve data. The framework addresses the challenge of few-shot classification across 65 species by combining:

1. **Stage 1 — Text-based species representation via BERT**: Melting curve features are converted into structured textual prompts describing curve characteristics (peak counts, temperature ranges, statistical properties, etc.). A fine-tuned BERT model learns discriminative species representations from these descriptions.

2. **Stage 2 — Multimodal fusion with cross-attention**: A patch-based time-series encoder processes raw melting curves, while the pre-trained BERT model provides textual embeddings. These two modalities are fused via a cross-attention mechanism with learnable queries, combined with supervised contrastive learning for improved class separation.

## Architecture

| Component | Description |
|---|---|
| **Time-series encoder** | Patch embedding + cross-attention with learnable queries |
| **Text encoder** | Pre-trained BERT (fine-tuned in Stage 1) with [CLS] token pooling |
| **Fusion** | Text embedding concatenated as an additional patch token for cross-modal attention |
| **Loss** | Cross-entropy + optional supervised contrastive loss |

We also provide baseline comparisons using **ResNet-18** and **Vision Transformer (ViT)** operating on melting curve plot images.

## Project Structure

```
├── run.py                  # Main entry: multimodal fusion model (Stage 2)
├── bert_classification.py  # Stage 1: BERT fine-tuning on text prompts
├── bert_attn_score.py      # BERT attention analysis and visualization
├── train_mlp.py            # MLP baseline on handcrafted features
├── run_resnet18.py         # ResNet-18 image baseline
├── run_vit.py              # ViT image baseline
├── model.py                # Multimodal fusion model (patch encoder + BERT + cross-attention)
├── model_resnet18.py       # ResNet-18 token model
├── model_vit.py            # ViT classifier
├── experiment.py           # Training/evaluation loop for the fusion model
├── dataset.py              # Time-series dataset with optional text prompts
├── dataset_multimodal.py   # Multimodal dataset (time-series + images)
├── contrastive_loss.py     # Supervised contrastive loss (SupCon)
├── utils.py                # Utility functions (seed setting)
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
- matplotlib, Pillow (for image baselines)

Install dependencies:
```bash
pip install torch transformers scikit-learn pandas numpy scipy matplotlib pillow tqdm
```

## Usage

### Stage 1: Fine-tune BERT on text prompts

```bash
python bert_classification.py
```

This trains a BERT classifier on species-descriptive text prompts across 10 random seeds and saves checkpoints to `bert_checkpoints/`.

### Stage 2: Train multimodal fusion model

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
| `--train_epochs` | 3000 | Number of training epochs |
| `--learning_rate` | 1e-6 | Learning rate |
| `--batch_size` | 32 | Batch size |

### Baselines

**ResNet-18 (image-based):**
```bash
python run_resnet18.py --seeds 0 1 2 3 4 5 6 7 8 9
```

**ViT (image-based):**
```bash
python run_vit.py --variant small --seeds 0 1 2 3 4 5 6 7 8 9
```

**MLP (handcrafted features):**
```bash
python train_mlp.py
```

## Data

The dataset consists of HRM melting curve measurements for 65 species. Each sample contains:
- **Melting curve**: raw fluorescence intensity values across a temperature gradient
- **Text prompt**: structured natural language description of curve characteristics (generated from extracted features)
- **Species label**: one of 65 species classes

Data splits are generated with 10 random seeds for robust evaluation.

> **Note**: The dataset is not included in this repository. Please contact the authors for access.

## Citation

If you find this work useful, please cite:

```bibtex
@article{anonymous2025twostage,
  title={A Two-Stage Fusion Framework for Few-Shot Species Identification from Melting Curve Data},
  author={Anonymous},
  year={2026}
}
```
