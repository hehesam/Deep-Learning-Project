# Swin Transformer: A Hierarchical Vision Transformer Using Shifted Windows

This repository contains a detailed review, summary, and hands-on implementation of the Swin Transformer—a state-of-the-art vision transformer architecture that introduces a hierarchical structure and shifted windows for efficient vision tasks.

> **Authors**: Bahaar Khalilian, Hesam Mohebi  
> **Course**: Deep Learning – Final Project

---

## 📄 Paper Summary

The **Swin Transformer** (ICCV 2021) addresses limitations in prior vision transformers (ViT, DeiT), especially for dense prediction tasks. It reduces computational complexity from quadratic to linear by applying **window-based self-attention** and introduces **shifted windows** for cross-window connection.

Key improvements over ViT:
- Hierarchical structure with progressive downsampling
- Shifted Window (SW-MSA) mechanism
- Strong scalability and flexibility for classification, detection, and segmentation

---

## 🧠 Methodology

### 🔁 The Pipeline

- Input image → split into **4x4 patches**
- Linear embedding projects to C dimensions
- Hierarchical structure reduces spatial dimensions while increasing channel depth
- Final structure:

| Stage         | Spatial Size     | Channel Size |
|---------------|------------------|---------------|
| Input         | H × W × 3        | 3             |
| Patch Embed   | H/4 × W/4        | 48 → C        |
| Stage 1       | H/4 × W/4        | C             |
| Stage 2       | H/8 × W/8        | 2C            |
| Stage 3       | H/16 × W/16      | 4C            |
| Stage 4       | H/32 × W/32      | 8C            |

### 🧱 Swin Transformer Block

- Replaces global attention with:
  - **Window-based MSA (W-MSA)**: localized attention
  - **Shifted Window MSA (SW-MSA)**: connects non-overlapping windows
- Uses MLPs, LayerNorm, and residuals in Transformer-style architecture

### 🧬 Architecture Variants

| Variant   | Channels (C) | Layers per Stage | Size w.r.t ViT-B |
|-----------|--------------|------------------|------------------|
| Swin-T    | 96           | {2,2,6,2}        | ~0.25×           |
| Swin-S    | 96           | {2,2,18,2}       | ~0.5×            |
| Swin-B    | 128          | {2,2,18,2}       | 1×               |
| Swin-L    | 192          | {2,2,18,2}       | ~2×              |

---

## 📊 Experiments

### 📌 Image Classification on ImageNet-1K

- Best results from Swin-L (384²) with fine-tuning: **87.26% Top-1 accuracy**
- Efficient training using AdamW, cosine decay LR, warm-up, large batch sizes
- Scales well with ImageNet-22K pretraining

### 📦 Object Detection on COCO

- Tested with Cascade Mask R-CNN, ATSS, Sparse RCNN
- Swin-T improves AP when replacing ResNet50
- Swin-L + HTC++ achieves top results in full detection pipeline

### 🧩 Semantic Segmentation on ADE20K

- Swin + UperNet achieves **62.8 mIoU**
- Outperforms both CNNs and older Transformer-based models

### 🔍 Ablation Studies

- Shifted Windows (SW-MSA) consistently improve:
  - +1.1% accuracy (ImageNet)
  - +2.0 AP (COCO)
  - +0.8 mIoU (ADE20K)

---

## 💻 Code & Reproducibility

### ✅ Image Classification

- Repository: [Swin Transformer GitHub](https://github.com/microsoft/Swin-Transformer)
- Challenges faced:
  - PyTorch DDP errors on Windows → bypassed with single-GPU mode
  - Folder naming required WordNet synsets
  - Reorganized ImageNet validation data using ground-truth mappings
- Final reproduced results:

| Model             | Pretrained | Top-1 | Top-5 | Time (s) | Memory (MB) |
|------------------|------------|-------|-------|----------|--------------|
| Swin-T           | No         | 81.17 | 95.52 | 0.294    | 515          |
| Swin-S           | No         | 83.21 | 96.23 | 0.320    | 597          |
| Swin-B           | No         | 83.42 | 96.45 | 0.348    | 876          |
| Swin-B (384²)    | Yes        | 80.91 | 96.01 | 0.299    | 515          |
| Swin-L (224²)    | Yes        | 86.44 | 98.05 | 0.465    | 1539         |
| Swin-L (384²)    | Yes        | 86.25 | 97.88 | 0.650    | 2719         |
| Swin-L (384²)+FT | Yes        | 87.26 | 98.24 | 0.952    | 4285         |

### ❌ Object Detection

- Repo: [Swin Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
- Tried on WSL with Anaconda
- MMDetection and MMCV version mismatches caused errors
- Scripts not compatible with newer MMEngine-based frameworks

### ❌ Semantic Segmentation

- Repo used MMSegmentation with MMCV
- Version conflicts caused dependency deadlocks
- Difficult to upgrade without breaking CUDA or PyTorch compatibility

---

## ⚠️ Limitations

### ⚡ Instability with Large-Scale Models

- Swin V1 suffers from exploding gradients with 1B+ parameters
- Swin V2 introduces:
  - Pre-normalization
  - RMSNorm for better gradient flow

### 🔁 Relative Positional Bias

- Swin V1 uses fixed-size learnable bias → not scale-invariant
- Swin V2 uses continuous log-spaced bias → generalizes to any resolution

### 🎯 Downstream Generalization

| Task         | Swin V1 (Base) | Swin V2 (Base) | Swin V2 (Giant) |
|--------------|----------------|----------------|------------------|
| ImageNet-1K  | ~83.5%         | ~85.5%         | 87.1%            |
| COCO box AP  | ~51.6          | ~54.4          | 57.5             |
| ADE20K mIoU  | ~48.1          | ~51.6          | 55.5             |

---

## 🧾 References

1. Swin Transformer – ICCV 2021  
2. ViT – ICLR 2021  
3. DeiT – PMLR 2021  
4. Swin V2 – CVPR 2022  

---

## 📁 Repo Structure

```bash
.
├── swin_review.md                # This README
├── notebooks/
│   └── swin_eval.ipynb           # Image classification experiments
├── scripts/
│   └── eval_single_gpu.py        # Modified evaluation script
├── reports/
│   └── swin_review.pdf           # Full paper summary and evaluation
