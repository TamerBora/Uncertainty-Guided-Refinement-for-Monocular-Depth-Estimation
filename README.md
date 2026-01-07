  # Uncertainty-Guided Refinement for Monocular Depth Estimation

**Author:** Tamer Bora İkizoğlu  
**Course:** Deep Learning – Final Project  
**Supervisor:** Dr. Mohammad Sabokrou  

---

## 📌 Project Overview

This project addresses the **Sim-to-Real domain gap** in generative monocular depth estimation, specifically for **Marigold-based diffusion models**.  
It introduces a **Flat Residual CNN** that leverages the diffusion model’s own **uncertainty (variance)** to refine depth predictions and correct artifacts in real-world sensor data.

### Key Contributions

- **Uncertainty-Guided Refinement**  
  Uses pixel-wise variance maps to focus refinement on geometrically ambiguous regions.

- **Flat Residual Architecture**  
  A lightweight CNN designed to preserve high-frequency details better than traditional U-Net architectures.

- **Performance**  
  Achieves approximately **7% reduction in Absolute Relative Error (AbsRel)** on the **NYU Depth v2** dataset.

---

## 📂 Project Structure

```
.
├── make_marigold_data.py   # Generates training data (RGB + Mean Depth + Uncertainty)
├── train.py                # Main training loop (Flat Refiner or U-Net baseline)
├── flat_model.py           # Proposed Flat Residual Refiner architecture
├── model.py                # Baseline MiniUNet architecture
├── benchmark.py            # Quantitative evaluation (AbsRel, Delta1)
├── visualize.py            # Qualitative visual comparisons
├── evaluate.py             # Differentiable Least Squares alignment logic
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Generate Training Data (Crucial Step)

The repository **does not include large training tensors**.  
You must generate them locally before training.

This step requires:
- A GPU  
- Internet access (to stream the NYU Depth v2 dataset)

Run:

```bash
python make_marigold_data.py
```

**Output:**  
Creates a `data/processed_train/` directory containing the required `.npz` files for training.

---

## 🔬 Usage

### 1. Train the Models

You can train either the **proposed Flat Refiner** or the **U-Net baseline**.

#### Train Flat Refiner (Proposed Method)

```bash
python train.py --model flat
```

Saved checkpoint:

```
checkpoints/flat_refiner_best.pth
```

---

#### Train U-Net (Baseline)

```bash
python train.py --model unet
```

Saved checkpoint:

```
checkpoints/unet_refiner_best.pth
```

---

### 2. Evaluate (Benchmark)

To reproduce quantitative results (**AbsRel**, **Delta1**) on the test set:

#### Evaluate Proposed Method

```bash
python benchmark.py --checkpoint checkpoints/flat_refiner_best.pth
```

#### Evaluate Baseline

```bash
python benchmark.py --checkpoint checkpoints/unet_refiner_best.pth
```

### 3. Visual Analysis

To generate qualitative comparison images (saved to `viz_report_candidates/`):

**Visualize Proposed Method (Default):**

```bash
python visualize.py
```

**Visualize Baseline (U-Net):**

```bash
python visualize.py --checkpoint checkpoints/unet_refiner_best.pth
```

## 📊 Results Summary

- Improved robustness to real-world artifacts  
- Better preservation of geometric details  
- Demonstrates the effectiveness of uncertainty-aware refinement for diffusion-based depth estimation

---

## 📄 License & Acknowledgements

This project was developed as part of a **Deep Learning course final project**.  
All datasets and pretrained components are subject to their respective licenses.
