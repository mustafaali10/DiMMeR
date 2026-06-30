# DiMMeR: Diffusion-Based Mobile Motion with DDIM for ROS2 Nav2

**Authors:** Shuaib Ahmed · Mohammed Mustafa Ali · Pankaj Somkuwar (Mentor) · Gulam · Sai
*Montclair State University — MS Computer Science, 2026*

---

## Overview

DiMMeR ports **DiPPeR** (an ICRA 2024 diffusion-based 2D path planner) from ROS1 into **ROS2 Nav2** as a custom Global Planner plugin, replacing the standard NavfnPlanner with a learned diffusion model. The system was deployed on **KryptoClean** — a HiWonder JetRover Mecanum-wheel robot running on a Jetson Orin Nano (~$900 platform vs. $74,500+ legged robots used in original DiPPeR). The key contribution is swapping the DDPM-100 sampler for **DDIM-50**, achieving **100% path success rate** with **2.5× faster inference**.

---

## How It Works

DiMMeR frames path planning as a denoising problem. Given a 100×100 occupancy map image, a ResNet-18 encoder extracts visual features. A FiLM-conditioned U-Net iteratively denoises a random noise trajectory into a smooth, collision-free path — with start and goal coordinates pinned via inpainting throughout the process.

### Figure 1 — Diffusion Denoising Progression

<img width="1536" height="1024" alt="Diffusion Progress" src="https://github.com/user-attachments/assets/86fe1f98-aa76-45ec-a57d-923dc9371982" />


*From left to right: (1) randomly scattered noise trajectories, (2) partial convergence with irregularities mid-denoising, (3) final clean, collision-free path connecting start to goal. This illustrates what DDIM accomplishes across its denoising steps.*

---

## System Architecture

### Figure 2 — DiMMeR Architecture Stack

<img width="694" height="680" alt="dimmer architecture" src="https://github.com/user-attachments/assets/9771c2d6-341b-4c2b-b8a5-15e64e2e19cc" />


*The maze occupancy image is processed by a ResNet-18 encoder into a latent feature vector. This vector conditions a FiLM-based U-Net via timestep embeddings, which progressively denoises the path trajectory. Skip connections preserve spatial detail across down/bottleneck/up blocks. Endpoint pinning fixes start and goal throughout inference, ensuring the output path always connects the correct points.*

---

## Sampler Comparison Results

All sampler experiments were run offline by Gulam on Google Colab (A100 GPU) across **100 test mazes** using the epoch 150 checkpoint (loss: 0.00136).

### Figure 3 — Sampler Success Rate & Inference Time

<img width="3940" height="1092" alt="chart_v3_all_panel" src="https://github.com/user-attachments/assets/f4bb2318-8eeb-43f7-9246-028b4f932780" />


| Sampler        | Success Rate | Inference Time |
|----------------|-------------|----------------|
| DDPM-100       | ~87%        | ~1140ms        |
| **DDIM-50**    | **100%**    | **~456ms**     |
| DPM-Solver++-50| 87%         | —              |
| DDIM-20        | 62%         | —              |
| PNDM           | ❌ Incompatible | —           |

**DDIM-50 is the optimal sampler** — highest success rate, 2.5× faster than DDPM-100.

---

## Trajectory Visualizations

### Figure 4 — Sampler Path Comparison (4-Panel)

<img width="1956" height="985" alt="image (17)" src="https://github.com/user-attachments/assets/34c5fe9c-59cc-4dd6-bdea-536908398d7d" />
<img width="1956" height="985" alt="image (16)" src="https://github.com/user-attachments/assets/1e747f0e-fd51-49d0-9bfb-ae89d192e226" />
<img width="1956" height="985" alt="image (15)" src="https://github.com/user-attachments/assets/d7a6cb89-f803-45a7-875e-0f16ff819738" />
<img width="1956" height="985" alt="image (15)" src="https://github.com/user-attachments/assets/af924887-3d0a-4e42-8083-e4aaa2ab9174" />




*Side-by-side trajectory outputs across samplers on the same maze. DDIM-50 produces the cleanest, most direct path. DDIM-20 shows shortcuts that clip into walls. PNDM produces scattered, unusable trajectories.*

---

## Hardware — KryptoClean

| Component         | Spec                          |
|-------------------|-------------------------------|
| Platform          | HiWonder JetRover (Mecanum)   |
| Compute           | NVIDIA Jetson Orin Nano 8GB   |
| LiDAR             | RPLIDAR A1                    |
| Depth Camera      | Orbbec DaBai DCW              |
| Object Detection  | YOLOv8 + CBAM (TACO dataset)  |
| Total Cost        | ~$900                         |

---

## ROS2 Stack
SLAM Toolbox → AMCL → Nav2 (NavfnPlanner + DWB) → YOLOv8 Detector → Arm Pickup
DiMMeR Plugin lives in ros2_ws (research) — live demo runs NavfnPlanner (coordinate scaling bug is future work)

---

## Key Results

- ✅ DDIM-50: **100% success rate**, 456ms avg inference across 100 mazes  
- ✅ 2.5× faster than DDPM-100  
- ✅ KryptoClean completed end-to-end trash pickup on real hardware (video proof)  
- 🔧 DiMMeR live hardware deployment blocked by coordinate scaling bug → listed as future work

---
