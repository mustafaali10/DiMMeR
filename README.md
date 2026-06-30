# DiMMeR: Diffusion-Based Mobile Motion with DDIM for ROS2 Nav2

**Authors:** Mohammed Mustafa Ali · Gulam · Sai · Pankaj (Mentor)  
*Montclair State University — MS Computer Science, 2026*

---

## Overview

DiMMeR ports **DiPPeR** (an ICRA 2024 diffusion-based 2D path planner) from ROS1 into **ROS2 Nav2** as a custom Global Planner plugin, replacing the standard NavfnPlanner with a learned diffusion model. The system was deployed on **KryptoClean** — a HiWonder JetRover Mecanum-wheel robot running on a Jetson Orin Nano (~$900 platform vs. $74,500+ legged robots used in original DiPPeR). The key contribution is swapping the DDPM-100 sampler for **DDIM-50**, achieving **100% path success rate** with **2.5× faster inference**.

---

## How It Works

DiMMeR frames path planning as a denoising problem. Given a 100×100 occupancy map image, a ResNet-18 encoder extracts visual features. A FiLM-conditioned U-Net iteratively denoises a random noise trajectory into a smooth, collision-free path — with start and goal coordinates pinned via inpainting throughout the process.

### Figure 1 — Diffusion Denoising Progression

![Diffusion Denoising Progression](images/diffusion_progression.png)

*From left to right: (1) randomly scattered noise trajectories, (2) partial convergence with irregularities mid-denoising, (3) final clean, collision-free path connecting start to goal. This illustrates what DDIM accomplishes across its denoising steps.*

---

## System Architecture

### Figure 2 — DiMMeR Architecture Stack

![DiMMeR Architecture](images/dimmer_architecture.png)

*The maze occupancy image is processed by a ResNet-18 encoder into a latent feature vector. This vector conditions a FiLM-based U-Net via timestep embeddings, which progressively denoises the path trajectory. Skip connections preserve spatial detail across down/bottleneck/up blocks. Endpoint pinning fixes start and goal throughout inference, ensuring the output path always connects the correct points.*

---

## Sampler Comparison Results

All sampler experiments were run offline by Gulam on Google Colab (A100 GPU) across **100 test mazes** using the epoch 150 checkpoint (loss: 0.00136).

### Figure 3 — Sampler Success Rate & Inference Time

![Sampler Comparison Chart](images/chart_v3_all_panel.png)

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

![Sampler Trajectory Comparison](images/sampler_trajectories.png)

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
