# OpenVLA on NVIDIA Jetson AGX Thor

This repository demonstrates **deployment of OpenVLA (Vision-Language-Action) on NVIDIA Jetson Thor T5000**, enabling real-time multimodal inference for robotic control on embedded AI hardware.

---

## Overview

OpenVLA is a **multimodal AI framework** that integrates:

- **Vision encoders**: ViT (DINOv2) and SigLIP  
- **Language model backbone**: LLaMA-based transformer  
- **Action decoder**: Generates robot control tokens and converts them into continuous actions  

This implementation adapts OpenVLA for **Jetson Thor T5000**, optimizing for GPU acceleration and low-latency inference in robotics applications.

---

## Features

- Jetson Thor-compatible OpenVLA inference pipeline  
- Fusion of multiple vision backbones for enriched patch embeddings  
- LLaMA-based multimodal transformer for text + image input  
- Token-to-action conversion for continuous robotic control  
- Debug-friendly logging for tensor shapes, prefill, and decoding steps  
- End-to-end example scripts for action prediction

---

## Inference Pipeline

1. **Image Preprocessing**  
   Input image is processed into patches for ViT and SigLIP backbones.

2. **Vision Feature Extraction**  
   Extract patch features and fuse embeddings from both backbones.

3. **Projection to LLaMA Hidden Space**  
   Vision features projected to match the transformer hidden size.

4. **Multimodal Fusion**  
   Concatenate vision embeddings with text embeddings (prompt tokens).

5. **Transformer Forward Pass**  
   LLaMA processes the fused embeddings, producing logits for each token.

6. **Decoder & Action Generation**  
   Autoregressive generation produces action tokens, which are mapped to discrete bins and converted into **continuous robot actions**.

---

