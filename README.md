# Automated and Scalable SEM Image Analysis of Perovskite Solar Cell Materials via a Deep Segmentation Framework

This repository presents an **automated deep learning-based segmentation framework** designed for **Scanning Electron Microscopy (SEM)** image analysis of **Perovskite Solar Cell (PSC)** materials. The framework enables efficient and precise identification of **PbI<sub>2</sub>** (lead iodide) and **perovskite** domains with the general formula **ABX<sub>3</sub>** across diverse morphologies, improving the characterization of **PSC thin films**.

## Overview

The analysis of SEM images plays a critical role in understanding the microstructure of thin films during the fabrication of perovskite solar cells. The ability to **identify and quantify PbI<sub>2</sub> and perovskite materials** is essential for:

- Optimizing crystallization processes
- Enhancing the performance of devices
- Accelerating material characterization

This framework overcomes the challenges of manual SEM analysis, offering an **automated, scalable solution** for real-time process monitoring and **quantitative microstructural analysis**.

### Features
- **Automated segmentation** for PbI<sub>2</sub> and perovskite materials
- Built on **YOLOv8x architecture**
- Incorporates novel modules:
  - **Adaptive Shuffle Dilated Convolution Block** for fine-grained feature extraction
  - **Separable Adaptive Downsampling Module** for better adaptability
- Achieves **87.25% mean average precision** (mAP)
- **Reduced computational load** by **24.43%** and **model size** by **25.22%**
- Reliable estimation of **PbI<sub>2</sub> domain area** and distribution
- Scalable tool for **quantitative analysis** in materials research

## Installation

To install the required dependencies, simply run:

```bash
pip install -r requirements.txt
