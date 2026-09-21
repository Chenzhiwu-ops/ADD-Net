# ADD-Net: Core Modules

This repository provides the core modules proposed for ADD-Net:

- `models/HAFM.py` — HAFM implementation
- `models/PCIM.py` — PCIM implementation
- `models/LPC_Head.py` — LPC-Head implementation

## Base Framework

ADD-Net builds on the [official Mamba-YOLO implementation](https://github.com/HZAI-ZJNU/Mamba-YOLO). Please refer to that repository for the baseline framework and its installation instructions. The original Mamba-YOLO code is not redistributed here.

## Scope of This Repository

This repository contains the ADD-Net-specific modules listed above. It does not currently include a standalone training and evaluation pipeline or the datasets. The modules need to be integrated into the baseline framework according to the architecture and experimental settings described in the manuscript.

## Acknowledgment

We thank the authors of Mamba-YOLO for making their implementation publicly available.
