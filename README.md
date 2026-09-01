OSPA-Net

Official implementation of OSPA-Net: Optical–SAR Semantic Collaboration with Physics-Guided Arbitration for Semisupervised SAR Ship Detection.

The code is being organized. Detailed documentation, checkpoints, and experimental results will be added after the paper is accepted.

Overview

OSPA-Net is a SAR-centered semisupervised ship detection framework. It consists of two semantic teacher–student pairs and a parameter-free physical arbitration teacher. The optical branch provides transferable ship semantics, while the SAR branch performs target-domain learning and final inference. Physical evidence is used to verify teacher-agreed, SAR-only, and optical-only pseudo-label candidates.

Main Files

stage1_dior_pretrain.py: optical branch pretraining on DIOR ship images.

stage2_dior_sar100_pretrain.py: SAR branch pretraining on labeled DIOR and SARDet-100K images.

stage3_ospa_sar100_10percent.py: semisupervised OSPA-Net training configuration.

sar100_10percent_dataset.py: labeled and unlabeled SARDet-100K data configuration.

triple_teacher.py: main OSPA-Net detector implementation.

Requirements

The current implementation is based on:

PyTorch

MMEngine

MMCV 2.x

MMDetection 3.3.0

Data Preparation

The code uses the DIOR ship subset and SARDet-100K ship subset. Update the dataset paths in the configuration files before training:

dior_root = '/path/to/DIOR/dior_ship/'
sar100_root = '/path/to/SARDet_100K/'

The annotations should follow COCO format.

Training

Run the three training stages in sequence:

python tools/train.py configs/ospa_net/stage1_dior_pretrain.py
python tools/train.py configs/ospa_net/stage2_dior_sar100_pretrain.py
python tools/train.py configs/ospa_net/stage3_ospa_sar100_10percent.py

Before Stage 3, replace the checkpoint paths in the configuration file:

stage1_checkpoint = '/path/to/stage1_checkpoint.pth'
stage2_checkpoint = '/path/to/stage2_checkpoint.pth'

Ensure that Stage 2 and Stage 3 use the same labeled-data ratio.

Evaluation

python tools/test.py \
    configs/ospa_net/stage3_ospa_sar100_10percent.py \
    /path/to/checkpoint.pth

The EMA SAR teacher is used for inference by default.

Acknowledgements

This implementation is built on MMDetection.
