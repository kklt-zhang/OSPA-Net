OSPA-Net: Optical–SAR Semantic Collaboration with Physics-Guided Arbitration for Semisupervised SAR Ship Detection





This is the official PyTorch implementation of "OSPA-Net: Optical–SAR Semantic Collaboration with Physics-Guided Arbitration for Semisupervised SAR Ship Detection."

📢 News

[2026.09] The initial code and training configurations of OSPA-Net are released.

Detailed documentation, checkpoints, and experimental results will be updated after the paper is accepted.

💡 Introduction

Existing semisupervised object detection methods usually select pseudo labels according to network-derived semantic confidence. In cluttered SAR scenes, however, semantic confidence alone may be unreliable. Directly transferring optical ship knowledge to the SAR domain may also introduce semantic and localization errors because of the substantial imaging discrepancy between the two modalities.

To address these issues, we propose OSPA-Net, a SAR-centered semantic collaboration framework consisting of two trainable semantic teacher–student pairs and a parameter-free physical arbitration teacher:

Optical Semantic Branch: provides transferable ship semantics learned from labeled DIOR images.

SAR Semantic Branch: serves as the primary target-domain learning and detection branch.

Physical Arbitration Teacher: evaluates candidate boxes using local contrast, structural response, and background complexity.

Source- and Scale-Aware Pseudo-Label Generation: associates optical and SAR predictions, distinguishes teacher-agreed, SAR-only, and optical-only candidates, and verifies them using source-specific semantic and physical criteria.

During inference, only the EMA SAR teacher is used.

Usage

Requirements

Python=3.8

PyTorch=2.0+

MMCV=2.x

MMEngine

MMDetection=3.3.0

Please install MMDetection 3.3.0 and place the configuration, detector, and hook files in their corresponding MMDetection directories.

📂 Data

We use the optical DIOR dataset and the SAR SARDet-100K dataset:

DIOR

Paper

Dataset Download: Google Drive | Baidu Netdisk

SARDet-100K

Paper

Dataset Download: Kaggle | Baidu Netdisk

Official Repository

Please organize the processed ship subsets as follows:

data/
├── DIOR/
│   └── dior_ship/
│       ├── train.json
│       └── train/
│           └── images/
└── SARDet_100K/
    ├── Annotations/
    │   ├── instances_train_10percent.json
    │   ├── instances_unlabeled_90percent.json
    │   ├── val_ship.json
    │   └── test_ship.json
    └── JPEGImages/
        ├── train/
        ├── val/
        └── test/

Update dior_root and sar100_root in the configuration files according to your local paths.

🚀 Getting Started

Training

OSPA-Net follows a three-stage training procedure.

Stage 1: Pretrain the optical branch on DIOR

python tools/train.py configs/ospa_net/stage1_dior_pretrain.py

Stage 2: Pretrain the SAR branch on labeled DIOR and SARDet-100K images

python tools/train.py configs/ospa_net/stage2_dior_sar100_pretrain.py

Stage 3: Train OSPA-Net with labeled and unlabeled data

python tools/train.py configs/ospa_net/stage3_ospa_sar100_10percent.py

Before Stage 3, replace stage1_checkpoint and stage2_checkpoint with the corresponding pretrained weights. Please also ensure that Stage 2 and Stage 3 use the same labeled-data ratio.

Evaluation

python tools/test.py configs/ospa_net/stage3_ospa_sar100_10percent.py /path/to/checkpoint.pth

📝 Citation

If you find this project useful in your research, please consider citing our paper. The complete citation information will be updated after acceptance.

🙏 Acknowledgement

This project is built upon MMDetection. The semisupervised detection implementation is inspired by SoftTeacher, while the cross-domain training pipeline benefits from the insights of Dual Teacher. We sincerely thank the authors of these projects for their valuable open-source contributions.

✉️ Contact

For any questions, please feel free to open an issue or contact kklt_zhang@dlmu.edu.cn.
