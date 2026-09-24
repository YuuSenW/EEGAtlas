# EEGAtlas Code

This folder contains the EEGAtlas model implementation and training scripts for the five downstream tasks. It does not include a pretraining engine, datasets, or model weights.

- `EEGAtlas.py`, `Layer/`, and `utils.py`: encoder and its model components.
- `Downstream/EEGAtlas_PhysioP300.py`: P300 detection.
- `Downstream/EEGAtlas_KaggleERN.py`: error-related negativity detection.
- `Downstream/EEGAtlas_BCIC2A.py` and `Downstream/EEGAtlas_BCIC2B.py`: motor imagery.
- `Downstream/EEGAtlas_Sleepedf.py`: sleep staging.
