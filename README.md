# EEGAtlas code

This directory accompanies the EEGAtlas paper. It contains the encoder and its components, the pretraining configuration and Lightning module, and scripts for five downstream tasks. The paper describes the method and evaluation protocol. Appendix A documents dataset preparation, while Appendix B reports implementation settings and pretraining results.

## Contents

| Path | Purpose |
| --- | --- |
| `EEGAtlas.py` | Encoder, predictor, and reconstructor definitions. The momentum target encoder is instantiated in `engine_pretraining.py`. |
| `Layer/` | Patch embedding, Dynamic Channel Relation Atlas, GroupAttention, PMoE, and their helper functions. |
| `utils.py` | Masking and optimization helpers. |
| `configs.py` | Pretraining model and data-loader configuration. |
| `engine_pretraining.py` | PyTorch Lightning module for the self-supervised objectives and optimization. |
| `requirements.txt` | Pinned packages from the development environment, with installation notes. |
| `Downstream/EEGAtlas_PhysioP300.py` | Frozen-probe training for P300 detection. |
| `Downstream/EEGAtlas_KaggleERN.py` | Frozen-probe training for error-related negativity detection. |
| `Downstream/EEGAtlas_BCIC2A.py` | Frozen-probe training for four-class motor imagery. |
| `Downstream/EEGAtlas_BCIC2B.py` | Frozen-probe training for two-class motor imagery. |
| `Downstream/EEGAtlas_Sleepedf.py` | Frozen-probe training for five-class sleep staging. |
| `Downstream/prepare_sleep.py` | SleepEDF epoch preparation. |
| `Downstream/readme.md` | Dataset download and preparation notes. |

## Environment

The development environment used Python 3.10. The principal dependencies are PyTorch, torchvision, PyTorch Lightning, NumPy, SciPy, pandas, scikit-learn, MNE, einops, and tqdm. SleepEDF preparation additionally requires Braindecode.

`requirements.txt` records the development environment's package versions and includes platform setup notes. Install a PyTorch build appropriate for your hardware, then use the listed dependencies to configure the environment. Several training scripts are configured for CUDA.

From the repository root, the model module can be imported with:

```bash
python -c "from ICLR2027.Code.EEGAtlas import EEGTransformer; print('EEGAtlas import OK')"
```

## Data and checkpoints

Pretraining uses SEED, M3CV, THU-Benchmark, PhysioNet-MI, and the laboratory-collected Cue-Reactivity corpus. The five downstream tasks use public datasets. Their sources and preprocessing are described in `Downstream/readme.md` and Appendix A.

`configs.py` currently points to a local prepared pretraining corpus under `C:\EEGPretraing\All_float32`. Replace its training and validation roots with paths to your prepared data. The downstream scripts expect a pretrained checkpoint containing `target_encoder.*` parameters, conventionally named `EEGAtlas_large.ckpt`. Update their checkpoint paths before running.

Each downstream script expects prepared data in its own layout:

| Task | Expected input | Evaluation grouping |
| --- | --- | --- |
| PhysioP300 | Class folders of subject-coded trial files | Nine-subject leave-one-subject-out (LOSO) |
| KaggleERN | Train/test CSV recordings and label files | Four predefined, overlapping subject folds |
| BCIC-2A | `sub{ID}/data.pt` and `label.pt` from the official training partition | Nine-subject LOSO |
| BCIC-2B | `sub{ID}/data.pt` and `label.pt` from the official training partition | Nine-subject LOSO |
| SleepEDF | Class-organized, 30-second `.pt` epochs | Four folds over eight subjects |

The BCIC-2A and BCIC-2B evaluations use the official training partitions. `Downstream/prepare_sleep.py` prepares SleepEDF epochs.

## Using the code

1. Obtain and preprocess the datasets according to Appendix A and `Downstream/readme.md`.
2. Replace the local data roots in `configs.py` and the relevant downstream script. Place the pretrained checkpoint at the path expected by that script, or update its `load_path`.
3. For pretraining, use the Lightning module in `engine_pretraining.py`. For downstream evaluation, run the corresponding task script with its task-specific data and helper modules.
4. Match the subject partitions, random seeds, checkpoint-selection rules, and metrics in Appendix B before comparing results with the paper.

The model components can be imported from the repository root. The downstream scripts use task-specific `Modules`, `utils`, and `utils_eval` helpers. The paper evaluates all downstream tasks with seeds 7, 42, and 718; Appendix B specifies the task-specific settings and result aggregation.
