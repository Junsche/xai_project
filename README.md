# XAI Project

Protocol-driven experiment and analysis code for a thesis on how data augmentation affects clean performance, calibration, and corruption robustness in image classification.

This repository studies augmentation beyond standard clean-test accuracy. It asks whether methods that look strong on clean in-distribution data remain strong under distribution shift, and whether those effects stay stable across domains and model architectures.

## Highlights

- Unified `Stage-1` / `Stage-2` / `Stage-3` evaluation pipeline
- Cross-domain comparison on natural and medical image datasets
- Cross-architecture comparison with `ResNet-18` and `ViT-B`
- Joint analysis of accuracy, calibration, and corruption robustness
- Thesis-ready tables, plots, and summary artifacts

## Overview

The project covers:

- natural-image datasets: `CIFAR-10`, `CIFAR-100`
- medical-image datasets: `DermaMNIST`, `PathMNIST`
- corruption benchmarks: `CIFAR-C`, `MedMNIST-C`
- backbone families: `ResNet-18`, `ViT-B`

For the medical-image setting, robustness is evaluated on a shared subset of `8` `MedMNIST-C` corruptions:

- `defocus_blur`
- `motion_blur`
- `pixelate`
- `jpeg_compression`
- `brightness_up`
- `brightness_down`
- `contrast_up`
- `contrast_down`

## Research Goal

Many augmentation methods are still judged mainly through clean accuracy. This repository follows a broader evaluation perspective and studies:

- clean predictive performance
- calibration and confidence reliability
- robustness under common corruptions
- domain dependence of augmentation effects
- architecture dependence of augmentation effects

The goal is not only to rank augmentation methods once, but to understand when those rankings change with the data domain, corruption protocol, and model family.

## Pipeline

The experiments follow a three-stage protocol:

1. `Stage-1`: learning-rate selection  
   A controlled short-budget search is used to select stable learning rates for each `(model, dataset)` pair.

2. `Stage-2`: formal training and augmentation comparison  
   Models are trained on clean in-distribution data using the selected learning rate and a fixed stage budget.

3. `Stage-3`: robustness and calibration evaluation  
   Saved checkpoints are evaluated on clean test data and corruption benchmarks.

This separation helps avoid tuning on corrupted test results and keeps the robustness analysis easier to interpret.

## What This Repository Contains

This repository mainly stores:

- experiment analysis scripts
- exported metrics and intermediate CSV files
- generated robustness tables and figures
- summary artifacts for the thesis
- LaTeX thesis sources and integrated outputs

It is best understood as an experiment-and-analysis repository rather than a polished end-user training framework.

## Repository Layout

```text
.
├── common/      # shared helpers, schema, naming, loading, checks
├── data/        # export and preprocessing utilities
├── stage1/      # learning-rate selection analysis
├── stage2/      # clean-data comparison analysis
├── stage3/      # corruption robustness analysis
├── summary/     # thesis-ready summary artifacts
├── assets/      # figure-generation utilities
├── thesis/      # LaTeX thesis source and compiled outputs
└── temp_data/   # local temporary dataset files
```

For the full training repository, the execution code is typically structured around:

- `configs/` for dataset, model, augmentation, and protocol configuration
- `runners/` for stage orchestration
- `data_modules/` for dataset and corruption loading
- `models/` for architecture construction
- `train/` and `eval/` for optimization and evaluation logic
- `runs/` and `wandb/` for checkpoints and experiment tracking

## Configuration Philosophy

The original experiment code is designed around explicit protocol files so that different `(model, dataset)` combinations can be configured consistently.

Typical configuration layers include:

- `configs/base/`
- `configs/datasets/`
- `configs/models/`
- `configs/augs/`
- `configs/protocol/`

Typical protocol files include:

- `stage1_lr_grid.yaml`
- `stage1_epochs.yaml`
- `stage2_selected_lrs.yaml`
- `stage2_epochs.yaml`
- `default_augmentations.yaml`

This setup makes the experiment matrix easier to maintain, audit, and reproduce.

## Typical Workflow

1. Export run data from Weights & Biases into local CSV files.
2. Inspect Stage-1 outputs and select stable learning rates.
3. Run Stage-2 training for the target augmentations.
4. Evaluate Stage-2 checkpoints in Stage-3 on clean and corrupted test sets.
5. Aggregate metrics into figures and tables.
6. Integrate the outputs into the thesis.

## Example Commands

If your remote experiment repository still uses the protocol-driven runner layout, typical commands look like this.

### Stage-1

```bash
python runners/run_stage1_lr_grid.py <dataset> <model>
```

Examples:

```bash
python runners/run_stage1_lr_grid.py cifar10 resnet18
python runners/run_stage1_lr_grid.py cifar10 vit_b
```

### Stage-2

```bash
python runners/run_stage2_aug_comparison.py <dataset> <model> [aug_name]
```

Examples:

```bash
python runners/run_stage2_aug_comparison.py cifar10 resnet18 baseline
python runners/run_stage2_aug_comparison.py cifar10 vit_b baseline
python runners/run_stage2_aug_comparison.py cifar10 resnet18
```

### Stage-3

```bash
python runners/run_stage3_robustness.py <dataset> [aug] [severity]
```

Examples:

```bash
python runners/run_stage3_robustness.py cifar10 baseline
python runners/run_stage3_robustness.py cifar10
```

Depending on the implementation, the Stage-3 runner may delegate to:

```bash
python eval/stage3.py ...
```

If your remote repository uses different file names, adjust these examples before publishing.

## Metrics

The evaluation covers both predictive performance and predictive reliability.

Main metrics include:

- `accuracy`
- `balanced accuracy`
- `ECE`
- `MCE`
- `mCA`
- `mECE`

This makes it possible to compare not only whether a model predicts correctly, but also whether its confidence remains trustworthy under shift.

## Compared Methods

The experiment pipeline is designed for comparisons across multiple augmentation families, including:

- baseline training
- `AutoAugment`
- `RandAugment`
- `Mixup`
- `CutMix`
- `AugMix`
- `StyleAug`
- `Rotation + Random Erasing`
- `DiffuseMix`

The exact set available in a given run depends on the exported W&B runs and the training repository configuration.

## Checkpoint Convention

In the protocol-driven version of the training repository, Stage-2 checkpoints are typically written to `runs/` using a naming convention that encodes:

- model
- dataset
- augmentation
- experiment identifier
- learning rate
- seed

Example pattern:

```text
runs/<model>_<dataset>_<aug>_<exp_id>_lr<lr>_seed1437_last.pt
```

This allows Stage-3 evaluation scripts to reconstruct checkpoint paths automatically.

## Reproducibility Notes

This repository supports reproducibility by preserving:

- exported experiment metrics
- intermediate analysis tables
- generated figures
- thesis-ready summary outputs
- the thesis source itself

Full reproduction may still require:

- access to the original Weights & Biases runs
- access to the training repository or checkpoint sources
- local environment setup with the required Python packages
- access to the original datasets under their respective licenses

## Practical Notes

- Prefer `Ctrl+C` over `Ctrl+Z` when interrupting long experiments.
- Check GPU state before launching memory-heavy runs, especially for `ViT-B`.
- Keep `runs/` and `wandb/` organized so formal outputs remain easy to audit.
- In practice, finishing the main `ResNet-18` track before the `ViT-B` extension track often simplifies scheduling and result management.

## Data and Third-Party Resources

This repository uses or evaluates on publicly available datasets and corruption benchmarks. All credit for the original datasets, benchmark definitions, and official implementations belongs to the respective authors and maintainers.

Relevant upstream resources include:

- MedMNIST official repository: <https://github.com/MedMNIST/MedMNIST>
- MedMNIST paper: <https://www.nature.com/articles/s41597-022-01721-8>
- MedMNIST-C official repository: <https://github.com/francescodisalvo05/medmnistc-api>
- MedMNIST-C paper: <https://arxiv.org/abs/2406.17536>
- CIFAR-C benchmark paper: <https://arxiv.org/abs/1903.12261>

For the medical-image track, this project uses `DermaMNIST` and `PathMNIST` from the `MedMNIST` collection and evaluates robustness using a fixed shared corruption subset derived from `MedMNIST-C`.

If specific files in this repository are adapted from third-party code, their original sources and licenses should be acknowledged either in-file or in a separate `THIRD_PARTY_NOTICES.md` document.

## License and Usage Considerations

Before redistributing data, derived assets, or copied source code, please check the original licenses of the underlying resources.

Important examples:

- `MedMNIST` code is distributed under `Apache-2.0`
- `MedMNIST` data is generally distributed under `CC BY 4.0`
- `MedMNIST-C` code is distributed under `Apache-2.0`
- some medical subsets may include additional restrictions, including non-commercial conditions depending on the source dataset

Users of this repository are responsible for ensuring that their use of the underlying datasets and benchmark material complies with the original licenses and data terms.

## Citation

If you use this repository, its analysis protocol, or its benchmark framing in academic work, please cite the relevant original benchmark papers.

```bibtex
@article{yang2023medmnist,
  title={MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={41},
  year={2023}
}

@article{disalvo2024medmnistc,
  title={MedMNIST-C: Comprehensive Benchmark and Improved Classifier Robustness by Simulating Realistic Image Corruptions},
  author={Di Salvo, Francesco and Doerrich, Sebastian and Ledig, Christian},
  journal={arXiv preprint arXiv:2406.17536},
  year={2024}
}

@inproceedings{hendrycks2019benchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}
```

## Status

This is an academic experiment repository accompanying a thesis. Its primary purpose is to document the evaluation pipeline, preserve outputs, and support transparent reporting across datasets, corruption benchmarks, and model architectures.
