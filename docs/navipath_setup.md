# NaviPath Setup Notes

This branch is the working branch for adapting ConSlide into the NaviPath
continual WSI pipeline.

## Git Remotes

- `origin`: `git@github.com:alin080417-maker/ConSlide.git`
- `upstream`: `https://github.com/HKU-MedAI/ConSlide.git`
- working branch: `navipath-conch-integration`

## Current ConSlide Data Assumptions

The original `seq-wsi` dataset is defined in `datasets/seq_wsi.py`.

Current loader behavior:

- Dataset class: `Sequential_Generic_MIL_Dataset`
- Slide dataset class: `Generic_MIL_Dataset`
- Expected feature path:
  `DATA_DIR/h5_files/{slide_id}.h5`
- Expected HDF5 keys:
  - `features`
  - `features2`
  - `coords`
- Current task order after `datasets.reverse()`:
  ESCA -> RCC -> BRCA -> NSCLC

## NaviPath First Integration Target

The first code change makes the WSI loader configurable for CONCH features,
while preserving the original ConSlide loader path for baseline reproduction.

Initial target file:

- `datasets/seq_wsi.py`

Implemented knobs:

- `--wsi_data_root`
- `--wsi_split_root`
- `--wsi_feature_subdir`
- `--wsi_feature_format {h5,pt}`
- `--wsi_h5_feature_key`
- `--wsi_h5_feature2_key`
- `--wsi_h5_coords_key`
- `--wsi_missing_feature2 {error,copy,zeros}`

The default values reproduce the original ConSlide layout. For single-stream
CONCH features, use `--wsi_missing_feature2 copy` or `--wsi_missing_feature2
zeros` until the second stream is replaced by a model-side design.

## Feature Inventory Check

Before training, verify that CSV files, split files, feature files, and feature
keys align:

```bash
python scripts/inspect_wsi_features.py \
  --wsi_data_root ../Dataset \
  --wsi_split_root ../HIT/10fold_splits \
  --wsi_feature_subdir patch_4096/convnexts_l0l1_512_4096 \
  --wsi_feature_format h5 \
  --feature_key features
```
