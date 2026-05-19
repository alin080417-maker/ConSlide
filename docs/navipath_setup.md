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
- Current task order:
  BRCA -> NSCLC -> RCC -> ESCA

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
- `--wsi_feature_dim`
- `--wsi_pt_feature_layout {flat,hit}`
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

## CAN/QPMIL CONCH Dataset Adapter

The downloaded CAN-style dataset uses this layout:

```text
can_dataset/
  tcga_brca/
    table/*.csv
    datasplit/fold_*.npz
    feats-l1-s256_CONCH/pt_files/*.pt
```

Convert it into a ConSlide-compatible tree without copying feature tensors:

```bash
python scripts/prepare_can_dataset.py \
  --source_root /home/alan0804/CL/Dataset/03_Data/can_dataset \
  --output_data_root /home/alan0804/CL/Dataset/navipath_conslide \
  --output_split_root /home/alan0804/CL/Dataset/navipath_conslide_splits
```

Then use these training data arguments:

```bash
--wsi_data_root /home/alan0804/CL/Dataset/navipath_conslide \
--wsi_split_root /home/alan0804/CL/Dataset/navipath_conslide_splits \
--wsi_feature_subdir feats-l1-s256_CONCH \
--wsi_feature_format pt \
--wsi_backbone abmil \
--wsi_feature_dim 512 \
--wsi_pt_feature_layout flat \
--wsi_missing_feature2 copy \
--wsi_num_workers 0
```

Use `--wsi_num_workers 0` in restricted or notebook-like environments. On a
normal training machine, increase it back to `4` or higher.

`--wsi_pt_feature_layout hit` is a compatibility adapter for the original HIT
backbone. It expands each flat CONCH patch feature into repeated `8 x 8`
tokens so the original ConSlide forward path can run. For NaviPath, prefer a
single-stream ABMIL-style backbone instead of treating this adapter as the final
model design.

Run a short ABMIL sanity check with:

```bash
PYTHONPATH=/home/alan0804/CL/ConSlide python utils/main.py \
  --model conslide \
  --dataset seq-wsi \
  --exp_desc brca_abmil_sanity \
  --buffer_size 1100 \
  --alpha 0.2 \
  --beta 0.2 \
  --n_epochs 1 \
  --wsi_data_root /home/alan0804/CL/Dataset/navipath_conslide \
  --wsi_split_root /home/alan0804/CL/Dataset/navipath_conslide_splits \
  --wsi_feature_subdir feats-l1-s256_CONCH \
  --wsi_feature_format pt \
  --wsi_backbone abmil \
  --wsi_feature_dim 512 \
  --wsi_pt_feature_layout flat \
  --wsi_missing_feature2 copy \
  --wsi_num_workers 0 \
  --non_verbose
```

Run an ER-ACE baseline sanity check with:

```bash
PYTHONPATH=/home/alan0804/CL/ConSlide python utils/main.py \
  --model er_ace \
  --dataset seq-wsi \
  --exp_desc er_ace_abmil_sanity \
  --buffer_size 1100 \
  --minibatch_size 16 \
  --n_epochs 1 \
  --wsi_data_root /home/alan0804/CL/Dataset/navipath_conslide \
  --wsi_split_root /home/alan0804/CL/Dataset/navipath_conslide_splits \
  --wsi_feature_subdir feats-l1-s256_CONCH \
  --wsi_feature_format pt \
  --wsi_backbone abmil \
  --wsi_feature_dim 512 \
  --wsi_pt_feature_layout flat \
  --wsi_missing_feature2 copy \
  --wsi_num_workers 0 \
  --non_verbose
```
