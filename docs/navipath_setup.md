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

The first code change should make the WSI loader configurable for CONCH
features, while preserving the original ConSlide loader path for baseline
reproduction.

Initial target file:

- `datasets/seq_wsi.py`

Likely work items:

- Add configurable dataset roots for the four TCGA cohorts.
- Support CONCH feature files if their on-disk layout differs from the
  original `h5_files/{slide_id}.h5` layout.
- Keep the original `features` / `features2` path available for ConSlide
  baseline runs.
- Add a small inventory script or sanity check before training.
