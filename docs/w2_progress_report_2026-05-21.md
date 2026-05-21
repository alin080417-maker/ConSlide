# W2 進度整理

日期：2026-05-21

這份文件整理目前 W2 在 ConSlide -> NaviPath integration 的進度。
內容以工作紀錄為主，不是論文式報告。

## 1. 這裡的 baseline 是什麼

在這個專案裡，baseline 不只是一個數字。

它通常指任何拿來比較的參考 run，例如：

- lower-bound 方法，例如 plain finetuning 或簡單 rehearsal 方法
- 論文 / 原始 repository 的 ConSlide 設定
- 使用 CONCH features 的 NaviPath adaptation

所以這裡的 baseline，本質上是 performance comparison point。
真正重要的是 baseline 必須對齊：

- 同一份 dataset
- 同一個 task order
- 同一個 split protocol
- 同一種 feature format
- 同一類 backbone

如果這些沒有對齊，數字仍然可以當 sanity check，
但不能當成嚴格 reproduction number。

## 2. 目前專案狀態

目前 branch：

- `navipath-conch-integration`

目前工作重點是讓 ConSlide 可以吃 NaviPath / CONCH feature tree，
同時保留原本 ConSlide pipeline 的可讀性與可執行性。

目前已確認的事情：

- 原本 ConSlide training code path 在本機可以跑
- 移除 hardcoded GPU pin 後，GPU access 正常
- CONCH `.pt` features 可以被 ConSlide loader 讀取
- dataset conversion script 可以產生 ConSlide-compatible folders
- current AKD-PMP working model 的 smoke test 已經成功跑完

目前還沒完全定案的地方：

- dataset 仍然不完整，所以正式 reproduction number 還不是最終版
- AUC 在小 split 或單一類別 split 下可能不穩定，甚至出現 `nan`
- AKD-PMP model 是 working adaptation，不是原始論文方法

## 3. ConSlide 是什麼

ConSlide 是一個針對 WSI classification 的 continual learning framework。

核心概念是：

- 每個 task 對應一個 cancer cohort / subtype group
- model 依照 task sequential training
- rehearsal-based 方法會用 memory buffer
- evaluation 會追蹤 class-il 與 masked accuracy

在原始 repository 中，WSI pipeline 主要在：

- `datasets/seq_wsi.py`
- `backbone/`
- `models/`
- `utils/training.py`

## 4. WSI 是什麼

WSI 是 Whole Slide Image。

在這份 codebase 裡，WSI 不會直接餵 raw gigapixel image。
它是被轉成 patch-level features 的 bag。

對單一 slide 來說：

- `N` = patch 數量
- `D` = 每個 patch 的 feature dimension

model 看到的 tensor 形狀是：

- `features`: `[N, D]`

目前 NaviPath / CONCH 設定下：

- `N` 會因 slide 而變
- `D = 512`

原始 ConSlide 預設設定下：

- `D = 768`

## 5. BRCA 是什麼

BRCA 指 breast cancer cohort。

在目前 sequential task setup 中：

- BRCA task classes 是 `IDC` 和 `ILC`
- mapped labels 是 `0` 和 `1`

目前轉好的 dataset 中，BRCA 有：

- 53 slides total
- 38 `IDC`
- 15 `ILC`

## 6. CONCH 是什麼

CONCH 是 NaviPath 這邊用的 feature extractor。

目前 dataset 是 CONCH patch embeddings，存成 `.pt` files。

本機實際看到的內容是：

- payload type: `torch.Tensor`
- tensor shape: `[337, 512]`
- dtype: `torch.float32`

也就是說，單一 slide 是一個 bag，裡面有 337 個 patch token，
每個 token 是 512 維 feature。

補充：

- loader 也支援 dictionary payload，例如包含 `features` 和 `features2`
- 但目前看到的 CONCH files 是 plain tensor

## 7. 什麼叫 ConSlide feature

原本 ConSlide pipeline 是為 dual-stream HDF5 feature layout 設計的。

`datasets/seq_wsi.py` 的原始預設假設是：

- `features`
- `features2`
- `coords`

原始預設 feature directory 是：

- `patch_4096/convnexts_l0l1_512_4096`

原始 default backbone 吃的 feature dimension 是：

- `768`

所以在原本設定裡，slide 會被當成 HDF5 feature tensors，
而且要符合 dual-stream loader。

在 NaviPath integration 裡，我們把 loader 改成也能讀 single-stream `.pt`
feature bags。

## 8. 目前 dataset 長什麼樣

轉好的 dataset 放在：

- `/home/alan0804/CL/Dataset/navipath_conslide`
- `/home/alan0804/CL/Dataset/navipath_conslide_splits`

目前資料夾結構是：

```text
navipath_conslide/
  TCGA-BRCA/
    tcga-brca_label.csv
    feats-l1-s256_CONCH -> 指向 source CONCH feature directory 的 symlink
  TCGA-NSCLC/
    tcga-nsclc_label.csv
    feats-l1-s256_CONCH -> symlink
  TCGA-RCC/
    tcga-kidney_label.csv
    feats-l1-s256_CONCH -> symlink
  TCGA-ESCA/
    tcga-esca_label.csv
    feats-l1-s256_CONCH -> symlink

navipath_conslide_splits/
  BRCA_100/
    splits_0.csv
    ...
  NSCLC_100/
    splits_0.csv
    ...
  RCC_100/
    splits_0.csv
    ...
  ESCA_100/
    splits_0.csv
    ...
```

## 9. CAN / QPMIL CONCH data 是怎麼轉的

原始 dataset source：

- `/home/alan0804/CL/Dataset/03_Data/can_dataset`

原始 layout 如下：

```text
can_dataset/
  tcga_brca/
    table/*.csv
    datasplit/fold_*.npz
    feats-l1-s256_CONCH/pt_files/*.pt
  tcga_lung/
  tcga_rcc/
  tcga_esca/
```

conversion script 是：

- `scripts/prepare_can_dataset.py`

它的工作流程：

1. 讀 source table CSV，路徑在 `table/`
2. 過濾掉沒有 `.pt` features 的 slide
3. 把 source subtype labels 映射成 ConSlide 用的 label names
4. 寫出 ConSlide-style label CSV，欄位有：
   - `slide_id`
   - `case_id`
   - `label`
5. 把 CONCH feature directory 用 symlink 接到 ConSlide tree
6. 把每個 `fold_*.npz` 轉成 `splits_*.csv`

目前 label mapping 是：

- BRCA: `IDC`, `ILC`
- NSCLC: `LUAD`, `LUSC`
- RCC: `CCRCC`, `PRCC`
- ESCA: `Adenocarcinoma`, `Squamous cell carcinoma`

被忽略的 labels：

- RCC 忽略 `CHRCC`
- BRCA 忽略一些 rare labels，例如 `MDLC`, `PD`, `ACBC`,
  `IMMC`, `BRCNOS`, `BRCA`, `SPC`, `MBC`, `MPT`
- ESCA 忽略 `Tubular adenocarcinoma` 與
  `Basaloid squamous cell carcinoma`

## 10. split file 格式

split CSV 由 source `fold_*.npz` 生成。

每個 split file 有三欄：

- `train`
- `val`
- `test`

每一欄都是 slide ID，長度不夠的地方會用 `NaN` 補齊。

目前 fold 0 的數量是：

- BRCA_100: train 49, val 3, test 1
- NSCLC_100: train 47, val 6, test 3
- RCC_100: train 471, val 55, test 61
- ESCA_100: train 120, val 15, test 15

conversion script 預設會產生 10 folds，因為
`prepare_can_dataset.py` 的 `--n_folds` 預設值是 10。

## 11. ConSlide loader 格式

目前 WSI dataset loader 在：

- `datasets/seq_wsi.py`

重要 class：

- `Sequential_Generic_MIL_Dataset`
- `Generic_MIL_Dataset`
- `Generic_Split`

loader 期望每個 sample 回傳：

- `inputs0`
- `inputs1`
- `label`

對 HDF5 來說：

- `inputs0` 會從 primary key 讀，預設是 `features`
- `inputs1` 會從 secondary key 讀，預設是 `features2`
- coordinates 可以存在 `coords`

對 `.pt` 來說：

- 如果檔案內容是 tensor，就直接把那個 tensor 當成 `inputs0`
- `inputs1` 則依照 `--wsi_missing_feature2` 產生

支援的 fallback：

- `error`
- `copy`
- `zeros`

對 CONCH 來說，目前使用的是：

- `--wsi_missing_feature2 copy`

意思是單一 stream 的 CONCH bag 會被複製成兩份，
讓原本 dual-input code path 還能繼續跑。

## 12. PT layout 選項

loader 支援兩種 `.pt` layout：

- `flat`
- `hit`

意思是：

- `flat`: 保持 tensor 為 `[N, D]`
- `hit`: 把 flat `[N, D]` 展開成 HIT-compatible token map

HIT-compatible expansion 會把每個 patch feature 變成 `8 x 8` token grid：

- `[N, D] -> [N, 8, 8, D]`

對 NaviPath / CONCH 來說，建議使用：

- `--wsi_pt_feature_layout flat`

## 13. 目前訓練設定

原始 repository README 只給一個很短的 example：

```bash
python utils/main.py --state train --model conslide --dataset seq-wsi \
  --exp_desc conslide --buffer_size 1100 --alpha 0.2 --beta 0.2
```

真正的 default 設定在 `utils/args.py`：

- `lr = 1e-5`
- `n_epochs = 50`
- `batch_size = 1`
- `wsi_data_root = ../Dataset`
- `wsi_split_root = ../HIT/10fold_splits`
- `wsi_feature_subdir = patch_4096/convnexts_l0l1_512_4096`
- `wsi_feature_format = h5`
- `wsi_backbone = hit`
- `wsi_feature_dim = 768`
- `wsi_missing_feature2 = error`
- `wsi_num_workers = 4`
- `num_folds = 5`

rehearsal-based methods 還需要：

- `buffer_size`

目前 `utils/training.py` 的訓練流程是：

- task 0 有 10 epoch SSL warm-up
- 每個 task 接著訓練 `n_epochs`
- 每個 epoch 都會做 validation
- early stopping 的設定是：
  - patience = 10
  - stop_epoch = 10
- checkpoint 會存在 `./checkpoints/{exp_desc}`

## 14. GPU 與 runtime notes

早期遇到一個環境問題：

- `utils/main.py` 裡有 hardcoded 的 `CUDA_VISIBLE_DEVICES="1"`

這會讓本機看不到真正可用的 GPU，因為這台機器實際上是 GPU 0。
後來已經把這行拿掉。

修正後：

- 在 host 環境下 `torch.cuda.is_available()` 變成 `True`
- model 會進到 `cuda:0`
- `nvtop` / `nvidia-smi` 可以看到實際 GPU memory 使用量

## 15. 目前已經做了哪些 baseline 工作

目前 working tree 裡的主要改動有：

- `utils/args.py`
  - 新增 `--num_folds`
  - 保留 WSI feature configuration 的明確設定
- `utils/main.py`
  - 移除 hardcoded GPU pin
  - 印出 CUDA availability 與 model device
- `utils/training.py`
  - 修正 CSV logging，不要在原地 mutation `args`
- `models/akd_pmp.py`
  - 新增一個可跑的 AKD-PMP baseline model
  - 支援 rehearsal buffer 與 attention / logit distillation
- `docs/navipath_setup.md`
  - 補上較新的 setup notes 與 smoke command

## 16. AKD-PMP working model

目前的 `akd_pmp` 是一個 NaviPath-side baseline，
目的是讓 W2 工作持續往前。

它目前的用途是：

- 驗證 pipeline 可以在 CONCH features 上 end to end 跑通
- 提供一個可控的 rehearsal-style baseline
- 在正式 reproduction 前先建立比較點

目前 model 參數：

- model name: `akd_pmp`
- replay buffer: `buffer_size`
- distillation weight: `alpha`
- replay weight: `beta`
- pseudo-bag ratio: `pmp_ratio`
- distillation temperature: `akd_temperature`

## 17. smoke test 結果

目前 host-side smoke test 已成功完成，設定如下：

- `model = akd_pmp`
- `dataset = seq-wsi`
- `feature_format = pt`
- `feature_subdir = feats-l1-s256_CONCH`
- `wsi_backbone = abmil`
- `wsi_feature_dim = 512`
- `wsi_pt_feature_layout = flat`
- `wsi_missing_feature2 = copy`
- `wsi_num_workers = 0`
- `buffer_size = 2200`
- `n_epochs = 1`
- `num_folds = 1`

run 中有印出：

- `cuda_available: True | model_device: cuda:0`

這個 smoke run 有產生非隨機的 metrics。
不過因為 dataset 仍然不完整，所以這些數字不能直接當最終 reproduction number。

## 18. 目前 cohort 統計

目前轉好的 dataset 有：

- TCGA-BRCA: 53 slides
- TCGA-NSCLC: 56 slides
- TCGA-RCC: 674 slides
- TCGA-ESCA: 150 slides

class 分布：

- BRCA: 38 IDC, 15 ILC
- NSCLC: 38 LUAD, 18 LUSC
- RCC: 379 CCRCC, 208 PRCC, 87 CHRCC
- ESCA: 64 Adenocarcinoma, 86 Squamous cell carcinoma

經過 ConSlide-style cohort filtering 後：

- BRCA task 用 IDC / ILC
- NSCLC task 用 LUAD / LUSC
- RCC task 用 CCRCC / PRCC
- ESCA task 用 Adenocarcinoma / Squamous cell carcinoma

所以 continual sequence 還是 4 個 tasks，每個 task 2 classes。

### 18.1 feature / patch 統計

目前 CONCH `.pt` features 的維度固定為 `512`，但 patch 數量會變。

以下是掃過各 cohort 所有 `.pt` 檔後得到的統計：

| Task | Cohort | slides | min patches | median patches | mean patches | max patches |
|---|---|---:|---:|---:|---:|---:|
| 1 | BRCA | 73 | 49 | 365 | 352.64 | 511 |
| 2 | NSCLC | 101 | 35 | 328 | 307.40 | 511 |
| 3 | RCC | 711 | 96 | 3540 | 3505.87 | 16018 |
| 4 | ESCA | 158 | 62 | 3143.5 | 3261.23 | 11364 |

這代表：

- 每個 patch 的 feature 向量長度都是 `512`
- 每張 slide 的 patch 數量不固定
- BRCA / NSCLC 的 bag 比較小
- RCC / ESCA 的 bag 明顯大很多，常常到幾千個 patch

## 19. 這裡的 4 tasks 是什麼意思

sequential WSI benchmark 被分成四個 task：

1. BRCA
2. NSCLC
3. RCC
4. ESCA

每個 task 有兩個 classes，所以總 label space 是 8 classes。

model 會在每個 task 結束後做 evaluation，
追蹤的 metrics 包含：

- task-wise accuracy
- masked accuracy
- AUC
- forgetting / transfer 類型的摘要

## 20. 現在下一步是什麼

下一個真正有意義的步驟，是把所有設定固定後做正式比較：

- 同一份 dataset root
- 同一個 split root
- 同一個 backbone family
- 同一個 `num_folds`
- 同一個 buffer size
- 同一個 epoch budget

只有這樣，數字才會是正式的 baseline comparison，
不是單純的 sanity check。

## 21. 指令速查

轉換 CAN/QPMIL CONCH data：

```bash
python scripts/prepare_can_dataset.py \
  --source_root /home/alan0804/CL/Dataset/03_Data/can_dataset \
  --output_data_root /home/alan0804/CL/Dataset/navipath_conslide \
  --output_split_root /home/alan0804/CL/Dataset/navipath_conslide_splits
```

跑一個 CONCH sanity check：

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

跑 current AKD-PMP smoke test：

```bash
PYTHONPATH=/home/alan0804/CL/ConSlide python utils/main.py \
  --model akd_pmp \
  --dataset seq-wsi \
  --exp_desc akd_pmp_smoke \
  --buffer_size 2200 \
  --alpha 0.2 \
  --beta 0.2 \
  --n_epochs 1 \
  --num_folds 1 \
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

• 主要是這個檔：

  [datasets/seq_wsi.py](/home/alan0804/CL/ConSlide/datasets/seq_wsi.py:390)

  它才是真正把 ConSlide 的 WSI loader 改成可以讀 CONCH .pt features 的地方。關鍵改動在這幾段：

  - feature_format='h5' / 'pt' 的切換在 :392-408
  - 找 .pt 檔的路徑解析在 :431-450
  - 讀 .pt feature 的邏輯在 :482-509
  - 最後 __getitem__ 依照 h5 或 pt 走不同讀法在 :511-528

  另外還有兩個支援檔：

  - [utils/args.py](/home/alan0804/CL/ConSlide/utils/args.py:39)
    這裡新增了 --wsi_feature_format pt、--wsi_pt_feature_layout flat/hit、--wsi_missing_feature2 copy/zeros 這些參數，讓 loader 可以被命令列切換。
  - [scripts/prepare_can_dataset.py](/home/alan0804/CL/ConSlide/scripts/prepare_can_dataset.py:81)
    這個不是在「讀 CONCH」本身，而是把原始 can_dataset 轉成 ConSlide 可吃的資料夾結構，包含 tcga-*_label.csv 和 splits_*.csv。

  如果你只想記一個答案，就是：

  真正修改 ConSlide 讓它能讀 CONCH feature 的主檔是 datasets/seq_wsi.py。

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


loader可以被命令列切換


• 意思是：你不用改 Python 程式碼本身，就可以在執行時用指令決定「要讀 HDF5 還是讀 CONCH .pt」。

  例如這兩種跑法：

  --wsi_feature_format h5

  或

  --wsi_feature_format pt

  這個參數是在 utils/args.py 定義的，然後傳到 datasets/seq_wsi.py 裡的 loader。

  ———

  ## 白話理解

  以前 loader 可能是寫死的：

  - 只讀 .h5
  - 或只讀某種固定格式

  現在改成：

  - 你下指令時指定 --wsi_feature_format pt
  - loader 就去讀 .pt
  - 你指定 --wsi_feature_format h5
  - loader 就照原本方式讀 .h5

  這就是「可以被命令列切換」。

  ———

  ## 你現在能切換哪些東西

  除了 feature_format，你們還有這些也能用命令列切：

  - --wsi_feature_subdir feats-l1-s256_CONCH
      - 決定去資料夾哪個子目錄找 feature
  - --wsi_pt_feature_layout flat
      - 決定 .pt 要用 flat 還是 HIT 相容格式
  - --wsi_missing_feature2 copy
      - 如果 .pt 只有單一 feature，就怎麼補第二路

  ———

  ## 為什麼要這樣設計

  因為你們現在同一套程式要支援兩種資料來源：

  - 原始 ConSlide / HIT 那套 .h5
  - 你現在的 CONCH .pt

  如果沒有命令列切換，就得一直改程式碼，會很麻煩，也容易出錯。

  一句話講完：

  命令列切換就是「用參數決定 loader 行為」，不用改程式就能在不同資料格式間切換。

 為什麼會需要照h5或pt讀  我以為我們轉成固定格式嘞


• 因為你們現在的「固定格式」是 對 NaviPath 這條資料路線固定，不是整個 ConSlide 專案只剩一種格式。

  更直接講：

  - 原始 ConSlide 本來是讀 h5
  - 你們現在的 CONCH dataset 是轉成 pt
  - 所以 loader 需要同時支援兩種

  ———

  ## 你現在的「固定」是什麼

  你們現在已經把 NaviPath 這套資料定成：

  - navipath_conslide/TCGA-*/feats-l1-s256_CONCH/pt_files/*.pt
  - --wsi_feature_format pt
  - --wsi_pt_feature_layout flat

  這表示 跑 NaviPath 時，格式是固定的。

  但程式本身還保留：

  - h5 路徑，為了原始 ConSlide baseline
  - pt 路徑，為了 CONCH feature

  所以「固定」的是你實際要用的資料版本，不是把整個 loader 的能力刪掉。

  ———

  ## 為什麼要保留 h5 和 pt

  因為同一個專案常常要跑兩種東西：

  1. 原始 baseline
      - 用原始 ConSlide 資料
      - 讀 h5
  2. NaviPath / CONCH 版本
      - 用你們轉好的資料
      - 讀 pt

  這樣你才可以比較：

  - 原版 ConSlide 的表現
  - 你們 CONCH 版本的表現

  如果只留 pt，那原始 baseline 就不能直接跑了。
  如果只留 h5，那你們現在的 CONCH feature 就不能跑了。

  ———

  ## 所以你可以這樣理解

  不是「資料格式不固定」。
  而是：

  - 資料本身已經固定成 pt
  - loader 還保留 h5 是為了兼容原始 ConSlide 和做對照實驗

  ———

  ## 最短版

  你們轉好的 NaviPath 資料確實是固定成 pt 了；但 ConSlide loader 同時支援 h5 和 pt，是為了保留原始 baseline 和新版本對照，不是因為你現在這份資料還不固定。

