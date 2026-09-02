# SpongeNet for Deepfake Detection

This is the official implementation of the paper: **"SpongeNet: Preserving Forgery Traces by Knowledge Sponge with Binary Information Bottleneck for Deepfake Detection"**.

## 📢 News
- **[2026.3]** Our paper has been submitted to *Pattern Recognition* (PR) and is currently under review.
- **[2026.4]** Our paper is currently under major revision at *Pattern Recognition* (PR), and we plan to release the portions of the code related to the model architecture.
## 🚀 Code Release
**The complete source code, pre-trained weights, and dataset configurations will be released in this repository immediately upon the paper's acceptance.**

## ⚙️ Training Configuration & Hyperparameters

Because of space limitations in the main paper, this section documents the full set of training hyperparameters. They come from two places: the config file (`./configs/results_cifake_T.cfg`) and the fixed hyperparameters hard-coded in `./main_model.py` (optimizer, warm-up scheduler, BIB block, and the loss-weighting coefficients such as the `λ` referenced in the code comments).

### CheckPoint
Cross COCO_Fake : https://drive.google.com/file/d/1_40k3kGJrTvcDKEv492ZwcOF0dgKKhrw/view?usp=drive_link

Cross DFFD : https://drive.google.com/file/d/1bwWQ7eFx7wKzgzXvxi9xn6txejKjDsYC/view?usp=drive_link

### 1. Config-file hyperparameters (`./configs/results_cifake_T.cfg`)

These are the dataset / model / training knobs that are read at runtime via `load_config(...)`.

#### Dataset

| Key | Value | Description |
|-----|-------|-------------|
| `name` | `cifake` | Active dataset used for training/testing. |
| `labels` | `2` | Number of classes (real / fake). Binary, so the head outputs a single logit. |
| `cifake_path` | `/data/.../cifake` | Root path of the CIFAKE dataset. |
| `coco2014_path` / `coco_fake_path` / `dffd_path` | — | Paths of the auxiliary datasets (used for cross-dataset evaluation). |

#### Model

| Key | Value | Description                                                                                           |
|-----|----|-------------------------------------------------------------------------------------------------------|
| `backbone` | `BNext-T` | Binary backbone (the "T" of BNext / CBNext), now instantiated **inside** the merged `BIB` block.      |
| `freeze_backbone` | `false` | The pre-trained backbone is **frozen** if true(only the adapter / fusion / head modules are trained.) |
| `add_fft_channel` | `true` | Append an FFT (frequency-magnitude) channel to the RGB input.                                         |
| `add_lbp_channel` | `true` | Append an LBP (local binary pattern) channel to the RGB input.                                        |
| `add_magnitude_channel` | `false` | Sobel-gradient magnitude channel is **disabled** for this config.                                     |

> With 2 extra channels enabled (FFT + LBP) the network input is **5 channels** (3 RGB + 2). A `Conv2d(5→3, k=3, s=1, p=1)` adapter (or the `HFM` RGB-frequency fusion module when `use_rgbfreq=True`) maps it back to 3 channels before the backbone, after which it is normalized with the ImageNet mean/std.

**FFT (frequency-magnitude) channel**
1. Convert RGB to grayscale: `g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.
2. Apply 2-D FFT and shift the DC component to the center: `F = fftshift(fft2(g))`.
3. Compute log-magnitude: `S = 20 · log(|F| + 1e-9)`, then divide by `255`.

**LBP (Local Binary Pattern) channel**
1. Convert RGB to grayscale: `g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.
2. Compare each pixel with its local neighbors and encode the comparisons as a binary pattern.
3. Store the resulting LBP code as a single channel and divide by `255`.
4. 
#### Train

| Key | Value | Description |
|-----|-------|-------------|
| `batch_size` | `48` | Per-step mini-batch size. |
| `accumulation_batches` | `4` | Gradient-accumulation steps → **effective batch size = 48 × 4 = 192**. |
| `epoch_num` | `10` | Maximum number of epochs (`Trainer.max_epochs`). |
| `resolution` | `224` | Input resolution (224 × 224). |
| `mixed_precision` | `true` | Uses `16-mixed` (AMP) precision. |
| `limit_train_batches` / `limit_val_batches` | `1.0` | Use 100% of the train / val data per epoch. |
| `seed` | `5` | Global seed (also set for `torch` / `numpy` / `random`). |

#### Test

| Key | Value | Description |
|-----|-------|-------------|
| `batch_size` | `128` | Evaluation batch size. |
| `resolution` | `224` | Evaluation resolution. |
| `mixed_precision` | `true` | AMP during inference. |
| `limit_test_batches` | `1.0` | Use 100% of the test data. |
| `weights_path` | `./weights` | Directory of checkpoints to load for testing. |
| `seed` | `5` | Evaluation seed. |

### 2. Fixed hyperparameters (`./main_model.py`)

These are **not** exposed in the `.cfg` file; they are fixed inside `BNext4DFR.configure_optimizers()` and the loss code. The classification block itself (binary backbone + information bottleneck) is the merged `BIB` module defined in `./BIB/bib.py`.

#### Optimizer & LR schedule

| Component | Setting | Source / note |
|-----------|---------|---------------|
| Optimizer | **AdamW** | `optim.AdamW(...)`, default `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=1e-2`. |
| Base learning rate | `5e-5` | Passed in `train.py` (`learning_rate=5e-5`); the `main_model.py` default is `1e-4`. |
| Trained params | adapter/fusion + head (+ backbone only if unfrozen) | Backbone is frozen, so its params are excluded from the optimizer. |
| Warm-up scheduler | **LinearLR**, `warmup_epochs = 3` | `start_factor = lr × 0.025`, `end_factor = 1.0`, `total_iters = 3`. Linearly ramps the LR up to the base LR over the first 3 epochs. |
| Main scheduler | **CosineAnnealingLR** | `T_max = 8 − warmup_epochs = 5`, `eta_min = 1e-6`. Cosine decay after warm-up. |
| Combination | **SequentialLR**, `milestones=[3]` | Switches warm-up → cosine at the end of epoch 3. |
| Gradient clipping | `clip_val = 1.0`, algorithm `"norm"` | Set in the Lightning `Trainer`. |
| Precision | `16-mixed` | AMP, controlled by `mixed_precision`. |

```
Learning-rate schedule (base lr = 5e-5)
lr
 |            ____________
 |          /             \___
 |        /                   \___          warm-up (LinearLR, 3 ep)
 |      /                          \___      then cosine decay (T_max=5)
 |    /                                \__
 |  /                                     \____ eta_min = 1e-6
 +--+----+----+----+----+----+----+----+----+----> epoch
    0    1    2    3    4    5    6    7    8
         |<-- warm-up -->|<------ cosine ------>
                 (3)              (T_max = 5)
```

#### BIB (Binary Information Bottleneck) block

The former separate `VIB` head and `BNext`/`CBNext` backbone are now **merged** into a single `BIB` block (`./BIB/bib.py`), instantiated in `main_model.py` as `self.BIB = BIB(...)`. The information-bottleneck path is enabled via `use_vib=True`. The block is `BIB(y_dim=1, beta=1e-4, num_samples=20, backbone='BNext-T', use_CBNN=True, ...)`, where the bottleneck dimension is computed internally as `dimZ = inplanes // 4`:

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `x_dim` | `inplanes` (backbone feature dim, e.g. 1024) | Input feature dimension (the backbone output now lives inside `BIB`). |
| `dimZ` | `inplanes // 4` | Bottleneck (latent `Z`) dimension. |
| `y_dim` | `1` | Single logit (binary classification). |
| `beta` (`β`) | `1e-4` | Weight of the information-bottleneck term `I(Z;X)`. The block returns `I_ZX_bound × β`. |
| `num_samples` | `20` | Monte-Carlo samples drawn from the encoder posterior. |

### 3. Loss composition

The total loss is assembled in `main_model.BNext4DFR._step(...)`. For the main configuration (`use_vib=True`, `use_rgbfreq=True`) it is:

```
L_total = L_cls  +  c · L_BIB  +  λ · L_HSIC
```

| Term | Formula / code | Coefficient | Description |
|------|----------------|-------------|-------------|
| `L_cls` | `BCEWithLogitsLoss(logits, labels, pos_weight)` | 1.0 | Binary classification loss on the single logit. |
| `pos_weight` | `N_neg / N_pos` | dynamic | Computed from the **training-set class balance** (`negative_samples / positive_samples`) to handle real/fake imbalance. |
| `L_BIB` | `ck[1] = β · I(Z;X)`, `β = 1e-4` | `c` | Information-bottleneck regularizer from the `BIB` block. `c = 1 if use_hsic else 0` — a switch; with `use_hsic=False` this term is currently disabled (`c = 0`). |
| `L_HSIC` | `ck[2] = hsic` | `λ = 1e-3` | The redundancy / dependency (HSIC) term from the RGB-frequency fusion (`HFM`). **This `1e-3` is the `λ` (`#lamda`) referenced in the code comments.** |

Notes:
- The `β = 1e-4` weighting of `I(Z;X)` is applied **inside** the `BIB` block, so `ck[1]` already equals `β · I(Z;X)`; the outer `c` is an additional on/off switch.
- The HSIC weight `λ = 1e-3` is applied **outside**, in `_step`, as `ck[2] * 1e-3`.
- The bottleneck-free path (`use_vib=False`) reduces to `L_cls (+ λ · L_HSIC` when `use_rgbfreq=True`).

### 4. Quick reference summary

| Item | Value |
|------|-------|
| Backbone | BNext-T (frozen) |
| Input | 224×224, 5-channel (RGB + FFT + LBP) → 3-channel adapter/HFM |
| Optimizer | AdamW, lr = 5e-5, wd = 1e-2 |
| LR schedule | LinearLR warm-up (3 ep) → CosineAnnealingLR (T_max = 5, η_min = 1e-6) |
| Batch / effective batch | 48 / 192 (accum = 4) |
| Precision | 16-mixed (AMP) |
| Grad clip | norm, 1.0 |
| Epochs | 10 (config) |
| BIB | merged backbone + bottleneck; dimZ = dim/4, β = 1e-4, 20 samples |
| Loss | `BCE(pos_weight) + c·β·I(Z;X) + λ·HSIC`, λ = 1e-3 |
| Seed | 5 |
