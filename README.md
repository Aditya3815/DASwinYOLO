<div align="center">
  <h1>DA-YOLO: Deformable Attention YOLO</h1>
  <p>
    A novel object detection architecture built on YOLOv5, combining<br>
    <strong>Multi-Scale Deformable Attention (DC3SWT)</strong>
    · <strong>WIoU Geometric Loss</strong>
    · <strong>SE-BiFPN Neck</strong>
    · <strong>Coordinate Attention</strong>
    · <strong>4-Scale Detection (P2–P5)</strong><br>
    Optimised for dense, high-resolution imagery and small-object detection.
  </p>
</div>

---

## Overview

**DA-YOLO** (Deformable Attention YOLO) is a novel small-object detection framework that extends YOLOv5s with five principled architectural improvements, each motivated by the challenges of drone-captured and remote-sensing imagery:

| # | Component | Where | What it solves |
|---|---|---|---|
| 1 | **DC3SWT** | Backbone + PANet neck | Fixed-window attention breaks on arbitrary object locations |
| 2 | **WIoU v1** | Box regression loss | CIoU gives equal weight to easy and hard anchors |
| 3 | **SE-BiFPN** | All 6 BiFPN fusion nodes | Channel redundancy after multi-path feature fusion |
| 4 | **CoordAttMulti** | Detection head input | Spatial position lost in channel-only attention |
| 5 | **4-Scale Head (P2–P5)** | Detection output | 3-scale head misses sub-8-pixel objects |

The core inspiration is *"Swin-Transformer-Based YOLOv5 for Small-Object Detection in Remote Sensing Images"* (Sensors 2023, 23, 3634), extended by replacing fixed-window Swin attention with **Multi-Scale Deformable Attention** (MSDA, arxiv:2010.04159).

---

## Novel Contributions

### 1. DC3SWT — Deformable C3 Swin Transformer Block

> `models/dc3swt.py` · `DC3SWT` and `MSDABlock` classes

The standard Swin Transformer (used in `C3SWT`) partitions feature maps into fixed 8×8 windows and computes attention only within each window. This fails in remote-sensing imagery because:

- Small objects rarely align with pre-defined window boundaries
- Dense clusters span multiple windows simultaneously
- Objects appear at arbitrary locations and scales

**DC3SWT** replaces the entire W-MSA + SW-MSA mechanism with **Multi-Scale Deformable Attention (MSDA)**. For each query location the network *learns where to look* by predicting M×K sampling offsets — allowing attention to span any spatial position, regardless of window boundaries.

```
For each query position q at (x_q, y_q):
  1. Reference point  p_q  = (x_q/W, y_q/H)            ← normalised [0,1] grid
  2. Predict offsets  Δp_{mk} ∈ (−0.5, 0.5)            ← tanh-clamped linear projection
  3. Sample values    v at (p_q + Δp_{mk})              ← F.grid_sample (bilinear)
  4. Attend          out = Σ_m Σ_k A_{mk} · v_{mk}     ← softmax attention weights
  5. Project          output = W_o · out
```

**Pure PyTorch — no custom CUDA extensions** (sampling via `F.grid_sample`).

| Property | C3SWT (baseline) | DC3SWT (proposed) |
|---|---|---|
| Attention type | W-MSA + SW-MSA (fixed windows) | MSDA (learned offsets) |
| Window boundary handling | Split → shifted windows | Implicit via offset prediction |
| Non-square inputs | Requires padding to window multiple | Native — no padding needed |
| Small feature maps | Clamp window to min(H,W) | Clamp n_points to H×W |
| Custom CUDA | No | No |
| Parameters (YOLOv5s) | ~6.39M | ~5.64M (−11%) |

---

### 2. WIoU v1 — Weighted IoU Geometric Loss

> `utils/metrics.py` · `bbox_iou(WIoU=True)` · activated via `loss_type: wiou` in any hyp yaml

Standard CIoU treats all regression targets equally regardless of geometric difficulty. WIoU v1 (AAAI 2023) introduces a **geometric focusing factor** that down-weights easy, well-aligned anchors and concentrates gradient on geometrically hard predictions:

```
g   = exp(ρ² / c²)          where ρ = centre distance, c = enclosing diagonal
WIoU_loss = g × (1 − IoU)
```

When ρ ≈ 0 (already well-centred), g ≈ 1 and the loss is identical to standard IoU loss.  
When ρ is large relative to c (poor centring), g > 1 and the loss is amplified — forcing the network to fix geometrically poor predictions.

This is particularly effective on VisDrone where the anchor set spans a 75× range of object sizes (4px–300px) and many anchors have large positional mismatches.

**Integration:** `bbox_iou()` returns `1 − g*(1−iou)`, so the existing `lbox += (1−iou).mean()` call site in `loss.py` works unchanged. Zero risk of call-site breakage.

---

### 3. SE Channel Attention in BiFPN

> `models/bifpn.py` · `SEBlock` class · `BiFPNLayer(use_se=True)`

After BiFPN weighted fusion, different channels carry features from paths with very different spatial histories (backbone skip, top-down, bottom-up). Standard convolution treats all output channels equally, wasting capacity on redundant or suppressed paths.

**SEBlock** (Squeeze-and-Excitation, Hu et al. CVPR 2018) applies lightweight channel recalibration after every fusion convolution:

```
scale = Sigmoid( Linear(SiLU(Linear(AvgPool2d(x)))) )
output = x × scale
```

Placed at **all 6 BiFPN fusion nodes** (p4_td, p3_td, p2_out, p3_out, p4_out, p5_out) for systematic coverage. Total cost: ~49K parameters at C=256 (< 1% of model). Set `use_se=False` to use `nn.Identity` passthrough at zero cost.

---

### 4. CoordAttMulti — Coordinate Attention

> `models/coord_attention.py` · `CoordAttMulti` class

Standard channel attention (SE) pools spatial information entirely, losing positional context. **Coordinate Attention** (Hou et al. CVPR 2021) decomposes spatial pooling into H-axis and W-axis directional pooling, preserving directional position information:

```
x_h = AvgPool(x, kernel=(1,W))  → (B,C,H,1)
x_w = AvgPool(x, kernel=(H,1))  → (B,C,1,W)
```

`CoordAttMulti` applies one CA layer per detection scale (P2, P3, P4, P5), injecting spatial position information immediately before the detection head. This is **orthogonal to SE** — SE recalibrates channels after fusion; CoordAtt encodes spatial position before detection.

---

### 5. 4-Scale Detection Head (P2–P5)

Standard YOLOv5 uses 3 detection scales (P3/P4/P5, strides 8/16/32). The minimum detectable object at stride 8 and 640px input is approximately 8×8 pixels. For VisDrone where objects can be as small as 2–4 pixels, this is insufficient.

DA-YOLO adds a **P2 ultra-high-resolution head** at stride 4:

| Head | Stride | Object size range | VisDrone context |
|---|---|---|---|
| P2 | 4 | 2–16 px | Pedestrians, bicycles at distance |
| P3 | 8 | 16–48 px | Small vehicles, people |
| P4 | 16 | 48–128 px | Vehicles, buses |
| P5 | 32 | 128+ px | Large vehicles, buses |

DCNv2 is placed **exclusively at the P3 output BiFPN node** because P3 is the only node that simultaneously receives three distinct spatial paths (backbone skip, top-down from P4/P5, bottom-up from P2) — the highest geometric misalignment junction. At P4/P5 the benefit diminishes while the +150K parameter cost is constant.

---

## Architecture Diagram

```
Input Image (any resolution)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  YOLOv5s-CSP Backbone                                           │
│  Conv → C3 → Conv → C3 → Conv → C3 → Conv → DC3SWT → SPPF     │
│   P1/2   P2/4        P3/8        P4/16        P5/32             │
└──────────┬──────────┬─────────────┬────────────────────────────┘
           │ P2       │ P3          │ P4              │ P5 (SPPF)
           │          │             │                  │
┌──────────▼──────────▼─────────────▼──────────────────▼─────────┐
│  PANet Top-Down Pathway (DC3SWT at every level)                 │
│  P5 → upsample+concat(P4) → DC3SWT → upsample+concat(P3)      │
│     → DC3SWT → upsample+concat(P2) → DC3SWT                   │
└──────────┬──────────┬─────────────┬────────────────────────────┘
           │ P2_td    │ P3_td       │ P4_td           │ P5
           │          │             │                  │
┌──────────▼──────────▼─────────────▼──────────────────▼─────────┐
│  BiFPN Cross-Scale Fusion (SE channel attention, DCNv2 @ P3)   │
│  6 weighted fusion nodes with SEBlock recalibration            │
│  DeformConv at P3_out for 3-way junction alignment             │
└──────────┬──────────┬─────────────┬────────────────────────────┘
           │ P2_out   │ P3_out      │ P4_out          │ P5_out
           │          │             │                  │
┌──────────▼──────────▼─────────────▼──────────────────▼─────────┐
│  CoordAttMulti (per-scale Coordinate Attention)                 │
└──────────┬──────────┬─────────────┬────────────────────────────┘
           │          │             │                  │
        Head P2    Head P3       Head P4            Head P5
       (tiny)     (small)       (medium)            (large)
```

---

## Installation

```bash
git clone <this-repo>
cd DASwin-YOLO

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
# Key packages: torch>=1.12, torchvision>=0.13, timm>=0.9.0

# Verify all components (no pytest required)
venv/bin/python3 tests/test_components.py
# Expected: 36/36 passed — All tests passed! ✅
```

---

## Dataset Setup

### VisDrone2019-DET

VisDrone is a drone-captured aerial dataset with 10 object classes. The detection split contains 6,471 training images, 548 validation images, and 1,610 test-dev images at 1920×1080 resolution.

**Step 1 — Download the dataset**

The dataset is hosted on Google Drive via the official VisDrone GitHub at https://github.com/VisDrone/VisDrone-Dataset. Use `gdown` to download directly from the command line:

```bash
pip install gdown

mkdir -p /data/VisDrone && cd /data/VisDrone

# VisDrone2019-DET — train, val, test-dev
gdown --fuzzy "https://drive.google.com/file/d/1a2oHjcEcwXP8oUF9gaL7xyqMYwkigTe0" -O VisDrone2019-DET-train.zip
gdown --fuzzy "https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59" -O VisDrone2019-DET-val.zip
gdown --fuzzy "https://drive.google.com/file/d/1PFdW_VFSCfZ_sTSZAGjZggiC9qfpfMm3" -O VisDrone2019-DET-test-dev.zip

unzip VisDrone2019-DET-train.zip
unzip VisDrone2019-DET-val.zip
unzip VisDrone2019-DET-test-dev.zip
```

If `gdown` fails (Google Drive quota), download the three zip files manually from the VisDrone GitHub and place them under `/data/VisDrone/`.

After extraction the tree should be:
```
/data/VisDrone/
  VisDrone2019-DET-train/
    images/       *.jpg
    annotations/  *.txt
  VisDrone2019-DET-val/
    images/
    annotations/
  VisDrone2019-DET-test-dev/
    images/
    annotations/  (may be empty)
```

**Step 2 — Convert to YOLO format**

```bash
cd /path/to/DASwin-YOLO

venv/bin/python3 utils/visdrone_converter.py \
    --root /data/VisDrone \
    --out  /data/VisDrone/yolo
```

Output tree under `/data/VisDrone/yolo/`:
```
images/
  train/    *.jpg  (6471 images)
  val/      *.jpg  (548 images)
  test-dev/ *.jpg  (1610 images)
labels/
  train/    *.txt  (YOLO format)
  val/      *.txt
  test-dev/ *.txt  (empty if no annotations)
```

Category mapping: VisDrone class 0 (ignored) and 11 (others) are skipped. Classes 1–10 map to YOLO IDs 0–9 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor).

**Step 3 — Update dataset yaml**

Edit `data/visdrone.yaml` and set the `path` field:
```yaml
path: /data/VisDrone/yolo   # ← set to your --out path
```

**Step 4 — Re-cluster anchors (strongly recommended)**

VisDrone objects span 4px–300px — very different from the COCO default anchors. Re-clustering with CIOU distance gives a significant mAP boost:

```bash
venv/bin/python3 utils/ciou_kmeans.py \
    --label-dir /data/VisDrone/yolo/labels/train \
    --img-size 1280 \
    --n-clusters 12

# Paste the printed anchors into models/da_yolo.yaml anchors section
```

---

### DOTA v1.0 HBB (Horizontal Bounding Boxes)

DOTA is a large-scale dataset for object detection in aerial images (2806 images, 15 categories, up to 4000×4000 pixels). DA-YOLO currently supports the HBB (axis-aligned box) variant.

**Step 1 — Download**

Register and download from the official DOTA page: https://captain-whu.github.io/DOTA/dataset.html

Download `DOTA-v1.0_train.zip` and `DOTA-v1.0_val.zip`.

**Step 2 — Tile the images**

DOTA images are too large for direct training (up to 4000×4000). They must be tiled into 1024×1024 patches with 200px overlap:

```bash
venv/bin/python3 utils/dota_tiler.py \
    --root   /data/DOTA \
    --out    /data/DOTA/yolo \
    --size   1024 \
    --stride 824

# Note: utils/dota_tiler.py is planned — see roadmap below
```

**Step 3 — Train**

```bash
venv/bin/python3 train_da_yolo.py \
    --data data/dota.yaml \
    --mode scratch \
    --img-size 1024 \
    --batch-size 4
```

> DOTA tiling script and `data/dota.yaml` are on the roadmap. Contributions welcome.

---

### DIOR-R (Oriented Bounding Boxes)

DIOR-R is a remote sensing detection dataset with 20 categories and oriented annotations (OBB). OBB support requires an angle regression head extension and is planned as a future addition to DA-YOLO.

Current status: DIOR-R HBB (converting OBB to HBB by taking the bounding box of the polygon) can be trained immediately. Full OBB head support is on the roadmap.

---

## Training

### Fine-tune mode (pretrained backbone, recommended starting point)

```bash
# VisDrone — fine-tune, lr=1e-6, 100 epochs, cosine LR
venv/bin/python3 train_da_yolo.py \
    --data data/visdrone.yaml \
    --mode finetune \
    --img-size 1280

# Override batch size and epochs
venv/bin/python3 train_da_yolo.py \
    --data data/visdrone.yaml \
    --mode finetune \
    --img-size 1280 \
    --batch-size 8 \
    --epochs 150
```

### From-scratch mode (no pretrained weights)

```bash
# VisDrone — from-scratch, lr=1e-4, 150 epochs, WIoU loss
venv/bin/python3 train_da_yolo.py \
    --data data/visdrone.yaml \
    --mode scratch \
    --img-size 1280

# DOTA
venv/bin/python3 train_da_yolo.py \
    --data data/dota.yaml \
    --mode scratch \
    --img-size 1024 \
    --batch-size 4
```

### Resume from checkpoint

```bash
venv/bin/python3 train_da_yolo.py \
    --data data/visdrone.yaml \
    --resume runs/da_yolo/exp/weights/last.pt
```

### Long-running training (recommended: tmux)

```bash
tmux new -s da_yolo
venv/bin/python3 train_da_yolo.py --data data/visdrone.yaml --mode scratch --img-size 1280
# Detach: Ctrl+B, D — re-attach: tmux attach -t da_yolo
```

### Training defaults summary

| Setting | Fine-tune | From-scratch |
|---|---|---|
| Initial LR | 1e-6 | 1e-4 |
| Epochs | 100 | 150 |
| Early stop patience | 30 | 80 |
| Label smoothing | 0.1 | 0.0 |
| Close-mosaic (last N epochs) | 10 | 20 |
| Optimizer | AdamW | AdamW |
| LR scheduler | Cosine | Cosine |
| Batch size | 4 | 4 |
| Image size | 1024 | 1024 |
| Loss type | WIoU v1 | WIoU v1 |

### Training hardening (all active by default via `train_da_yolo.py`)

| Feature | Detail |
|---|---|
| **AdamW optimizer** | Required for Transformer blocks — SGD causes gradient spikes |
| **ValLoss early stopping** | Stops on validation loss plateau, not mAP (more stable on dense datasets) |
| **NaN loss guard** | Skips non-finite batches to protect optimizer momentum buffers |
| **Gradient clip** | `max_norm=1.0` — standard for Transformer-based models |
| **Close-mosaic** | Disables heavy augmentation for final N epochs (YOLOv8 trick) |
| **Cosine LR + DropPath=0.1** | Smooth convergence for Swin Transformer-based training |
| **WIoU v1 loss** | Geometric focus factor down-weights easy anchors |
| **SE channel attention** | Recalibrates BiFPN fusion output channels |

---

## Evaluation

```bash
# Validate on VisDrone val split
venv/bin/python3 val.py \
    --data data/visdrone.yaml \
    --weights runs/da_yolo/exp/weights/best.pt \
    --img-size 1280 \
    --batch-size 8

# Generate COCO-format JSON for submission
venv/bin/python3 val.py \
    --data data/visdrone.yaml \
    --weights runs/da_yolo/exp/weights/best.pt \
    --img-size 1280 \
    --save-json
```

---

## Inference

```bash
# Inference on a single image
venv/bin/python3 detect.py \
    --weights runs/da_yolo/exp/weights/best.pt \
    --source  /path/to/image.jpg \
    --img-size 1280 \
    --conf-thres 0.25 \
    --iou-thres 0.45

# Inference on a directory
venv/bin/python3 detect.py \
    --weights runs/da_yolo/exp/weights/best.pt \
    --source  /path/to/images/ \
    --img-size 1280

# Inference on video
venv/bin/python3 detect.py \
    --weights runs/da_yolo/exp/weights/best.pt \
    --source  /path/to/video.mp4 \
    --img-size 1280

# SLD (Stained-Laminate Defect) inference — uses dedicated inference code
# See inference_code/ directory for SLD-specific scripts
```

---

## Export

```bash
# Export to ONNX (for deployment)
venv/bin/python3 export.py \
    --weights runs/da_yolo/exp/weights/best.pt \
    --include onnx \
    --img-size 1280 \
    --batch-size 1

# Export to TorchScript
venv/bin/python3 export.py \
    --weights runs/da_yolo/exp/weights/best.pt \
    --include torchscript
```

---

## Ablation Table

All variants use the same `models/da_yolo.yaml` config, controlled by hyp yaml flags and architecture arguments.

| Variant | Backbone attention | BiFPN SE | Loss | Params | Notes |
|---|---|---|---|---|---|
| YOLOv5s baseline | C3 bottleneck | — | CIoU | ~7.0M | Reference |
| + C3SWT neck | W-MSA + SW-MSA | — | CIoU | ~6.39M | Fixed-window Swin |
| + DC3SWT | **MSDA (learned offsets)** | — | CIoU | ~5.64M | This work — deformable |
| + WIoU | MSDA | — | **WIoU v1** | ~5.64M | Geometric loss focus |
| + SE-BiFPN | MSDA | **SEBlock × 6** | WIoU | ~5.69M | Channel recalibration |
| **DA-YOLO (full)** | **MSDA** | **SEBlock × 6** | **WIoU v1** | **~5.69M** | **All components** |

> fp16 inference halves memory: DA-YOLO fits in ~10.7 MB at fp16.  
> All counts at `nc=80`, `width_multiple=0.50`.  
> Actual mAP values are dataset-dependent — train and evaluate on your dataset.

---

## Generate Dataset-Specific Anchors

The default anchors are scaled for COCO objects. For any new dataset (especially VisDrone with its tiny objects) re-clustering dramatically improves recall:

```bash
venv/bin/python3 utils/ciou_kmeans.py \
    --label-dir /data/VisDrone/yolo/labels/train \
    --img-size 1280 \
    --n-clusters 12

# Paste the printed anchors block into models/da_yolo.yaml, replacing the anchors: section
```

---

## Custom Dataset

```bash
# Copy the dataset template
cp data/da_yolo_template.yaml data/my_dataset.yaml

# Edit my_dataset.yaml:
#   path:  /path/to/dataset
#   train: images/train
#   val:   images/val
#   nc:    <number of classes>
#   names: { 0: class_a, 1: class_b, ... }

# Regenerate anchors for your object size distribution
venv/bin/python3 utils/ciou_kmeans.py \
    --label-dir /path/to/dataset/labels/train \
    --img-size 1024 \
    --n-clusters 12

# Train
venv/bin/python3 train_da_yolo.py \
    --data data/my_dataset.yaml \
    --mode scratch
```

---

## Component Locations

| File | Purpose |
|---|---|
| `models/da_yolo.yaml` | **DA-YOLO architecture config (primary)** |
| `models/dc3swt.py` | `DC3SWT` + `MSDABlock` — pure-PyTorch deformable attention |
| `models/swin_block.py` | `C3SWT` — CSP + Swin window attention (ablation baseline) |
| `models/bifpn.py` | `BiFPNLayer` — BiFPN with DCNv2 @ P3 and SEBlock at all 6 nodes |
| `models/coord_attention.py` | `CoordAtt` + `CoordAttMulti` — Coordinate Attention |
| `utils/metrics.py` | `bbox_iou()` with WIoU v1 flag (`WIoU=True`) |
| `utils/loss.py` | `ComputeLoss` — activates WIoU via `loss_type: wiou` in hyp yaml |
| `utils/ciou_kmeans.py` | CIOU anchor clustering for dataset-specific anchors |
| `utils/visdrone_converter.py` | VisDrone2019-DET → YOLO format converter |
| `train_da_yolo.py` | **Primary training launcher** (wraps train.py with best-practice defaults) |
| `train.py` | Core YOLOv5 training loop (used internally by train_da_yolo.py) |
| `val.py` | Evaluation script |
| `detect.py` | Inference script |
| `export.py` | Export to ONNX / TorchScript |
| `data/da_yolo_template.yaml` | Dataset template — copy and fill for your data |
| `data/visdrone.yaml` | VisDrone2019-DET dataset config |
| `data/hyps/hyp.da-yolo.yaml` | Hyperparameters for fine-tuning (lr=1e-6, WIoU) |
| `data/hyps/hyp.visdrone.yaml` | Hyperparameters tuned for VisDrone (lr=1e-4, focal loss, copy-paste) |
| `tests/test_components.py` | 36-test component suite (standalone, no pytest required) |
| `inference_code/` | SLD (Stained-Laminate Defect) inference scripts |
| `inference_dataset/` | SLD inference dataset |
| `data/daswinyolo_sld.yaml` | SLD dataset config |

---

## Hyperparameter Reference

### `data/hyps/hyp.da-yolo.yaml` (fine-tune, generic)

| Key | Value | Rationale |
|---|---|---|
| `lr0` | 1e-6 | Conservative: avoids catastrophic forgetting when fine-tuning pretrained backbone |
| `lrf` | 0.01 | Cosine anneals to 1e-8 |
| `weight_decay` | 0.05 | AdamW standard |
| `loss_type` | wiou | WIoU v1 geometric loss |
| `label_smoothing` | 0.1 | Prevents overconfident predictions |

### `data/hyps/hyp.visdrone.yaml` (VisDrone from-scratch)

| Key | Value | Rationale |
|---|---|---|
| `lr0` | 1e-4 | Higher LR for from-scratch; VisDrone has no good pretrained init |
| `obj` | 1.5 | Up-weighted objectness: VisDrone has many tiny 2–16px objects at P2 head |
| `fl_gamma` | 1.5 | Focal loss active: severe car >> pedestrian >> bus class imbalance |
| `copy_paste` | 0.3 | Aggressive copy-paste: pads sparse background regions with objects |
| `scale` | 0.5 | Scale jitter ±50%: objects span 4px–300px range |
| `loss_type` | wiou | WIoU v1 geometric loss |
| `label_smoothing` | 0.0 | Keep positives sharp for densely occluded objects |
| `degrees` | 0.0 | No rotation: aerial view is fixed-orientation |

---

## Roadmap

- [ ] DOTA tiling script (`utils/dota_tiler.py`) and `data/dota.yaml`
- [ ] DIOR-R OBB head extension (angle regression branch)
- [ ] VisDrone benchmark results table (mAP@0.5, mAP@0.5:0.95)
- [ ] Mosaic + copy-paste integration for DOTA tiled training
- [ ] fp16 / TensorRT export validation on VisDrone

---

## Design Decisions

### Why DCNv2 at P3 only — not all BiFPN nodes?

P3_out is the only BiFPN node that simultaneously receives **three distinct spatial paths**:

```
P3_out = conv(  w₁·P3_original          ← lateral skip from backbone
              + w₂·P3_td                ← top-down signal from P4/P5 context
              + w₃·P2_out_downsampled   ← bottom-up signal from high-res P2  )
```

These tensors originate from different receptive fields and sampling geometries. DCNv2's learned offsets correct this misalignment adaptively. P4/P5 have proportionally smaller misalignment at lower resolution. Each `DeformConvBlock` adds ~150K params — placing it at P3 only gives the highest benefit-to-cost ratio.

### Why single-level MSDA in DC3SWT?

Deformable DETR uses multi-level MSDA (L=4 feature levels) because the decoder needs global cross-scale context. DC3SWT operates within a single pyramid level because:
- The surrounding BiFPN already handles cross-scale aggregation
- Multi-level sampling would multiply memory by L=4 inside each block
- "Multi-scale" in DC3SWT refers to M heads × K sampling points within a single level, not cross-level attention

### Why SE in BiFPN rather than CoordAtt?

SE and CoordAtt are complementary. SE handles **channel redundancy** after feature fusion — deciding which channels to emphasise. CoordAtt encodes **spatial position** for the detection head. Both are needed; placing SE in BiFPN and CoordAtt at head input avoids duplication.

---

## References

- **Paper (inspiration)**: Gong, H. et al. *Swin-Transformer-Based YOLOv5 for Small-Object Detection in Remote Sensing Images.* Sensors 2023, 23, 3634. [DOI:10.3390/s23073634](https://doi.org/10.3390/s23073634)
- **Deformable DETR / MSDA**: Zhu, X. et al. *Deformable DETR: Deformable Transformers for End-to-End Object Detection.* ICLR 2021. [arxiv:2010.04159](https://arxiv.org/abs/2010.04159)
- **WIoU v1**: Tong, Z. et al. *Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism.* AAAI 2023. [arxiv:2301.10051](https://arxiv.org/abs/2301.10051)
- **Squeeze-and-Excitation**: Hu, J. et al. *Squeeze-and-Excitation Networks.* CVPR 2018. [arxiv:1709.01507](https://arxiv.org/abs/1709.01507)
- **Swin Transformer**: Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. [arxiv:2103.14030](https://arxiv.org/abs/2103.14030)
- **BiFPN / EfficientDet**: Tan, M. et al. *EfficientDet: Scalable and Efficient Object Detection.* CVPR 2020. [arxiv:1911.09070](https://arxiv.org/abs/1911.09070)
- **Coordinate Attention**: Hou, Q. et al. *Coordinate Attention for Efficient Mobile Network Design.* CVPR 2021. [arxiv:2103.02907](https://arxiv.org/abs/2103.02907)
- **DCNv2**: Zhu, X. et al. *Deformable ConvNets v2: More Deformable, Better Results.* CVPR 2019. [arxiv:1811.11168](https://arxiv.org/abs/1811.11168)
- **VisDrone**: Zhu, P. et al. *VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results.* ICCVW 2019. [github.com/VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
- **YOLOv5**: Jocher, G. et al. Ultralytics YOLOv5. [github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
