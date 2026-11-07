# Training

## Data

Desert scenes (Duality Falcon). Layout:

```
dataset/
  train/Color_Images/*.png   train/Segmentation/*.png   (2,857 pairs)
  val/Color_Images/*.png     val/Segmentation/*.png     (317 pairs)
```

**Mask decoding (critical).** Masks store `class_id // 256` (high byte of the
16-bit class id), not raw ids — so `remap_mask` maps pixel values, and
colliding classes merge into 6 groups:

| Pixel | Source ids | Class |
|---|---|---|
| 0 | 100, 200 | Dense_Vegetation |
| 1 | 300, 500 | Dry_Vegetation |
| 2 | 550, 600, 700 | Ground_Objects |
| 3 | 800 | Rocks |
| 27 | 7100 | Landscape |
| 39 | 10000 | Sky |
| other | boundary/noise | IGNORE (255) |

The pre-fix script looked for raw ids that never appear, mapping nearly
everything to IGNORE and stalling IoU at 40–50%.

## Train

```bash
python train_fixed.py --data_root ./dataset --arch deeplabv3+ \
    --epochs 50 --batch_size 8 --img_size 512 --lr 6e-5
```

- Default arch: **DeepLabV3+ / ResNet-101** (ImageNet encoder).
  Also supported: `deeplabv3+-r50`, `unet++`, `fpn`, `pspnet`.
- Loss: focal (γ=2.0) + Dice (0.4), class-weighted; 3 warmup epochs.
- Each run writes `runs/<arch>_<timestamp>/`: `best.pth`, `last.pth`,
  `best_iou.json`, `history.json`, `training_curves.png`.
- `train_segformer.py` is the SegFormer-B4 variant (same data layout,
  `--arch segformer-b4` style config).

## Deployed checkpoint

`runs/deeplabv3+_20260403_211706/best.pth` → published as `best.pth` to
[`mk1647/terrain-intelligence`](https://huggingface.co/mk1647/terrain-intelligence).
**mIoU 0.6676** (val):

| Class | IoU |
|---|---|
| Dense_Vegetation | 0.755 |
| Dry_Vegetation | 0.700 |
| Ground_Objects | 0.453 |
| Rocks | 0.464 |
| Landscape | 0.651 |
| Sky | 0.982 |

`MODEL_MIOU` in `pipeline.py` must match this when the checkpoint changes.
Known weak spots: Ground_Objects vs Rocks confusion in shadowed areas —
see `generate_failure_analysis` in `explainer.py` for the LLM diagnosis helper.
