"""
FIXED Semantic Segmentation Training Script
============================================
ROOT CAUSE OF LOW IoU (40-50%):
  The Duality Falcon masks store class_id // 256 as the pixel value (high byte of 16-bit class ID).
  The original remap_mask() looked for raw class IDs (100, 200, 300...) which NEVER appear
  in the mask pixels, so almost everything was mapped to IGNORE_INDEX=255.
  The model was learning from nearly empty supervision -> stuck at 40-50% IoU.

MASK ENCODING (discovered by pixel analysis):
  pixel=0  -> Trees (ID=100)  OR LushBushes (ID=200)  [100//256=0, 200//256=0]
  pixel=1  -> DryGrass (ID=300) OR DryBushes (ID=500) [300//256=1, 500//256=1]
  pixel=2  -> GroundClutter(550) OR Flowers(600) OR Logs(700) [all//256=2]
  pixel=3  -> Rocks (ID=800)      [800//256=3]
  pixel=27 -> Landscape (ID=7100) [7100//256=27]
  pixel=39 -> Sky (ID=10000)      [10000//256=39]
  other    -> boundary/noise -> IGNORE

Since colliding classes (Trees+LushBushes, DryGrass+DryBushes, etc.) cannot be
distinguished from the high byte alone, we merge them into 6 semantic groups.
This is the correct interpretation of the available mask data.

Architecture: DeepLabV3+ with ResNet-101 (MPS-compatible, no view() issues)
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — CORRECTED MASK DECODING
# ─────────────────────────────────────────────

# Maps raw class ID (from uint16 mask) -> contiguous class index
PIXEL_TO_CLASS = {
    100:  0,    # Dense Vegetation: Trees
    200:  0,    # Dense Vegetation: LushBushes
    300:  1,    # Dry Vegetation: DryGrass
    500:  1,    # Dry Vegetation: DryBushes
    550:  2,    # Ground Objects: GroundClutter
    600:  2,    # Ground Objects: Flowers
    700:  2,    # Ground Objects: Logs
    800:  3,    # Rocks
    7100: 4,    # Landscape
    10000: 5,   # Sky
}

CLASS_NAMES = [
    "Dense_Vegetation",  # Trees + LushBushes
    "Dry_Vegetation",    # DryGrass + DryBushes
    "Ground_Objects",    # GroundClutter + Flowers + Logs
    "Rocks",
    "Landscape",
    "Sky",
]
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255


# ─────────────────────────────────────────────
# MASK REMAPPING — THE CRITICAL FIX
# ─────────────────────────────────────────────

def remap_mask(mask_np: np.ndarray) -> np.ndarray:
    """
    Convert raw class IDs to contiguous class indices.
    
    Masks are stored as uint16 with raw class IDs: 100, 200, 300, 500, 550, 800, 7100, 10000
    Unrecognized IDs -> IGNORE_INDEX (boundary/noise)
    
    Args:
        mask_np: HxW uint16 array with raw class IDs
    Returns:
        HxW uint8 array with class indices 0-5 or IGNORE_INDEX=255
    """
    # Handle multi-channel (use first channel)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    out = np.full(mask_np.shape, IGNORE_INDEX, dtype=np.uint8)
    for class_id, class_idx in PIXEL_TO_CLASS.items():
        out[mask_np == class_id] = class_idx
    return out


# ─────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────

def get_train_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.RandomShadow(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0, always_apply=False),
        ToTensorV2(transpose_mask=True),
    ], is_check_shapes=False)


def get_val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0, always_apply=False),
        ToTensorV2(transpose_mask=True),
    ], is_check_shapes=False)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class DesertSegDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 512):
        self.root = Path(root)
        self.split = split
        self.transform = (get_train_transforms(img_size) if split == "train"
                          else get_val_transforms(img_size))

        img_subfolders  = ["Color_Images", "rgb", "images", "image", "imgs",
                           "color", "img", "color_images", "ColorImages", "RGB"]
        mask_subfolders = ["Segmentation", "segmentation", "masks", "mask", "labels"]
        split_variants  = [split, split.capitalize(), split.upper()]

        img_dir = mask_dir = None
        for sv in split_variants:
            for isf in img_subfolders:
                c = self.root / sv / isf
                if c.exists():
                    img_dir = c
                    break
            if img_dir:
                break
        if img_dir is None:
            for isf in img_subfolders:
                c = self.root / isf
                if c.exists():
                    img_dir = c
                    break

        for sv in split_variants:
            for msf in mask_subfolders:
                c = self.root / sv / msf
                if c.exists():
                    mask_dir = c
                    break
            if mask_dir:
                break
        if mask_dir is None:
            for msf in mask_subfolders:
                c = self.root / msf
                if c.exists():
                    mask_dir = c
                    break

        print(f"\n[Dataset split={split!r}]")
        print(f"  img_dir  : {img_dir}")
        print(f"  mask_dir : {mask_dir}")

        if img_dir is None or not img_dir.exists():
            raise FileNotFoundError(f"Image dir not found under {self.root}")
        if mask_dir is None or not mask_dir.exists():
            raise FileNotFoundError(f"Mask dir not found under {self.root}")

        exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
        self.images = sorted([p for e in exts for p in img_dir.glob(e)])
        self.masks  = sorted([p for e in exts for p in mask_dir.glob(e)])

        print(f"  images   : {len(self.images)}")
        print(f"  masks    : {len(self.masks)}")

        assert len(self.images) > 0, f"No images found in {img_dir}"
        assert len(self.images) == len(self.masks), (
            f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))
        mask = remap_mask(mask)  # FIXED: correct pixel->class mapping
        aug  = self.transform(image=img, mask=mask)
        # Ensure mask is long type and verify it contains valid class indices
        mask_out = aug["mask"].long()
        if mask_out.ndim == 3:  # Remove channel dim if added by ToTensorV2
            mask_out = mask_out[0]
        return aug["image"].float(), mask_out.contiguous()


# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────

def compute_class_weights(dataset: DesertSegDataset) -> torch.Tensor:
    print("Computing class weights...")
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, mask in dataset:
        m = mask.numpy()
        for c in range(NUM_CLASSES):
            counts[c] += (m == c).sum()
    print("  Pixel counts per class:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
        print(f"    [{i}] {name}: {int(count):,}")
    counts   = np.where(counts == 0, 1, counts)
    weights  = 1.0 / counts
    weights /= weights.sum()
    weights *= NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────
# LOSS — MPS-SAFE (no view() on 4D tensors)
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, ignore_index: int = 255):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        # Flatten to (N, C) — avoids MPS .view() incompatibility on 4D tensors
        logits_flat  = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        valid        = targets_flat != self.ignore_index
        logits_flat  = logits_flat[valid]
        targets_flat = targets_flat[valid]

        if targets_flat.numel() == 0:
            return logits.sum() * 0.0

        log_probs = torch.log_softmax(logits_flat, dim=1)
        probs     = torch.exp(log_probs)

        idx    = targets_flat.unsqueeze(1)
        log_pt = log_probs.gather(1, idx).squeeze(1)
        pt     = probs.gather(1, idx).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.weight is not None:
            w    = self.weight.to(logits.device)
            loss = loss * w[targets_flat]

        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs        = torch.softmax(logits, dim=1)
        probs_flat   = probs.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        valid        = targets_flat != self.ignore_index
        probs_flat   = probs_flat[valid]
        targets_flat = targets_flat[valid]

        if targets_flat.numel() == 0:
            return logits.sum() * 0.0

        one_hot = torch.zeros_like(probs_flat)
        one_hot.scatter_(1, targets_flat.unsqueeze(1), 1.0)

        intersection   = (probs_flat * one_hot).sum(dim=0)
        union          = probs_flat.sum(dim=0) + one_hot.sum(dim=0)
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice_per_class).mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight=None, focal_gamma: float = 2.0,
                 dice_w: float = 0.4, focal_w: float = 0.6):
        super().__init__()
        self.focal   = FocalLoss(gamma=focal_gamma, weight=weight)
        self.dice    = DiceLoss()
        self.dice_w  = dice_w
        self.focal_w = focal_w

    def forward(self, logits, targets):
        return self.focal_w * self.focal(logits, targets) + \
               self.dice_w  * self.dice(logits, targets)


# ─────────────────────────────────────────────
# MODEL — DeepLabV3+ (MPS-compatible, no SegFormer view() bug)
# ─────────────────────────────────────────────

def get_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "deeplabv3+":
        return smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    elif arch == "deeplabv3+-r50":
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    elif arch == "unet++":
        return smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    elif arch == "fpn":
        return smp.FPN(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    elif arch == "pspnet":
        return smp.PSPNet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown arch: {arch}. Choose from: deeplabv3+, deeplabv3+-r50, unet++, fpn, pspnet")


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class IoUMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union        = np.zeros(self.num_classes)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds   = preds.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        valid   = targets != self.ignore_index
        for c in range(self.num_classes):
            pred_c = (preds == c) & valid
            tgt_c  = (targets == c) & valid
            self.intersection[c] += (pred_c & tgt_c).sum()
            self.union[c]        += (pred_c | tgt_c).sum()

    def compute(self) -> dict:
        iou_per_class = np.where(
            self.union > 0,
            self.intersection / (self.union + 1e-8),
            np.nan
        )
        valid_ious = iou_per_class[~np.isnan(iou_per_class)]
        miou = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0
        return {
            "mIoU": miou,
            "per_class": {CLASS_NAMES[i]: float(iou_per_class[i])
                          for i in range(self.num_classes)},
        }


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Device selection — DeepLabV3+ works fine on MPS
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Device: {self.device}")

        self.run_dir = Path(cfg["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.train_ds = DesertSegDataset(cfg["data_root"], "train", cfg["img_size"])
        self.val_ds   = DesertSegDataset(cfg["data_root"], "val",   cfg["img_size"])
        self.train_loader = DataLoader(
            self.train_ds, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=cfg.get("num_workers", 4),
            pin_memory=False, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=cfg.get("num_workers", 4),
            pin_memory=False,
        )

        # Model
        self.model = get_model(cfg["arch"], NUM_CLASSES).to(self.device)
        print(f"Model: {cfg['arch']} | Classes: {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")

        # Loss
        class_weights = None
        if cfg.get("use_class_weights", True):
            class_weights = compute_class_weights(self.train_ds)
        self.criterion = CombinedLoss(
            weight=class_weights,
            focal_gamma=cfg.get("focal_gamma", 2.0),
            dice_w=cfg.get("dice_weight", 0.4),
            focal_w=cfg.get("focal_weight", 0.6),
        )

        # Optimizer & scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 1e-4)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg["epochs"] - cfg.get("warmup_epochs", 3),
            eta_min=1e-6
        )

        self.best_miou = 0.0
        self.history   = {"train_loss": [], "val_loss": [], "val_miou": []}

    def _warmup_lr(self, epoch: int):
        warmup = self.cfg.get("warmup_epochs", 3)
        if epoch < warmup:
            factor = (epoch + 1) / warmup
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg["lr"] * factor

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self._warmup_lr(epoch)
        total_loss = 0.0
        for images, masks in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            logits = F.interpolate(logits, size=images.shape[-2:],
                                   mode="bilinear", align_corners=False)
            loss = self.criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        metric     = IoUMetric(NUM_CLASSES)
        for images, masks in self.val_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            logits = self.model(images)
            logits = F.interpolate(logits, size=images.shape[-2:],
                                   mode="bilinear", align_corners=False)
            total_loss += self.criterion(logits, masks).item()
            metric.update(logits, masks)
        results = metric.compute()
        return total_loss / len(self.val_loader), results["mIoU"], results

    def save_checkpoint(self, epoch: int, miou: float, tag: str = "best"):
        torch.save({
            "epoch": epoch, "miou": miou,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "class_names": CLASS_NAMES,
            "pixel_to_class": PIXEL_TO_CLASS,
        }, self.run_dir / f"{tag}.pth")

    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history["train_loss"], label="Train Loss")
        axes[0].plot(self.history["val_loss"],   label="Val Loss")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
        axes[1].plot(self.history["val_miou"], label="Val mIoU", color="green")
        axes[1].set_title("mIoU"); axes[1].legend(); axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.run_dir / "training_curves.png", dpi=150)
        plt.close()
        print(f"  Saved training curves -> {self.run_dir}/training_curves.png")

    def fit(self) -> float:
        epochs = self.cfg["epochs"]
        warmup = self.cfg.get("warmup_epochs", 3)
        print(f"\n{'='*65}")
        print(f"Training | Device: {self.device} | Epochs: {epochs}")
        print(f"Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")
        print(f"{'='*65}\n")

        for epoch in range(epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss, val_miou, iou_detail = self.val_epoch()

            if epoch >= warmup:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_miou"].append(val_miou)

            elapsed = time.time() - t0
            print(f"Epoch [{epoch+1:03d}/{epochs}] "
                  f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} "
                  f"mIoU={val_miou:.4f} ({elapsed:.1f}s)")

            # Print per-class IoU every 5 epochs
            if (epoch + 1) % 5 == 0:
                for cls_name, iou in iou_detail["per_class"].items():
                    print(f"    {cls_name}: {iou:.4f}")

            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint(epoch, val_miou, "best")
                with open(self.run_dir / "best_iou.json", "w") as f:
                    json.dump(iou_detail, f, indent=2)
                print(f"  ✓ New best mIoU: {val_miou:.4f}")

            self.save_checkpoint(epoch, val_miou, "last")

        self.plot_history()
        json.dump(self.history, open(self.run_dir / "history.json", "w"), indent=2)
        print(f"\nBest mIoU: {self.best_miou:.4f}")

        # Print final class mapping summary
        print("\nClass mapping used:")
        for pixel_val, class_idx in PIXEL_TO_CLASS.items():
            print(f"  mask pixel {pixel_val:2d} -> [{class_idx}] {CLASS_NAMES[class_idx]}")
        return self.best_miou


# ─────────────────────────────────────────────
# INFERENCE / VISUALIZATION
# ─────────────────────────────────────────────

PALETTE = [
    [34,  139,  34],   # Dense_Vegetation  -> forest green
    [210, 180, 140],   # Dry_Vegetation    -> tan
    [139,  90,  43],   # Ground_Objects    -> brown
    [128, 128, 128],   # Rocks             -> gray
    [205, 133,  63],   # Landscape         -> sandy brown
    [135, 206, 235],   # Sky               -> sky blue
]

def visualize_prediction(model, image_path: str, device, save_path: str,
                         img_size: int = 512):
    """Run inference on a single image and save colorized segmentation."""
    tf = get_val_transforms(img_size)
    img_np = np.array(Image.open(image_path).convert("RGB"))
    aug    = tf(image=img_np)
    inp    = aug["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(inp)
        logits = F.interpolate(logits, size=(img_size, img_size),
                               mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(PALETTE):
        color_mask[pred == cls_idx] = color

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img_np); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(color_mask); axes[1].set_title("Prediction"); axes[1].axis("off")
    from matplotlib.patches import Patch
    legend = [Patch(color=[c/255 for c in PALETTE[i]], label=CLASS_NAMES[i])
              for i in range(NUM_CLASSES)]
    axes[1].legend(handles=legend, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization -> {save_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    default="./dataset")
    p.add_argument("--arch",         default="deeplabv3+",
                   choices=["deeplabv3+", "deeplabv3+-r50", "unet++", "fpn", "pspnet"])
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--img_size",     type=int,   default=512)
    p.add_argument("--lr",           type=float, default=6e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--focal_gamma",  type=float, default=2.0)
    p.add_argument("--dice_weight",  type=float, default=0.4)
    p.add_argument("--warmup_epochs",type=int,   default=3)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--run_dir",      default="./runs/train")
    p.add_argument("--use_class_weights", action="store_true", default=True)
    p.add_argument("--visualize",    type=str,   default=None,
                   help="Path to image for inference visualization after training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = vars(args)
    cfg["focal_weight"] = 1.0 - cfg["dice_weight"]
    cfg["run_dir"] = f"./runs/{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "="*65)
    print("MASK DECODING (the fix that was causing 40-50% IoU):")
    for pv, ci in PIXEL_TO_CLASS.items():
        print(f"  pixel {pv:2d} -> class {ci} ({CLASS_NAMES[ci]})")
    print("="*65)

    trainer = Trainer(cfg)
    trainer.fit()

    if args.visualize:
        ckpt = torch.load(Path(cfg["run_dir"]) / "best.pth", map_location=trainer.device)
        trainer.model.load_state_dict(ckpt["model_state"])
        visualize_prediction(
            trainer.model, args.visualize, trainer.device,
            str(Path(cfg["run_dir"]) / "prediction_viz.png"),
            cfg["img_size"]
        )