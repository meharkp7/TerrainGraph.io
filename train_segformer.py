"""
SegFormer-B4 Training Script
Optimized for Mac MPS (25.8GB RAM)
Overnight training — 60 epochs
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PIXEL_TO_CLASS = {
    100:  0,   # Trees        -> Dense_Vegetation
    200:  0,   # Lush Bushes  -> Dense_Vegetation
    300:  1,   # Dry Grass    -> Dry_Vegetation
    500:  1,   # Dry Bushes   -> Dry_Vegetation
    550:  2,   # Gnd Clutter  -> Ground_Objects
    600:  2,   # Flowers      -> Ground_Objects
    700:  2,   # Logs         -> Ground_Objects
    800:  3,   # Rocks        -> Rocks
    7100: 4,   # Landscape    -> Landscape
    10000:5,   # Sky          -> Sky
}

CLASS_NAMES = [
    "Dense_Vegetation", "Dry_Vegetation", "Ground_Objects",
    "Rocks", "Landscape", "Sky"
]
NUM_CLASSES  = len(CLASS_NAMES)
IGNORE_INDEX = 255


# ─────────────────────────────────────────────
# MASK REMAPPING
# ─────────────────────────────────────────────

def remap_mask(mask_np: np.ndarray) -> np.ndarray:
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    out = np.full(mask_np.shape, IGNORE_INDEX, dtype=np.uint8)
    for pixel_val, class_idx in PIXEL_TO_CLASS.items():
        out[mask_np == pixel_val] = class_idx
    return out


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────

def get_train_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.2),
        A.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.3, hue=0.1, p=0.8),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.RandomShadow(p=0.2),
        A.RandomFog(fog_coef_lower=0.1,
                    fog_coef_upper=0.25, p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)


def get_val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class DesertSegDataset(Dataset):
    def __init__(self, root: str, split: str = "train",
                 img_size: int = 512):
        self.root      = Path(root)
        self.transform = (get_train_transforms(img_size)
                          if split == "train"
                          else get_val_transforms(img_size))

        img_subfolders  = ["Color_Images", "rgb", "images",
                           "image", "imgs", "color", "img",
                           "color_images", "ColorImages", "RGB"]
        mask_subfolders = ["Segmentation", "segmentation",
                           "masks", "mask", "labels"]
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
            raise FileNotFoundError(
                f"Image dir not found under {self.root}")
        if mask_dir is None or not mask_dir.exists():
            raise FileNotFoundError(
                f"Mask dir not found under {self.root}")

        exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
        self.images = sorted(
            [p for e in exts for p in img_dir.glob(e)])
        self.masks  = sorted(
            [p for e in exts for p in mask_dir.glob(e)])

        print(f"  images   : {len(self.images)}")
        print(f"  masks    : {len(self.masks)}")

        assert len(self.images) > 0, \
            f"No images found in {img_dir}"
        assert len(self.images) == len(self.masks), (
            f"Mismatch: {len(self.images)} images "
            f"vs {len(self.masks)} masks"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = np.array(
            Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))
        mask = remap_mask(mask)
        aug  = self.transform(image=img, mask=mask)
        m    = aug["mask"].long()
        if m.ndim == 3:
            m = m[0]
        return aug["image"].float(), m.contiguous()


# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────

def compute_class_weights(dataset: DesertSegDataset,
                          sample: int = 300) -> torch.Tensor:
    print("Computing class weights...")
    counts  = np.zeros(NUM_CLASSES, dtype=np.float64)
    indices = np.random.choice(
        len(dataset), min(sample, len(dataset)), replace=False)
    for i in indices:
        _, mask = dataset[i]
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
# LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None,
                 ignore_index=255):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        B, C, H, W   = logits.shape
        lf           = logits.permute(0,2,3,1).contiguous().reshape(-1, C)
        tf           = targets.contiguous().reshape(-1)
        valid        = tf != self.ignore_index
        lf, tf       = lf[valid].contiguous(), tf[valid].contiguous()
        if tf.numel() == 0:
            return logits.sum() * 0.0
        log_p = torch.log_softmax(lf, dim=1)
        p     = torch.exp(log_p)
        lpt   = log_p.gather(1, tf.unsqueeze(1)).squeeze(1)
        pt    = p.gather(1, tf.unsqueeze(1)).squeeze(1)
        loss  = -((1 - pt) ** self.gamma) * lpt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[tf]
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        pf = torch.softmax(
            logits, dim=1).permute(0,2,3,1).contiguous().reshape(-1, C)
        tf = targets.contiguous().reshape(-1)
        valid  = tf != self.ignore_index
        pf, tf = pf[valid].contiguous(), tf[valid].contiguous()
        if tf.numel() == 0:
            return logits.sum() * 0.0
        oh    = torch.zeros_like(pf).scatter_(
            1, tf.unsqueeze(1), 1.0)
        inter = (pf * oh).sum(0)
        union = pf.sum(0) + oh.sum(0)
        return (1 - (2*inter + self.smooth) /
                (union + self.smooth)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight=None, focal_gamma=2.0,
                 dice_w=0.4, focal_w=0.6):
        super().__init__()
        self.focal   = FocalLoss(gamma=focal_gamma,
                                  weight=weight)
        self.dice    = DiceLoss()
        self.dice_w  = dice_w
        self.focal_w = focal_w

    def forward(self, logits, targets):
        return (self.focal_w * self.focal(logits, targets) +
                self.dice_w  * self.dice(logits, targets))


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class IoUMetric:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union        = np.zeros(self.num_classes)

    def update(self, preds, targets):
        preds   = preds.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        valid   = targets != self.ignore_index
        for c in range(self.num_classes):
            pred_c = (preds == c) & valid
            tgt_c  = (targets == c) & valid
            self.intersection[c] += (pred_c & tgt_c).sum()
            self.union[c]        += (pred_c | tgt_c).sum()

    def compute(self):
        iou = np.where(
            self.union > 0,
            self.intersection / (self.union + 1e-8),
            np.nan
        )
        valid = iou[~np.isnan(iou)]
        miou  = float(np.mean(valid)) if len(valid) > 0 else 0.0
        return {
            "mIoU": miou,
            "per_class": {
                CLASS_NAMES[i]: float(iou[i])
                for i in range(self.num_classes)
            }
        }


# ─────────────────────────────────────────────
# MODEL — SegFormer B4
# ─────────────────────────────────────────────

def build_segformer(variant: str = "b4",
                    num_classes: int = NUM_CLASSES) -> nn.Module:
    model_name = f"nvidia/mit-{variant}"
    print(f"Loading SegFormer-{variant.upper()} "
          f"from HuggingFace...")
    config = SegformerConfig.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Device
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
        self.train_ds = DesertSegDataset(
            cfg["data_root"], "train", cfg["img_size"])
        self.val_ds   = DesertSegDataset(
            cfg["data_root"], "val",   cfg["img_size"])

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=0,      # MPS = 0 workers
            pin_memory=False,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # Model
        self.model = build_segformer(
            cfg.get("variant", "b4"), NUM_CLASSES
        ).to(self.device)

        total_params = sum(
            p.numel() for p in self.model.parameters())
        print(f"Model: SegFormer-B{cfg.get('variant','4')} "
              f"| Params: {total_params/1e6:.1f}M")

        # Loss
        class_weights = compute_class_weights(self.train_ds)
        self.criterion = CombinedLoss(
            weight=class_weights,
            focal_gamma=cfg.get("focal_gamma", 2.0),
            dice_w=cfg.get("dice_weight", 0.4),
            focal_w=cfg.get("focal_weight", 0.6),
        )

        # Optimizer — different LR for encoder vs decoder
        encoder_params = list(
            self.model.segformer.parameters())
        decoder_params = list(
            self.model.decode_head.parameters())

        self.optimizer = optim.AdamW([
            {"params": encoder_params,
             "lr": cfg["lr"] * 0.1},      # encoder = 10x lower lr
            {"params": decoder_params,
             "lr": cfg["lr"]},
        ], weight_decay=cfg.get("weight_decay", 1e-4))

        # Scheduler — cosine with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg["epochs"] - cfg.get("warmup_epochs", 5),
            eta_min=1e-7,
        )

        self.best_miou = 0.0
        self.history   = {
            "train_loss": [], "val_loss": [], "val_miou": []}

    def _warmup_lr(self, epoch: int):
        warmup = self.cfg.get("warmup_epochs", 5)
        if epoch < warmup:
            factor = (epoch + 1) / warmup
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * factor

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self._warmup_lr(epoch)
        total_loss = 0.0

        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            self.optimizer.zero_grad()

            # SegFormer forward
            out    = self.model(pixel_values=images)
            logits = out.logits  # (B, C, H/4, W/4)

            # Upsample to input size
            logits = F.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            logits = logits.contiguous()
            masks  = masks.contiguous()
            loss = self.criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        metric     = IoUMetric(NUM_CLASSES)

        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            out    = self.model(pixel_values=images)
            logits = out.logits
            logits = F.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            logits = logits.contiguous()
            masks  = masks.contiguous()
            total_loss += self.criterion(logits, masks).item()
            metric.update(logits, masks)

        results = metric.compute()
        return (total_loss / len(self.val_loader),
                results["mIoU"], results)

    def save_checkpoint(self, epoch, miou, tag="best"):
        torch.save({
            "epoch":           epoch,
            "miou":            miou,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg":             self.cfg,
            "class_names":     CLASS_NAMES,
            "pixel_to_class":  PIXEL_TO_CLASS,
            "arch":            "segformer-b4",
        }, self.run_dir / f"{tag}.pth")

    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history["train_loss"],
                     label="Train Loss")
        axes[0].plot(self.history["val_loss"],
                     label="Val Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].set_xlabel("Epoch")

        axes[1].plot(self.history["val_miou"],
                     label="Val mIoU", color="green")
        axes[1].set_title("mIoU")
        axes[1].legend()
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            self.run_dir / "training_curves.png", dpi=150)
        plt.close()

    def fit(self) -> float:
        epochs = self.cfg["epochs"]
        warmup = self.cfg.get("warmup_epochs", 5)

        print(f"\n{'='*65}")
        print(f"SegFormer-B4 | Device: {self.device} "
              f"| Epochs: {epochs}")
        print(f"Train: {len(self.train_ds)} "
              f"| Val: {len(self.val_ds)}")
        print(f"Batch: {self.cfg['batch_size']} "
              f"| ImgSize: {self.cfg['img_size']}")
        print(f"LR encoder: {self.cfg['lr']*0.1:.2e} "
              f"| LR decoder: {self.cfg['lr']:.2e}")
        print(f"{'='*65}\n")

        for epoch in range(epochs):
            t0         = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss, val_miou, iou_detail = self.val_epoch()

            if epoch >= warmup:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_miou"].append(val_miou)

            elapsed = time.time() - t0
            eta_hrs = (epochs - epoch - 1) * elapsed / 3600

            print(f"Epoch [{epoch+1:03d}/{epochs}] "
                  f"Loss={train_loss:.4f} "
                  f"ValLoss={val_loss:.4f} "
                  f"mIoU={val_miou:.4f} "
                  f"({elapsed:.0f}s) "
                  f"ETA={eta_hrs:.1f}h")

            # Per-class every 5 epochs
            if (epoch + 1) % 5 == 0:
                print()
                for cls, iou in iou_detail["per_class"].items():
                    bar = '█' * int(iou * 20)
                    print(f"    {cls:25s}: {iou:.4f}  {bar}")
                print()

            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint(epoch, val_miou, "best")
                with open(self.run_dir / "best_iou.json",
                          "w") as f:
                    json.dump(iou_detail, f, indent=2)
                print(f"  ✓ New best mIoU: {val_miou:.4f}")

            self.save_checkpoint(epoch, val_miou, "last")

            # Save history every epoch
            # (safe even if training interrupted overnight)
            json.dump(
                self.history,
                open(self.run_dir / "history.json", "w"),
                indent=2
            )

        self.plot_history()
        print(f"\n{'='*65}")
        print(f"Training complete!")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"Saved to:  {self.run_dir}")
        print(f"{'='*65}")
        return self.best_miou


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     default="./dataset")
    p.add_argument("--variant",       default="b4",
                   choices=["b2", "b3", "b4", "b5"])
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--batch_size",    type=int,   default=6)
    p.add_argument("--img_size",      type=int,   default=512)
    p.add_argument("--lr",            type=float, default=6e-5)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--focal_gamma",   type=float, default=2.0)
    p.add_argument("--dice_weight",   type=float, default=0.4)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    args = p.parse_args()

    cfg = vars(args)
    cfg["focal_weight"] = 1.0 - cfg["dice_weight"]
    cfg["run_dir"] = (
        f"./runs/segformer_{cfg['variant']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print("\n" + "="*65)
    print(f"SegFormer-B{cfg['variant'].upper()} "
          f"Overnight Training")
    print("="*65)

    trainer = Trainer(cfg)
    trainer.fit()