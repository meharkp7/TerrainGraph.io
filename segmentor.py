"""
segmentor.py
────────────
Loads best.pth and runs inference on any image.
Returns segmentation mask + class distribution.
"""

import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

PIXEL_TO_CLASS = {
    100:0, 200:0, 300:1, 500:1,
    550:2, 600:2, 700:2, 800:3,
    7100:4, 10000:5,
}
CLASS_NAMES = [
    "Dense_Vegetation", "Dry_Vegetation", "Ground_Objects",
    "Rocks", "Landscape", "Sky"
]
NUM_CLASSES  = len(CLASS_NAMES)
IGNORE_INDEX = 255

TRAVERSABILITY = {
    0: 0.45,   # Dense_Vegetation
    1: 0.75,   # Dry_Vegetation
    2: 0.30,   # Ground_Objects
    3: 0.15,   # Rocks
    4: 0.95,   # Landscape
    5: 0.00,   # Sky
}
RISK = {
    0: "MED",
    1: "LOW",
    2: "MED-HIGH",
    3: "HIGH",
    4: "LOW",
    5: "N/A",
}

PALETTE = np.array([
    [34,  139,  34],   # Dense_Vegetation  → forest green
    [210, 180, 140],   # Dry_Vegetation    → tan
    [139,  90,  43],   # Ground_Objects    → brown
    [128, 128, 128],   # Rocks             → gray
    [205, 133,  63],   # Landscape         → sandy
    [135, 206, 235],   # Sky               → sky blue
], dtype=np.uint8)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str) -> tuple:
    """Load model from checkpoint. Returns (model, device, cfg)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt.get("cfg", {})
    arch = cfg.get("arch", "deeplabv3+").lower()

    if arch in ("deeplabv3+", "deeplabv3"):
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    elif arch == "deeplabv3+-r50":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    elif arch == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    elif arch == "fpn":
        model = smp.FPN(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    else:
        # fallback
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    print(f"✅ Model loaded: {arch} | mIoU={ckpt.get('miou',0):.4f} | Device={device}")
    return model, device, cfg


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def preprocess(image_path: str, img_size: int = 512) -> tuple:
    """Load and preprocess image. Returns (tensor, original_np)."""
    tf = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img_np = np.array(Image.open(image_path).convert("RGB"))
    aug    = tf(image=img_np)
    tensor = aug["image"].unsqueeze(0).float()
    return tensor, img_np


@torch.no_grad()
def predict(model, device, image_path: str,
            img_size: int = 512,
            use_tta: bool = True) -> dict:
    """
    Run inference on one image.
    Returns dict with mask, probs, class_dist, colorized.
    """
    tensor, orig_np = preprocess(image_path, img_size)
    inp = tensor.to(device)

    # Normal forward
    logits = model(inp)
    logits = F.interpolate(
        logits, size=(img_size, img_size),
        mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)

    # TTA — horizontal flip
    if use_tta:
        flipped       = torch.flip(inp, dims=[-1])
        logits_f      = model(flipped)
        logits_f      = F.interpolate(
            logits_f, size=(img_size, img_size),
            mode="bilinear", align_corners=False)
        probs_f       = torch.softmax(logits_f, dim=1)
        probs_f       = torch.flip(probs_f, dims=[-1])
        probs         = (probs + probs_f) / 2.0

    pred_mask  = probs.argmax(dim=1).squeeze(0).cpu().numpy()
    confidence = probs.max(dim=1).values.squeeze(0).cpu().numpy()

    # Class distribution
    class_dist = {}
    for i, name in enumerate(CLASS_NAMES):
        pct = float((pred_mask == i).mean())
        class_dist[name] = round(pct, 4)

    # Traversability map
    trav_map = np.zeros_like(pred_mask, dtype=np.float32)
    for cls_idx, score in TRAVERSABILITY.items():
        trav_map[pred_mask == cls_idx] = score

    # Colorized mask
    color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color_mask[pred_mask == c] = PALETTE[c]

    # Overlay on original
    orig_resized = np.array(
        Image.fromarray(orig_np).resize(
            (img_size, img_size), Image.BILINEAR))
    overlay = (orig_resized * 0.5 + color_mask * 0.5).astype(np.uint8)

    return {
        "mask":        pred_mask,       # HxW  int
        "probs":       probs.cpu(),     # 1xCxHxW
        "confidence":  confidence,      # HxW  float
        "class_dist":  class_dist,      # dict name→pct
        "trav_map":    trav_map,        # HxW  float 0-1
        "color_mask":  color_mask,      # HxW3 uint8
        "overlay":     overlay,         # HxW3 uint8
        "orig_np":     orig_resized,    # HxW3 uint8
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    ckpt_path  = sys.argv[1] if len(sys.argv) > 1 \
                 else "./runs/deeplabv3+_20260403_211706/best.pth"
    image_path = sys.argv[2] if len(sys.argv) > 2 \
                 else "./dataset/val/Color_Images"

    # Find first image if directory given
    from pathlib import Path
    if Path(image_path).is_dir():
        imgs = list(Path(image_path).glob("*.png"))[:1]
        if imgs:
            image_path = str(imgs[0])

    model, device, cfg = load_model(ckpt_path)
    result = predict(model, device, image_path)

    print("\nClass distribution:")
    for name, pct in result["class_dist"].items():
        bar = '█' * int(pct * 40)
        print(f"  {name:25s}: {pct:.1%}  {bar}")

    print(f"\nMask shape    : {result['mask'].shape}")
    print(f"Trav map range: {result['trav_map'].min():.2f} "
          f"- {result['trav_map'].max():.2f}")

# ─────────────────────────────────────────────
# Segmentor class — wraps module-level functions
# so pipeline.py can do: seg = Segmentor()
# ─────────────────────────────────────────────

class Segmentor:
    """Wraps load_model() + predict() for use by pipeline.py"""

    _SEARCH = [
        "./runs/deeplabv3+_20260403_211623/best.pth",
        "./runs/deeplabv3+_20260403_211706/best.pth",
        "./runs/deeplabv3+_20260404_204700/best.pth",
        "./best_model.pth",
        "./best_model2.pth",
        "./dino_best.pth",
    ]

    def __init__(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = self._find_ckpt()
        self.ckpt_path = checkpoint_path
        self.model, self.device, cfg = load_model(checkpoint_path)
        ckpt      = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.miou = float(ckpt.get("miou", 0.0))
        self.img_size = cfg.get("img_size", 512)

    def _find_ckpt(self) -> str:
        runs = sorted(Path("runs").glob("*/best.pth")) if Path("runs").exists() else []
        for p in self._SEARCH + [str(r) for r in runs]:
            if Path(p).exists():
                print(f"  Using checkpoint: {p}")
                return p
        raise FileNotFoundError(
            "No checkpoint found. Searched:\n" +
            "\n".join(f"  {p}" for p in self._SEARCH)
        )

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Args: img_rgb — HxW×3 uint8 numpy array (RGB)
        Returns: HxW int numpy array, values 0-5 (class indices)
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            Image.fromarray(img_rgb.astype(np.uint8)).save(tmp)
            result = predict(self.model, self.device, tmp,
                             img_size=self.img_size, use_tta=False)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return result["mask"]   # HxW int ndarray