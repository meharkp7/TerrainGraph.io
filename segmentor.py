"""
segmentor.py — Memory-optimized for Railway (512MB RAM)
Key changes: ResNet-50, CPU-only, 384px inference, no TTA on cloud
"""

import os
import gc
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

IS_RAILWAY = bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY"))
INFER_SIZE = int(os.getenv("INFER_SIZE", "384" if IS_RAILWAY else "512"))

CLASS_NAMES = [
    "Dense_Vegetation", "Dry_Vegetation", "Ground_Objects",
    "Rocks", "Landscape", "Sky"
]
NUM_CLASSES  = len(CLASS_NAMES)
IGNORE_INDEX = 255
TRAVERSABILITY = {0:0.45, 1:0.75, 2:0.30, 3:0.15, 4:0.95, 5:0.00}
RISK           = {0:"MED", 1:"LOW", 2:"MED-HIGH", 3:"HIGH", 4:"LOW", 5:"N/A"}
PALETTE = np.array([
    [34,139,34],[210,180,140],[139,90,43],
    [128,128,128],[205,133,63],[135,206,235]
], dtype=np.uint8)


def _build_model(arch, encoder):
    arch = arch.lower()
    kw   = dict(encoder_name=encoder, encoder_weights=None,
                in_channels=3, classes=NUM_CLASSES)
    if arch in ("deeplabv3+","deeplabv3"): return smp.DeepLabV3Plus(**kw)
    if arch == "unet++":                   return smp.UnetPlusPlus(**kw)
    if arch == "fpn":                      return smp.FPN(**kw)
    return smp.DeepLabV3Plus(**kw)          # fallback


def load_model(checkpoint_path: str) -> tuple:
    ckpt    = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg     = ckpt.get("cfg", {})
    arch    = cfg.get("arch", "deeplabv3+").lower()
    encoder = "resnet50" if IS_RAILWAY else cfg.get("encoder", "resnet101")
    if IS_RAILWAY:
        print("  [Railway] resnet50 encoder (memory budget)")

    model = _build_model(arch, encoder)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError:
        print("  [Warning] Partial weight load — encoder size mismatch.")
        print("  Upload a resnet50 checkpoint to HuggingFace for best accuracy.")
        model.load_state_dict(ckpt["model_state"], strict=False)

    model.eval()
    device = torch.device("cpu")
    model  = model.to(device)
    miou   = ckpt.get("miou", 0.0)
    print(f"✅ Model: {arch}/{encoder} | mIoU={miou:.4f} | cpu")
    return model, device, cfg


def preprocess(image_path: str, img_size=INFER_SIZE):
    tf = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    img_np = np.array(Image.open(image_path).convert("RGB"))
    tensor = tf(image=img_np)["image"].unsqueeze(0).float()
    return tensor, img_np


@torch.inference_mode()
def predict(model, device, image_path: str,
            img_size=INFER_SIZE, use_tta=False):
    tensor, orig_np = preprocess(image_path, img_size)
    inp    = tensor.to(device)
    logits = model(inp)
    logits = F.interpolate(logits, size=(img_size,img_size),
                           mode="bilinear", align_corners=False)
    if use_tta and not IS_RAILWAY:
        fl  = torch.flip(inp, dims=[-1])
        lf  = F.interpolate(model(fl), size=(img_size,img_size),
                            mode="bilinear", align_corners=False)
        logits = (logits + torch.flip(lf, dims=[-1])) / 2.0

    pred_mask  = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    confidence = torch.softmax(logits,dim=1).max(dim=1).values\
                      .squeeze(0).cpu().numpy()
    del logits, inp; gc.collect()

    class_dist = {n: round(float((pred_mask==i).mean()),4)
                  for i,n in enumerate(CLASS_NAMES)}
    trav_map   = np.zeros_like(pred_mask, dtype=np.float32)
    for ci,sc in TRAVERSABILITY.items(): trav_map[pred_mask==ci] = sc

    color_mask = np.zeros((*pred_mask.shape,3), dtype=np.uint8)
    for c in range(NUM_CLASSES): color_mask[pred_mask==c] = PALETTE[c]

    orig_r  = np.array(Image.fromarray(orig_np).resize(
                (img_size,img_size), Image.BILINEAR))
    overlay = (orig_r*0.5 + color_mask*0.5).astype(np.uint8)

    return {"mask":pred_mask,"confidence":confidence,"class_dist":class_dist,
            "trav_map":trav_map,"color_mask":color_mask,
            "overlay":overlay,"orig_np":orig_r}


class Segmentor:
    _SEARCH = [
        "./runs/deployed/best.pth",
        "./runs/deeplabv3+_20260403_211623/best.pth",
        "./runs/deeplabv3+_20260403_211706/best.pth",
        "./runs/deeplabv3+_20260404_204700/best.pth",
        "./best_model.pth","./best_model2.pth","./dino_best.pth",
    ]

    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self._find_ckpt()
        self.model, self.device, cfg = load_model(checkpoint_path)
        ckpt          = torch.load(checkpoint_path, map_location="cpu",
                                   weights_only=False)
        self.miou     = float(ckpt.get("miou", 0.0))
        self.img_size = INFER_SIZE
        del ckpt; gc.collect()

    def _find_ckpt(self):
        runs = sorted(Path("runs").glob("*/best.pth")) \
               if Path("runs").exists() else []
        for p in self._SEARCH + [str(r) for r in runs]:
            if Path(p).exists():
                print(f"  Checkpoint: {p}"); return p
        raise FileNotFoundError("No checkpoint found. Searched:\n" +
                                "\n".join(f"  {p}" for p in self._SEARCH))

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            Image.fromarray(img_rgb.astype(np.uint8)).save(tmp)
            return predict(self.model, self.device, tmp,
                           img_size=self.img_size, use_tta=False)["mask"]
        finally:
            Path(tmp).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    ckpt  = sys.argv[1] if len(sys.argv)>1 else "./runs/deeplabv3+_20260403_211706/best.pth"
    img   = sys.argv[2] if len(sys.argv)>2 else "./dataset/val/Color_Images"
    if Path(img).is_dir():
        imgs = list(Path(img).glob("*.png"))[:1]
        if imgs: img = str(imgs[0])
    m, dev, _ = load_model(ckpt)
    res = predict(m, dev, img)
    for name, pct in res["class_dist"].items():
        print(f"  {name:25s}: {pct:.1%}  {'█'*int(pct*40)}")