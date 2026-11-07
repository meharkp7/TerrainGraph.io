"""
Codec roundtrip test for the HF Space inference path.

Encodes a synthetic mask+confidence (Space side) and decodes it back
(Railway side), asserting the rebuilt segmentor.predict() dict is correct.

Needs only numpy + Pillow — no torch, no gradio_client, no network — so it
runs anywhere, including the lean backend. Run:  python tests/test_space_codec.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from terrain_const import NUM_CLASSES, TRAVERSABILITY, PALETTE
from segmentor_space import (
    SpaceInferenceError,
    decode_space_result,
    encode_mask_files,
)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tgtest_"))

    # Synthetic segmentation: 4 quadrants of classes 0,1,3,4
    mask = np.zeros((512, 512), dtype=np.int64)
    mask[:256, 256:] = 1
    mask[256:, :256] = 3
    mask[256:, 256:] = 4
    conf = np.linspace(0.2, 0.9, 512 * 512,
                       dtype=np.float32).reshape(512, 512)
    class_dist = {"Dense_Vegetation": 0.25, "Dry_Vegetation": 0.25,
                  "Rocks": 0.25, "Landscape": 0.25,
                  "Ground_Objects": 0.0, "Sky": 0.0}
    img_path = str(tmp / "input.png")
    Image.fromarray(
        np.random.randint(0, 255, (640, 480, 3),
                          dtype=np.uint8)).save(img_path)

    # Space side: encode
    mask_path, conf_path = encode_mask_files(mask, conf, out_dir=str(tmp))

    # Railway side: decode
    seg = decode_space_result(mask_path, conf_path, class_dist, img_path)

    assert set(seg) == {"mask", "confidence", "class_dist", "trav_map",
                        "color_mask", "overlay", "orig_np"}, set(seg)
    assert seg["mask"].shape == (512, 512)
    assert np.array_equal(seg["mask"], mask), "mask must survive PNG roundtrip"
    assert np.allclose(seg["confidence"], conf, atol=1 / 255), "conf quantized to 8-bit"
    assert seg["class_dist"] == class_dist
    assert seg["trav_map"].shape == (512, 512)
    assert seg["trav_map"].dtype == np.float32
    for cls_idx, score in TRAVERSABILITY.items():
        assert (seg["trav_map"][mask == cls_idx] == np.float32(score)).all()
    assert seg["color_mask"].shape == (512, 512, 3)
    for c in range(NUM_CLASSES):
        assert (seg["color_mask"][mask == c] == PALETTE[c]).all()
    assert seg["overlay"].shape == (512, 512, 3)
    assert seg["orig_np"].shape == (512, 512, 3)

    # Invalid class ids must fail loudly, not silently misroute
    bad_mask = np.full((8, 8), 7, dtype=np.uint8)
    bad_conf = np.full((8, 8), 0.5, dtype=np.float32)
    bad_mp, bad_cp = encode_mask_files(bad_mask, bad_conf, out_dir=str(tmp))
    try:
        decode_space_result(bad_mp, bad_cp, {}, img_path, img_size=8)
    except SpaceInferenceError:
        pass
    else:
        raise AssertionError("invalid class id was not rejected")

    # The whole codec path must stay torch-free (lean-backend safe)
    assert "torch" not in sys.modules, "torch got imported by the codec path"
    assert "gradio_client" not in sys.modules, \
        "gradio_client imported eagerly (must stay lazy)"

    print("test_space_codec: ALL PASS")


if __name__ == "__main__":
    main()
