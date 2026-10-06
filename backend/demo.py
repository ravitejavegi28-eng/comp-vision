"""Reproducible SYNTHETIC fixtures; never presented as real factory data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def _font(size: int, bold: bool = False):
    for filename in [f"C:/Windows/Fonts/{'segoeuib' if bold else 'segoeui'}.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def make_label(seed: int, defect: str | None = None) -> Image.Image:
    """A constant product design with small acceptable acquisition variations."""
    rng = np.random.default_rng(seed)
    canvas = Image.new("RGB", (640, 480), (218, 226, 230))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((104, 56, 540, 432), radius=7, fill=(194, 202, 207))
    draw.rounded_rectangle((100, 50, 536, 426), radius=7, fill=(252, 249, 240), outline=(155, 166, 170), width=2)
    draw.rectangle((101, 51, 535, 94), fill=(27, 73, 79))
    draw.text((124, 61), "FIELDWORK / ORIGINAL BLEND", fill="white", font=_font(15, True))
    draw.text((126, 121), "Oat & honey", fill=(29, 57, 62), font=_font(40, True))
    draw.text((128, 176), "Everyday granola", fill=(73, 95, 96), font=_font(23))
    draw.line((127, 224, 509, 224), fill=(202, 204, 189), width=2)
    draw.text((128, 244), "Whole oats. A little sweetness.", fill=(67, 79, 76), font=_font(18))
    draw.text((128, 273), "Made for a slower morning.", fill=(67, 79, 76), font=_font(18))
    draw.rounded_rectangle((128, 316, 233, 349), radius=4, outline=(64, 111, 109), width=2)
    draw.text((144, 322), "NET 400 g", fill=(29, 73, 79), font=_font(15, True))
    for i in range(65):
        x = 309 + i * 3
        if (i * 7 + 3) % 11 < 7:
            draw.rectangle((x, 315, x + (1 if i % 3 else 2), 367), fill=(37, 46, 48))
    draw.text((309, 372), "8 901234 567890", fill=(42, 65, 68), font=_font(14))
    draw.text((128, 391), "LOT FW-024    KEEP COOL & DRY", fill=(89, 104, 101), font=_font(12))
    # Validation uses smudge/missing print. Tear/stain are held-out families.
    x, y = int(rng.integers(180, 400)), int(rng.integers(150, 265))
    if defect == "smudge":
        draw.ellipse((x - 21, y - 10, x + 37, y + 35), fill=(30, 45, 51))
        draw.line((x - 25, y + 10, x + 58, y + 27), fill=(48, 59, 65), width=10)
    elif defect == "missing-print":
        draw.rectangle((x - 25, 135, x + 65, 207), fill=(252, 249, 240))
    elif defect == "tear":
        draw.polygon([(x, 100), (x + 17, 138), (x + 2, 167), (x + 30, 215), (x + 8, 273), (x - 7, 226), (x - 18, 180)], fill=(119, 132, 137))
        draw.line((x, 100, x + 12, 161, x + 20, 215, x + 8, 273), fill=(237, 230, 211), width=3)
    elif defect == "stain":
        draw.ellipse((x - 28, y - 19, x + 48, y + 43), fill=(139, 80, 43))
        draw.ellipse((x + 47, y + 13, x + 59, y + 24), fill=(139, 80, 43))
    elif defect == "barcode-damage":
        draw.rectangle((332, 324, 451, 352), fill=(252, 249, 240))
    canvas = canvas.rotate(float(rng.uniform(-0.65, 0.65)), Image.Resampling.BICUBIC, fillcolor=(218, 226, 230))
    arr = np.asarray(canvas).astype(np.float32)
    arr = arr * rng.uniform(0.96, 1.04) + rng.normal(0, 0.7, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def create_demo_dataset(directory: Path) -> list[dict]:
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.exists():
        records = json.loads(manifest.read_text(encoding="utf-8"))
        if records and all(Path(r["path"]).is_file() for r in records):
            return records
    records = []
    for split, n_good, n_defect, seed_base in [("train", 32, 0, 1000), ("validation", 10, 10, 2000), ("test", 12, 12, 3000)]:
        families = ["smudge", "missing-print"] if split == "validation" else ["smudge", "tear", "stain", "barcode-damage"]
        for label, count in [("good", n_good), ("defect", n_defect)]:
            for index in range(count):
                defect = families[index % len(families)] if label == "defect" else None
                seed = seed_base + index + (100 if defect else 0)
                name = f"{split}-{defect or 'good'}-{index + 1:02d}"
                target = directory / split / label / f"{name}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                make_label(seed, defect).save(target)
                records.append({"id": hashlib.sha256(name.encode()).hexdigest()[:20], "path": str(target), "name": name, "label": label, "split": split, "session_id": f"synthetic-{split}-{seed}", "source": "synthetic-demo", "defect_type": defect})
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    records = create_demo_dataset(Path(__file__).resolve().parents[1] / "data" / "demo")
    print(f"Created {len(records)} synthetic packaging samples. These are software fixtures, not factory validation data.")
