"""Download public ImageNet weights once; keep all model files in this project."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".cache" / "resnet18-f37072fd.pth"
URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.is_file() and hashlib.sha256(DEST.read_bytes()).hexdigest().startswith("f37072fd"):
        print(f"Verified pretrained backbone: {DEST}")
        return
    temporary = DEST.with_suffix(".download")
    print("Downloading public ResNet18 weights (about 45 MB)...", flush=True)
    with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Weight integrity check failed. Retry the download.")
    temporary.replace(DEST)
    print(f"Backbone downloaded and verified: {DEST}")


if __name__ == "__main__":
    main()
