import cv2
import numpy as np
from pathlib import Path

def normalize_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def preprocess(img, size=224):
    img = cv2.resize(img, (size, size))
    img = normalize_lighting(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def run(split):
    src = Path("data/raw") / split
    dst = Path("data/processed") / split
    count = 0
    for f in src.rglob("*"):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        out = preprocess(img)
        out_path = dst / f.parent.name / (f.stem + ".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out)
        count += 1
    print(split, "done:", count)

if __name__ == "__main__":
    for s in ("train", "val", "test"):
        run(s)