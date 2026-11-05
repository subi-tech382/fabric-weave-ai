import base64
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from preprocess import preprocess, normalize_lighting
import pattern as P

CLASSES = ["Banarasi", "Bandhani", "Kanjeevaram", "Patola", "Tussar"]
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fabric_model.pth"

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="Fabric Weave AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def b64png(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


@app.get("/")
def home():
    return {"status": "Fabric API running"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    mask: UploadFile = File(None),
    tile_w: int = Form(0),
    tile_h: int = Form(0),
):
    raw = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # limit size for speed
    s = 1024 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # 1. Damage inpainting (only if the user painted a mask)
    repaired = None
    if mask is not None:
        mdata = await mask.read()
        if mdata:
            m = cv2.imdecode(np.frombuffer(mdata, np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                img = P.inpaint(img, m)
                repaired = b64png(img)

    # 2. Classification
    p = preprocess(img)
    rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        probs = torch.softmax(model(tf(rgb).unsqueeze(0)), 1)[0]
    top = torch.topk(probs, 3)
    top3 = [{"name": CLASSES[int(i)], "prob": round(float(v), 3)}
            for v, i in zip(top.values, top.indices)]

    # 3. Pattern analysis + seamless tile
    k = 768 / max(img.shape[:2])
    work = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    work = normalize_lighting(work)
    tile, tw, th, method = P.build_tile(work, tile_w, tile_h)
    stats = P.weave_stats(work)

    return {
        "fabric_type": top3[0]["name"],
        "confidence": top3[0]["prob"],
        "top3": top3,
        "tile_width": int(tw),
        "tile_height": int(th),
        "method": method,
        "palette": P.palette(work),
        "tile_image": b64png(tile),
        "repaired_image": repaired,
        **stats,
    }


@app.post("/spec-pdf")
async def spec_pdf(d: dict = Body(...)):
    png = base64.b64decode(d["tile_image"])
    pdf = P.spec_pdf(d, png)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=weave_spec.pdf"})