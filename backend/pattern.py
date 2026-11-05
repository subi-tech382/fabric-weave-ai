import io
import cv2
import numpy as np


# ---------- Repeat-unit detection ----------
def _period(profile, min_p=8):
    n = len(profile) // 2
    p = profile[:n]
    low = np.where(p < 0.15)[0]
    if len(low) == 0:
        return None
    start = max(int(low[0]), min_p)
    for i in range(start + 1, n - 1):
        if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] > 0.3:
            lo, hi = max(i - 3, start), min(i + 4, n)
            return int(lo + np.argmax(p[lo:hi]))
    return None


def detect_period(img):
    """Return (px, py) repeat size or (None, None)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g -= g.mean()
    f = np.fft.fft2(g)
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac /= ac.max() + 1e-9
    h, w = g.shape
    return _period(ac[0, :w]), _period(ac[:h, 0])


# ---------- Seamless tiling ----------
def make_seamless(tile):
    """Cross-fade a tile with its half-shifted copy so opposite edges match."""
    h, w = tile.shape[:2]
    t = tile.astype(np.float32)
    shifted = np.roll(t, (h // 2, w // 2), axis=(0, 1))
    ax = np.sin(np.pi * (np.arange(w) + 0.5) / w)
    ay = np.sin(np.pi * (np.arange(h) + 0.5) / h)
    a = np.sqrt(ay[:, None] * ax[None, :])[..., None]
    out = t * a + shifted * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def mirror_tile(img):
    """Always-seamless fallback: 2x2 mirror of a centre crop."""
    h, w = img.shape[:2]
    s = int(min(h, w) * 0.5)
    y0, x0 = (h - s) // 2, (w - s) // 2
    b = img[y0:y0 + s, x0:x0 + s]
    top = np.hstack([b, b[:, ::-1]])
    return np.vstack([top, top[::-1]])


def build_tile(img, tile_w=0, tile_h=0):
    """Return (seamless_tile, width, height, method)."""
    h, w = img.shape[:2]
    if tile_w > 0 and tile_h > 0:
        tw = int(min(tile_w, w))
        th = int(min(tile_h, h))
        tile = img[:th, :tw]
        return make_seamless(tile), tw, th, "manual"
    px, py = detect_period(img)
    if px and py:
        return make_seamless(img[:py, :px]), px, py, "autocorrelation"
    t = mirror_tile(img)
    return t, t.shape[1], t.shape[0], "mirror-fallback"


# ---------- Damage inpainting ----------
def inpaint(img, mask_gray):
    m = cv2.resize(mask_gray, (img.shape[1], img.shape[0]))
    m = (m > 127).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    if m.max() == 0:
        return img
    return cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)


# ---------- Weave analysis ----------
def _dom_freq(g, axis):
    s = np.abs(np.fft.rfft(g - g.mean(), axis=axis)).mean(axis=1 - axis)
    s[:6] = 0
    return int(np.argmax(s[:128]))


def weave_stats(img):
    g = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = _dom_freq(g, 1)   # variations along x -> vertical threads
    weft = _dom_freq(g, 0)   # variations along y -> horizontal threads
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    ang = 0.5 * np.degrees(np.arctan2(2 * (gx * gy).sum(), (gx * gx).sum() - (gy * gy).sum()))
    yarn = ang + 90.0
    if yarn > 90:
        yarn -= 180
    a = abs(yarn)
    if a < 15 or a > 75:
        hint = "Plain / grid-like texture"
    elif 30 <= a <= 60:
        hint = "Twill-like diagonal texture"
    else:
        hint = "Mixed / jacquard-like texture"
    return {"warp_threads": warp, "weft_threads": weft,
            "yarn_angle": round(float(yarn), 1), "texture_hint": hint}


def palette(img, k=5):
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    out = []
    for i in np.argsort(-counts):
        b, g, r = centers[i]
        out.append({"hex": "#%02x%02x%02x" % (int(r), int(g), int(b)),
                    "percent": round(float(100.0 * counts[i] / counts.sum()), 1)})
    return out


# ---------- PDF spec sheet ----------
def spec_pdf(d, tile_png):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.colors import HexColor, black

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, H - 60, "Fabric Weave Specification Sheet")
    c.setFont("Helvetica", 12)
    rows = [
        ("Fabric type", str(d.get("fabric_type", ""))),
        ("Confidence", "%.1f %%" % (float(d.get("confidence", 0)) * 100)),
        ("Repeat unit", "%s x %s px" % (d.get("tile_width"), d.get("tile_height"))),
        ("Detection method", str(d.get("method", ""))),
        ("Warp threads (per 512 px)", str(d.get("warp_threads", ""))),
        ("Weft threads (per 512 px)", str(d.get("weft_threads", ""))),
        ("Yarn direction", "%s deg" % d.get("yarn_angle", "")),
        ("Texture estimate", str(d.get("texture_hint", ""))),
    ]
    y = H - 100
    for k, v in rows:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, k + ":")
        c.setFont("Helvetica", 12)
        c.drawString(230, y, v)
        y -= 22
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Colour palette")
    y -= 40
    x = 50
    for p in d.get("palette", []):
        c.setFillColor(HexColor(p["hex"]))
        c.rect(x, y, 40, 30, fill=1, stroke=0)
        c.setFillColor(black)
        c.setFont("Helvetica", 8)
        c.drawString(x, y - 10, "%s %s%%" % (p["hex"], p["percent"]))
        x += 90
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Repeat tile")
    img = ImageReader(io.BytesIO(tile_png))
    c.drawImage(img, 50, y - 260, width=250, height=250, preserveAspectRatio=True)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 40, "Thread counts and yarn direction are image-based estimates.")
    c.save()
    return buf.getvalue()