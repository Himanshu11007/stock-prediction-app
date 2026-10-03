"""
scripts/generate_brand_assets.py — every StockLens icon asset from the one
approved master icon (branding/stocklens-icon-master.png). Nothing is
redrawn: assets are resized copies of the master; the Android notification
icon is the master's bright artwork (magnifier and chart) as a white
silhouette, which Android requires for status-bar icons.

Usage: python scripts/generate_brand_assets.py [--master PATH] [--mobile-repo PATH]
       --master: import a new master (any format; stored as PNG, lossless)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BRANDING = ROOT / "branding"
MASTER = BRANDING / "stocklens-icon-master.png"
DEFAULT_MOBILE = ROOT.parent / "StockAIPro-Mobile" / "StockAIPro.Mobile" / "StockAIPro.Mobile"
APP_BG = (13, 17, 23)          # app background #0D1117 (splash / social image)


def resized(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


def notification_silhouette(img: Image.Image, size: int) -> Image.Image:
    """White-on-transparent status-bar icon from the master's bright artwork."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    brightness = a.max(axis=2)
    alpha = np.clip((brightness - 110.0) / 60.0, 0, 1)          # dark tile -> 0, artwork -> 1
    ys, xs = np.where(alpha > 0.5)
    pad = 20
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0]))
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    art = Image.fromarray(rgba, "RGBA").crop(box)
    side = max(art.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(art, ((side - art.size[0]) // 2, (side - art.size[1]) // 2))
    return square.resize((size, size), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master")
    ap.add_argument("--mobile-repo", default=str(DEFAULT_MOBILE))
    args = ap.parse_args()
    BRANDING.mkdir(exist_ok=True)
    if args.master:
        Image.open(args.master).convert("RGB").save(MASTER, optimize=True)
    master = Image.open(MASTER).convert("RGB")

    # Website / API / admin console
    for size, name in ((16, "favicon-16.png"), (32, "favicon-32.png"), (180, "apple-touch-icon.png"),
                       (192, "icon-192.png"), (512, "icon-512.png")):
        resized(master, size).save(BRANDING / name, optimize=True)
    resized(master, 256).save(BRANDING / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    og = Image.new("RGB", (1200, 630), APP_BG)
    og.paste(resized(master, 470), ((1200 - 470) // 2, 80))
    og.save(BRANDING / "og-image.png", optimize=True)

    # Mobile app (MAUI resizetizer inputs)
    mobile = Path(args.mobile_repo)
    if mobile.exists():
        resized(master, 1024).save(mobile / "Resources" / "AppIcon" / "stocklens_icon.png", optimize=True)
        resized(master, 512).save(mobile / "Resources" / "Splash" / "stocklens_splash.png", optimize=True)
        for density, px in (("mdpi", 24), ("hdpi", 36), ("xhdpi", 48), ("xxhdpi", 72), ("xxxhdpi", 96)):
            d = mobile / "Platforms" / "Android" / "Resources" / f"drawable-{density}"
            d.mkdir(parents=True, exist_ok=True)
            notification_silhouette(master, px).save(d / "ic_stat_stocklens.png", optimize=True)
    print("assets written")


if __name__ == "__main__":
    main()
