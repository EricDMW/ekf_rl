"""Characterise the onboard camera data and quantify its potential for a
vision-based extension. (The camera is not used by the current policy.)

Outputs:
  figures/img_montage.png, img_stats.png, img_potential.png
  cache/image_report.json
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from . import config as C
np.random.seed(0)


def list_runs_with_images():
    runs = sorted(glob.glob(os.path.join(C.DATA_DIR, "*", "Go1_dataset", "run_*")))
    out = []
    for r in runs:
        imgs = sorted(glob.glob(os.path.join(r, "images", "*.png")))
        if imgs:
            out.append((os.path.basename(r), imgs))
    return out


def analyse(per_run_sample=25):
    runs = list_runs_with_images()
    total = sum(len(i) for _, i in runs)
    bright, colorful, sharp, tdiff, hist, montage = [], [], [], [], [], []
    sizes = set()
    for name, imgs in runs:
        idxs = np.linspace(0, len(imgs) - 1, min(per_run_sample, len(imgs))).astype(int)
        prev = None
        for i in idxs:
            im = np.asarray(Image.open(imgs[i]).convert("RGB"))
            sizes.add(im.shape[:2])
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            bright.append(float(g.mean()))
            r, gg, b = im[..., 0].astype(float), im[..., 1].astype(float), im[..., 2].astype(float)
            rg, yb = np.abs(r - gg), np.abs(0.5 * (r + gg) - b)
            colorful.append(float(np.sqrt(rg.std()**2 + yb.std()**2)
                                  + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)))
            sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            h = cv2.calcHist([g], [0], None, [32], [0, 256]).flatten()
            hist.append(h / (h.sum() + 1e-9))
            if prev is not None:
                tdiff.append(float(np.abs(g.astype(float) - prev).mean()))
            prev = g.astype(float)
        montage.append((name, np.asarray(Image.open(imgs[len(imgs)//2]).convert("RGB"))))

    stats = dict(n_runs=len(runs), total_frames=int(total),
                 resolution=sorted([f"{h}x{w}" for h, w in sizes]),
                 brightness=dict(mean=float(np.mean(bright)), std=float(np.std(bright)),
                                 min=float(np.min(bright)), max=float(np.max(bright))),
                 colorfulness=dict(mean=float(np.mean(colorful))),
                 sharpness=dict(mean=float(np.mean(sharp)),
                                p10=float(np.percentile(sharp, 10)),
                                p90=float(np.percentile(sharp, 90))),
                 temporal_diff=dict(mean=float(np.mean(tdiff))))
    json.dump(stats, open(os.path.join(C.CACHE_DIR, "image_report.json"), "w"), indent=2)
    _montage(montage); _stats(bright, sharp, tdiff, hist)
    _potential(montage[len(montage)//2][1])
    print("[img]", json.dumps(stats, indent=2))
    return stats


def _montage(frames, ncol=5):
    n = min(len(frames), 15); nrow = int(np.ceil(n/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2.2, nrow*2.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes: ax.axis("off")
    for ax, (nm, im) in zip(axes, frames[:n]):
        ax.imshow(im); ax.set_title(nm.replace("run_", ""), fontsize=6)
    fig.suptitle("Go1 onboard camera — one frame per run", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "img_montage.png"), dpi=130); plt.close(fig)


def _stats(bright, sharp, tdiff, hist):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0,0].hist(bright, bins=40, color="#4C78A8"); ax[0,0].set_title("Brightness (mean grey)")
    ax[0,1].hist(sharp, bins=40, color="#F58518"); ax[0,1].set_title("Sharpness (var Laplacian)"); ax[0,1].set_yscale("log")
    ax[1,0].hist(tdiff, bins=40, color="#54A24B"); ax[1,0].set_title("Temporal frame difference (motion)")
    mh = np.mean(np.array(hist), 0); ax[1,1].bar(np.arange(len(mh)), mh, color="#B279A2")
    ax[1,1].set_title("Mean intensity histogram")
    fig.suptitle("Onboard camera — dataset image statistics", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "img_stats.png"), dpi=130); plt.close(fig)


def _potential(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(g, 60, 160)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    sal = cv2.GaussianBlur(np.sqrt(gx**2 + gy**2), (0, 0), 5)
    sal = (sal - sal.min()) / (np.ptp(sal) + 1e-9)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax[0].imshow(im); ax[0].set_title("RGB frame")
    ax[1].imshow(edges, cmap="gray"); ax[1].set_title("Canny edges (obstacles/structure)")
    ax[2].imshow(sal, cmap="magma"); ax[2].set_title("Gradient saliency")
    ax[3].imshow(hsv[..., 0], cmap="hsv"); ax[3].set_title("Hue (segmentation cue)")
    for a in ax: a.axis("off")
    fig.suptitle("Potential visual signals for a vision-based extension", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "img_potential.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    analyse()
