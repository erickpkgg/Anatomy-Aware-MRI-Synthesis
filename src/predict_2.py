#!/usr/bin/env python3
"""
predict.py — Inference for trained RegionConditionedUNet models.

Minimal usage (single model):
    python predict.py \
        --source subject_t1.nii.gz \
        --target subject_flair.nii.gz \
        --mask   subject_mask.nii.gz \
        --model_aa  ./exp_anatomy_aware/checkpoints/best.pt \
        --age 65 --sex M --domain 0 \
        --outdir ./predictions

With all three models (generates comparative plots):
    python predict.py \
        --source     subject_t1.nii.gz \
        --target     subject_flair.nii.gz \
        --mask       subject_mask.nii.gz \
        --model_aa   ./exp_anatomy_aware/checkpoints/best.pt \
        --model_unif ./exp_uniform_weights/checkpoints/best.pt \
        --model_base ./exp_no_mask/checkpoints/best.pt \
        --age 65 --sex M --domain 0 \
        --outdir ./predictions

Mandatory arguments:
    --source         Source image (T1 or FLAIR) in NIfTI format
    --outdir         Output directory
    At least one of: --model_aa, --model_unif, --model_base

Optional arguments:
    --target         Real target image (to calculate metrics and error maps)
    --mask           8-class segmentation mask (for WMH-L1 and overlay)
    --age            Subject age (default: 60)
    --sex            Sex: M or F (default: M)
    --domain         Cluster/domain index (default: 0)
    --direction      0=T1→FLAIR, 1=FLAIR→T1 (default: 0)
    --patch_size     Patch size for sliding window (default: 96 96 96)
    --overlap        Sliding window overlap 0-1 (default: 0.5)
    --base_ch        Model base channels (default: 32)
    --cond_dim       Dimension of the conditioning vector (default: 64)
    --no_nifti       Do not save NIfTI files, only plots
    --subject_id     Subject name for output files (default: "subject")
    --view           View: axial (Z), coronal (Y) or sagittal (X) (default: axial)
    --slices         Volume fractions for slices (default: 0.35 0.5 0.65)
                     (replaces --axial_slices)
    --dpi            Plot DPI (default: 180)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

try:
    from monai.inferers import SlidingWindowInferer
except ImportError:
    sys.exit("ERROR: unrecognized library monai → pip install monai")

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


REGION_COLORS = {
    0: (0.00, 0.00, 0.00, 0.00),  
    1: (0.20, 0.60, 0.86, 0.35),  
    2: (0.95, 0.77, 0.06, 0.30),  
    3: (0.22, 0.72, 0.29, 0.35),  
    4: (0.58, 0.40, 0.74, 0.30),  
    5: (0.99, 0.55, 0.24, 0.30),  
    6: (0.21, 0.49, 0.72, 0.35),  
    7: (1.00, 0.10, 0.10, 0.90),
}
REGION_LABELS = {
    0: "Background", 1: "Cortical GM", 2: "Cerebral WM",
    3: "Subcortical GM", 4: "Cerebellar GM", 5: "Cerebellar WM",
    6: "Brainstem", 7: "WMH",
}


class FiLMBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act   = nn.GELU()
        self.skip  = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.film  = nn.Linear(cond_dim, 2 * out_ch)

    def forward(self, x, cond):
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        res = self.skip(x)
        h   = self.act(self.norm1(self.conv1(x)))
        h   = self.norm2(self.conv2(h))
        h   = h * (1.0 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
        return self.act(h + res)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.pool  = nn.Conv3d(in_ch, in_ch, 2, stride=2, groups=in_ch, bias=False)
        self.block = FiLMBlock3D(in_ch, out_ch, cond_dim)

    def forward(self, x, cond):
        return self.block(self.pool(x), cond)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim):
        super().__init__()
        self.up    = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.block = FiLMBlock3D(out_ch + skip_ch, out_ch, cond_dim)

    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape != skip.shape:
            x = x[:, :, :skip.shape[2], :skip.shape[3], :skip.shape[4]]
        return self.block(torch.cat([x, skip], dim=1), cond)


class RegionConditionedUNet(nn.Module):
    def __init__(self, base_ch=32, cond_dim=64, n_domains=4, n_cov=3):
        super().__init__()
        C = base_ch
        self.domain_emb = nn.Embedding(n_domains, 16)
        self.cov_mlp    = nn.Sequential(nn.Linear(n_cov, 32), nn.GELU(), nn.Linear(32, 32))
        self.cond_proj  = nn.Sequential(nn.Linear(48, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim))
        self.stem   = FiLMBlock3D(1, C, cond_dim)
        self.down1  = DownBlock(C,   C*2, cond_dim)
        self.down2  = DownBlock(C*2, C*4, cond_dim)
        self.down3  = DownBlock(C*4, C*8, cond_dim)
        self.down4  = DownBlock(C*8, C*8, cond_dim)
        self.bottleneck = nn.Sequential(FiLMBlock3D(C*8, C*8, cond_dim), FiLMBlock3D(C*8, C*8, cond_dim))
        self.up4 = UpBlock(C*8, C*8, C*8, cond_dim)
        self.up3 = UpBlock(C*8, C*4, C*4, cond_dim)
        self.up2 = UpBlock(C*4, C*2, C*2, cond_dim)
        self.up1 = UpBlock(C*2, C,   C,   cond_dim)
        self.head = nn.Sequential(nn.Conv3d(C, C, 3, padding=1), nn.GELU(), nn.Conv3d(C, 1, 1))

    def make_cond(self, cov, domain):
        return self.cond_proj(torch.cat([self.domain_emb(domain), self.cov_mlp(cov)], dim=-1))

    def forward(self, x, cov, domain):
        cond = self.make_cond(cov, domain)
        s0 = self.stem(x, cond)
        s1 = self.down1(s0, cond); s2 = self.down2(s1, cond)
        s3 = self.down3(s2, cond); s4 = self.down4(s3, cond)
        b  = self.bottleneck[0](s4, cond); b = self.bottleneck[1](b, cond)
        u4 = self.up4(b, s3, cond); u3 = self.up3(u4, s2, cond)
        u2 = self.up2(u3, s1, cond); u1 = self.up1(u2, s0, cond)
        return self.head(u1)

def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"3D volume expected, {arr.shape} was obtained in {path}")
    return arr, img

def robust_normalize(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    vals = x[(mask > 0) & np.isfinite(x)] if mask is not None else x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    med   = np.median(vals)
    mad   = np.median(np.abs(vals - med))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(np.std(vals)) if np.std(vals) > 1e-6 else 1.0
    return np.clip((x - med) / scale, -5.0, 5.0).astype(np.float32)

def encode_sex(sex_str: str) -> float:
    return 1.0 if str(sex_str).strip().lower() in {"m", "male", "1", "man"} else 0.0

def display_normalize(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(vol)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)

def load_model(ckpt_path: str, base_ch: int, cond_dim: int,
               device: torch.device) -> Tuple[RegionConditionedUNet, int]:

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state" in ckpt:
        state  = ckpt["model_state"]
        n_dom  = ckpt.get("n_domains", state["domain_emb.weight"].shape[0])
    else:
        state  = ckpt
        n_dom  = state["domain_emb.weight"].shape[0]

    model = RegionConditionedUNet(
        base_ch=base_ch, cond_dim=cond_dim, n_domains=n_dom, n_cov=3
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, n_dom

def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    p1 = display_normalize(pred)
    t1 = display_normalize(target)

    l1   = float(np.mean(np.abs(p1 - t1)))
    mse  = float(np.mean((p1 - t1) ** 2))
    psnr = 20 * math.log10(1.0 + 1e-8) - 10 * math.log10(mse + 1e-8)

    ssim_vals = []
    if HAS_SKIMAGE:
        for axis, idx in [(0, p1.shape[0]//2), (1, p1.shape[1]//2), (2, p1.shape[2]//2)]:
            sl_p = np.take(p1, idx, axis=axis)
            sl_t = np.take(t1, idx, axis=axis)
            ws   = 7 if min(sl_p.shape) >= 7 else 3
            try:
                ssim_vals.append(float(skimage_ssim(sl_p, sl_t, data_range=1.0, win_size=ws)))
            except Exception:
                pass
    ssim = float(np.mean(ssim_vals)) if ssim_vals else float("nan")

    wmh_l1 = float("nan")
    if mask is not None:
        wmh_vox = mask == 7
        if wmh_vox.sum() > 0:
            wmh_l1 = float(np.mean(np.abs(p1[wmh_vox] - t1[wmh_vox])))

    return {"L1": l1, "PSNR": psnr, "SSIM": ssim, "WMH_L1": wmh_l1}


def mask_overlay(ax, slice_2d: np.ndarray, mask_2d: np.ndarray,
                 show_classes=(7,), alpha_scale: float = 1.0) -> None:
    
    h, w = slice_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls in show_classes:
        if cls not in REGION_COLORS:
            continue
        r, g, b, a = REGION_COLORS[cls]
        region = mask_2d == cls
        rgba[region, 0] = r
        rgba[region, 1] = g
        rgba[region, 2] = b
        rgba[region, 3] = a * alpha_scale
    ax.imshow(rgba, interpolation="nearest")


def _get_slices(vol: np.ndarray, fracs: List[float]
                ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    D, H, W = vol.shape
    result = []
    for f in fracs:
        zi = int(D * f); yi = int(H * f); xi = int(W * f)
        result.append((vol[zi], vol[:, yi, :], vol[:, :, xi]))
    return result


def get_slice(vol: np.ndarray, frac: float, view: str) -> np.ndarray:
    
    D, H, W = vol.shape
    if view == "axial":
        idx = min(int(D * frac), D - 1)
        return vol[idx]
    elif view == "coronal":
        idx = min(int(H * frac), H - 1)
        sl  = vol[:, idx, :]        # [D, W]
        return np.flipud(sl)        # cabeza arriba
    else:  # sagital
        idx = min(int(W * frac), W - 1)
        sl  = vol[:, :, idx]        # [D, H]
        return np.flipud(sl)        # cabeza arriba


def slice_label(frac: float, view: str, shape: tuple) -> str:
    
    D, H, W = shape
    labels = {"axial": ("z", D), "coronal": ("y", H), "sagital": ("x", W)}
    letter, dim = labels.get(view, ("z", D))
    return f"{letter}={int(dim * frac)}"


def plot_single_model(
    source:  np.ndarray,
    pred:    np.ndarray,
    target:  Optional[np.ndarray],
    mask:    Optional[np.ndarray],
    metrics: Optional[Dict[str, float]],
    model_label: str,
    direction_str: str,
    fracs:   List[float],
    outpath: Path,
    dpi:     int = 180,
    view:    str = "axial",
) -> None:
    has_target = target is not None
    has_mask   = mask is not None
    n_slices   = len(fracs)

    src_d  = display_normalize(source)
    pred_d = display_normalize(pred)
    tgt_d  = display_normalize(target) if has_target else None

    cols = ["Source\n(Input)"]
    if has_target: cols += ["Ground Truth"]
    if has_target and has_mask: cols += ["GT + WMH"]
    cols += ["Prediction"]
    if has_target: cols += ["Abs. Error"]
    if has_mask:   cols += ["Pred + WMH"]

    n_cols = len(cols)
    fig, axes = plt.subplots(n_slices, n_cols,
                             figsize=(3.2 * n_cols, 3.2 * n_slices),
                             squeeze=False)

    cmap_err = plt.cm.hot

    for row, frac in enumerate(fracs):
        src_sl  = get_slice(src_d,  frac, view)
        pred_sl = get_slice(pred_d, frac, view)
        tgt_sl  = get_slice(tgt_d,  frac, view) if has_target else None
        mask_sl = get_slice(mask.astype(np.float32), frac, view).astype(np.int64) if has_mask else None

        col = 0
        axes[row, col].imshow(src_sl, cmap="gray", vmin=0, vmax=1); col += 1
        if has_target:
            axes[row, col].imshow(tgt_sl, cmap="gray", vmin=0, vmax=1); col += 1
        # GT + WMH overlay column
        if has_target and has_mask:
            axes[row, col].imshow(tgt_sl, cmap="gray", vmin=0, vmax=1)
            mask_overlay(axes[row, col], tgt_sl, mask_sl, show_classes=(7,))
            col += 1
        axes[row, col].imshow(pred_sl, cmap="gray", vmin=0, vmax=1); col += 1
        if has_target:
            err = np.abs(pred_sl - tgt_sl)
            im  = axes[row, col].imshow(err, cmap=cmap_err, vmin=0, vmax=0.5); col += 1
        if has_mask:
            axes[row, col].imshow(pred_sl, cmap="gray", vmin=0, vmax=1)
            mask_overlay(axes[row, col], pred_sl, mask_sl, show_classes=(7,))
            col += 1

        # Etiqueta del corte
        axes[row, 0].set_ylabel(slice_label(frac, view, source.shape),
                                fontsize=8, rotation=0, labelpad=28, va="center")

    # Títulos de columnas
    for c, title in enumerate(cols):
        axes[0, c].set_title(title, fontsize=9, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    # Métricas en el título
    title_parts = [f"{model_label}  |  {direction_str}"]
    if metrics:
        parts = [f"L1={metrics['L1']:.4f}", f"PSNR={metrics['PSNR']:.2f}dB",
                 f"SSIM={metrics['SSIM']:.4f}"]
        if not math.isnan(metrics.get("WMH_L1", float("nan"))):
            parts.append(f"WMH-L1={metrics['WMH_L1']:.4f}")
        title_parts.append("  ".join(parts))

    plt.suptitle("\n".join(title_parts), fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {outpath.name}")


def plot_comparison(
    source:   np.ndarray,
    target:   Optional[np.ndarray],
    mask:     Optional[np.ndarray],
    preds:    Dict[str, np.ndarray],
    metrics:  Dict[str, Dict[str, float]],
    direction_str: str,
    fracs:    List[float],
    outpath:  Path,
    dpi:      int = 180,
    view:     str = "axial",
) -> None:
    
    has_target = target is not None
    has_mask   = mask   is not None
    model_names = list(preds.keys())
    n_models   = len(model_names)
    n_slices   = len(fracs)

    src_d  = display_normalize(source)
    pred_ds = {k: display_normalize(v) for k, v in preds.items()}
    tgt_d  = display_normalize(target) if has_target else None

    # Columnas: Source, [GT], (pred_i, error_i)*n, [WMH overlay del mejor]
    col_labels = ["Source"]
    if has_target: col_labels += ["Ground\nTruth"]
    if has_target and has_mask: col_labels += ["GT\n+ WMH"]
    for nm in model_names:
        short = nm.replace("Anatomy-Aware", "AA").replace("Uniform Weights", "Unif").replace("Baseline", "Base")
        col_labels += [f"{short}\nPrediction"]
        if has_target: col_labels += [f"{short}\nAbs. Error"]
    if has_mask: col_labels += ["Best Model\n+ WMH"]

    n_cols = len(col_labels)
    fig, axes = plt.subplots(n_slices, n_cols,
                             figsize=(2.8 * n_cols, 2.8 * n_slices),
                             squeeze=False)

    cmap_err = plt.cm.hot
    best_model = min(metrics, key=lambda k: metrics[k].get("L1", 1e9)) if metrics else model_names[0]

    for row, frac in enumerate(fracs):
        src_sl  = get_slice(src_d, frac, view)
        tgt_sl  = get_slice(tgt_d, frac, view) if has_target else None
        mask_sl = get_slice(mask.astype(np.float32), frac, view).astype(np.int64) if has_mask else None

        col = 0
        axes[row, col].imshow(src_sl, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(slice_label(frac, view, source.shape),
                                fontsize=7, rotation=0, labelpad=28, va="center")
        col += 1

        if has_target:
            axes[row, col].imshow(tgt_sl, cmap="gray", vmin=0, vmax=1); col += 1

        # GT + WMH overlay column
        if has_target and has_mask:
            axes[row, col].imshow(tgt_sl, cmap="gray", vmin=0, vmax=1)
            mask_overlay(axes[row, col], tgt_sl, mask_sl, show_classes=(7,))
            col += 1

        for nm in model_names:
            pred_sl = get_slice(pred_ds[nm], frac, view)
            axes[row, col].imshow(pred_sl, cmap="gray", vmin=0, vmax=1); col += 1
            if has_target:
                err = np.abs(pred_sl - tgt_sl)
                axes[row, col].imshow(err, cmap=cmap_err, vmin=0, vmax=0.5); col += 1

        if has_mask:
            best_sl = get_slice(pred_ds[best_model], frac, view)
            axes[row, col].imshow(best_sl, cmap="gray", vmin=0, vmax=1)
            mask_overlay(axes[row, col], best_sl, mask_sl, show_classes=(7,))
            col += 1

    for c, title in enumerate(col_labels):
        axes[0, c].set_title(title, fontsize=8, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    metric_lines = []
    for nm in model_names:
        m = metrics.get(nm, {})
        line = f"{nm}: L1={m.get('L1',float('nan')):.4f} | PSNR={m.get('PSNR',float('nan')):.2f}dB | SSIM={m.get('SSIM',float('nan')):.4f}"
        if not math.isnan(m.get("WMH_L1", float("nan"))):
            line += f" | WMH-L1={m['WMH_L1']:.4f}"
        metric_lines.append(line)

    title_block = f"Comparison — {direction_str}\n" + "\n".join(metric_lines)
    plt.suptitle(title_block, fontsize=8, y=1.02, family="monospace")
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {outpath.name}")


def plot_wmh_detail(
    pred:    np.ndarray,
    target:  Optional[np.ndarray],
    mask:    Optional[np.ndarray],
    model_label: str,
    outpath: Path,
    dpi: int = 180,
    n_samples: int = 6,
    view: str = "axial",
) -> None:
    
    if mask is None or (mask == 7).sum() == 0:
        return

    # Encontrar los n_samples cortes con más WMH según la vista elegida
    D, H, W = mask.shape
    axis_dim = {"axial": D, "coronal": H, "sagital": W}[view]
    def _wmh_count(i):
        frac = i / max(axis_dim - 1, 1)
        sl   = get_slice((mask == 7).astype(np.float32), frac, view)
        return sl.sum()
    wmh_per_idx = [_wmh_count(i) for i in range(axis_dim)]
    top_indices = sorted(range(axis_dim), key=lambda i: -wmh_per_idx[i])[:n_samples]
    top_indices = sorted(top_indices)

    pred_d = display_normalize(pred)
    tgt_d  = display_normalize(target) if target is not None else None
    has_target = tgt_d is not None

    # cols: pred | [GT] | [GT+WMH] | overlay(pred)
    n_cols = (2 if not has_target else 4) if mask is not None else (1 if not has_target else 2)
    if not has_target: n_cols = 2  # pred | overlay(pred)
    else: n_cols = 4               # pred | GT | GT+WMH | pred+WMH
    fig, axes = plt.subplots(len(top_indices), n_cols,
                             figsize=(3 * n_cols, 3 * len(top_indices)),
                             squeeze=False)

    for row, idx in enumerate(top_indices):
        frac    = idx / max(axis_dim - 1, 1)
        pred_sl = get_slice(pred_d, frac, view)
        mask_sl = get_slice(mask.astype(np.float32), frac, view).astype(np.int64)

        axes[row, 0].imshow(pred_sl, cmap="gray", vmin=0, vmax=1)
        lbl = slice_label(frac, view, mask.shape)
        axes[row, 0].set_ylabel(f"{lbl}\n({int(wmh_per_idx[idx])} WMH px)",
                                 fontsize=7, rotation=0, labelpad=40, va="center")
        col = 1
        if has_target:
            tgt_sl_d = get_slice(tgt_d, frac, view)
            axes[row, col].imshow(tgt_sl_d, cmap="gray", vmin=0, vmax=1); col += 1
            # GT + WMH overlay
            axes[row, col].imshow(tgt_sl_d, cmap="gray", vmin=0, vmax=1)
            mask_overlay(axes[row, col], tgt_sl_d, mask_sl, show_classes=(7,))
            col += 1
        axes[row, col].imshow(pred_sl, cmap="gray", vmin=0, vmax=1)
        mask_overlay(axes[row, col], pred_sl, mask_sl, show_classes=(7,))

    titles = ["Prediction"]
    if has_target: titles += ["Ground Truth", "GT + WMH (red)"]
    titles += ["Pred + WMH (red)"]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=9, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(f"WMH Detail — {model_label}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {outpath.name}")


def plot_metrics_bar(
    metrics: Dict[str, Dict[str, float]],
    outpath: Path,
    dpi: int = 150,
) -> None:
    if len(metrics) < 2:
        return

    metric_keys = ["L1", "PSNR", "SSIM", "WMH_L1"]
    metric_labels = ["L1 ↓", "PSNR (dB) ↑", "SSIM ↑", "WMH-L1 ↓"]
    model_names = list(metrics.keys())
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"][:len(model_names)]

    # Filtrar métricas con valores válidos para todos los modelos
    valid_keys, valid_labels = [], []
    for k, l in zip(metric_keys, metric_labels):
        if all(not math.isnan(metrics[m].get(k, float("nan"))) for m in model_names):
            valid_keys.append(k); valid_labels.append(l)

    if not valid_keys:
        return

    n_metrics = len(valid_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4), squeeze=False)

    x = np.arange(len(model_names))
    for i, (key, label) in enumerate(zip(valid_keys, valid_labels)):
        ax = axes[0, i]
        vals = [metrics[m][key] for m in model_names]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8)
        ax.set_ylabel(key, fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Model Comparison — Quantitative Metrics", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {outpath.name}")


@torch.no_grad()
def run_inference(model: RegionConditionedUNet,
                  source_norm: np.ndarray,
                  cov: torch.Tensor,
                  domain: torch.Tensor,
                  patch_size: Tuple[int, int, int],
                  overlap: float,
                  device: torch.device) -> np.ndarray:
    inferer = SlidingWindowInferer(
        roi_size=patch_size, sw_batch_size=2,
        overlap=overlap, mode="gaussian"
    )
    x = torch.from_numpy(source_norm[None, None]).to(device)  # [1,1,D,H,W]
    pred = inferer(x, lambda t: model(t, cov, domain))
    pred = torch.clamp(pred, -5.0, 5.0)
    return pred[0, 0].cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="RegionConditionedUNet model inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input data
    p.add_argument("--source",   required=True,  help="Source image (.nii/.nii.gz)")
    p.add_argument("--target",   default=None,   help="Real target image (ground truth, used for metrics and plots)")
    p.add_argument("--mask",     default=None,   help="Segmentation mask (.nii/.nii.gz)")

    # Models (at least one is required)
    p.add_argument("--model_aa",   default=None, help="Checkpoint Anatomy-Aware (best.pt o last.pt)")
    p.add_argument("--model_unif", default=None, help="Checkpoint Uniform Weights")
    p.add_argument("--model_base", default=None, help="Checkpoint Baseline (no mask)")

    # Covs
    p.add_argument("--age",       type=float, default=60.0, help="Subject age")
    p.add_argument("--sex",       type=str,   default="M",  help="Sex: M o F")
    p.add_argument("--domain",    type=int,   default=0,    help="Cluster index/domain")
    p.add_argument("--direction", type=int,   default=0,    choices=[0, 1],
                   help="0=T1→FLAIR, 1=FLAIR→T1")

    # Arquitecture
    p.add_argument("--base_ch",  type=int, default=32)
    p.add_argument("--cond_dim", type=int, default=64)

    # Inference
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--overlap",    type=float, default=0.5)

    # Output
    p.add_argument("--outdir",      required=True, help="Output path")
    p.add_argument("--subject_id",  default="subject", help="Base name for output files")
    p.add_argument("--no_nifti",    action="store_true", help="No NIfTI saving, just plots and graphs")
    p.add_argument("--view", type=str, default="axial",
                   choices=["axial", "coronal", "sagital"],
                   help="Slices view: axial (Z), coronal (Y) o sagital (X)")
    p.add_argument("--slices", type=float, nargs="+", default=[0.35, 0.50, 0.65],
                   help="Fractions of the selected axis for the plot slicing")
    p.add_argument("--axial_slices", type=float, nargs="+", default=None,
                   help="slices")
    p.add_argument("--dpi", type=int, default=180)

    args = p.parse_args()

    # Model validation
    model_specs = {
        "Anatomy-Aware":    args.model_aa,
        "Uniform Weights":  args.model_unif,
        "Baseline":         args.model_base,
    }
    active_models = {k: v for k, v in model_specs.items() if v is not None}
    if not active_models:
        p.error("You must pass at least one of: --model_aa, --model_unif, --model_base")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Models to use: {list(active_models.keys())}")

    direction_str = "T1 → FLAIR" if args.direction == 0 else "FLAIR → T1"

    # ── Data loading ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading images...")
    source_raw, ref_img = load_nifti(args.source)
    affine, header = ref_img.affine, ref_img.header

    support = np.abs(source_raw) > 1e-8
    source_norm = robust_normalize(source_raw, mask=support.astype(np.uint8))

    target_raw  = None
    target_norm = None
    if args.target:
        target_raw, _ = load_nifti(args.target)
        t_support      = np.abs(target_raw) > 1e-8
        target_norm    = robust_normalize(target_raw, mask=t_support.astype(np.uint8))

    mask_vol = None
    if args.mask:
        mask_vol, _ = load_nifti(args.mask)
        mask_vol = mask_vol.astype(np.int64)

    # Covs
    cov = torch.tensor([[
        np.clip(args.age, 0.0, 120.0) / 100.0,
        encode_sex(args.sex),
        float(args.direction),
    ]], dtype=torch.float32).to(device)
    domain = torch.tensor([args.domain], dtype=torch.long).to(device)

    # ── Inference ────────────────────────────────────────────────────────────
    print("\n[2/4] Running inference...")
    predictions: Dict[str, np.ndarray] = {}
    all_metrics: Dict[str, Dict[str, float]] = {}

    for model_name, ckpt_path in active_models.items():
        print(f"  → {model_name}: {ckpt_path}")
        model, n_dom = load_model(ckpt_path, args.base_ch, args.cond_dim, device)

        if args.domain >= n_dom:
            print(f"  [WARN] domain={args.domain} out of range for {model_name} "
                  f"(n_domains={n_dom}). Using domain=0.")
            dom_tensor = torch.tensor([0], dtype=torch.long).to(device)
        else:
            dom_tensor = domain

        pred_norm = run_inference(
            model, source_norm, cov, dom_tensor,
            tuple(args.patch_size), args.overlap, device
        )
        predictions[model_name] = pred_norm

        # Metrics
        if target_norm is not None:
            m = compute_metrics(pred_norm, target_norm, mask_vol)
            all_metrics[model_name] = m
            print(f"     L1={m['L1']:.4f} | PSNR={m['PSNR']:.2f}dB | SSIM={m['SSIM']:.4f}"
                  + (f" | WMH-L1={m['WMH_L1']:.4f}" if not math.isnan(m['WMH_L1']) else ""))

        # VRAM 
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Save NIfTI ─────────────────────────────────────────────────────────
    if not args.no_nifti:
        print("\n[3/4] Saving NIfTI...")
        suffix_map = {
            "Anatomy-Aware":   "aa",
            "Uniform Weights": "unif",
            "Baseline":        "base",
        }
        for model_name, pred in predictions.items():
            sfx  = suffix_map.get(model_name, model_name.lower().replace(" ", "_"))
            fname = f"{args.subject_id}_pred_{sfx}.nii.gz"
            nib.save(nib.Nifti1Image(pred, affine, header), outdir / fname)
            print(f"  [NIfTI] {fname}")
    else:
        print("\n[3/4] NIfTI omitted (--no_nifti)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[4/4] Generating plots...")
    
    fracs = args.axial_slices if args.axial_slices is not None else args.slices
    view  = args.view

    if len(active_models) == 1:
        
        model_name = list(predictions.keys())[0]
        pred       = predictions[model_name]
        metrics    = all_metrics.get(model_name)

        plot_single_model(
            source=source_norm, pred=pred,
            target=target_norm, mask=mask_vol,
            metrics=metrics,
            model_label=model_name,
            direction_str=direction_str,
            fracs=fracs,
            outpath=outdir / f"{args.subject_id}_plot_{model_name.lower().replace(' ','_')}.png",
            dpi=args.dpi,
            view=view,
        )
        plot_wmh_detail(
            pred=pred, target=target_norm, mask=mask_vol,
            model_label=model_name,
            outpath=outdir / f"{args.subject_id}_wmh_detail.png",
            dpi=args.dpi,
            view=view,
        )

    else:
        plot_comparison(
            source=source_norm, target=target_norm, mask=mask_vol,
            preds=predictions, metrics=all_metrics,
            direction_str=direction_str, fracs=fracs,
            outpath=outdir / f"{args.subject_id}_comparison.png",
            dpi=args.dpi,
            view=view,
        )

        suffix_map = {"Anatomy-Aware": "aa", "Uniform Weights": "unif", "Baseline": "base"}
        for model_name, pred in predictions.items():
            sfx = suffix_map.get(model_name, model_name.lower().replace(" ", "_"))
            plot_single_model(
                source=source_norm, pred=pred,
                target=target_norm, mask=mask_vol,
                metrics=all_metrics.get(model_name),
                model_label=model_name,
                direction_str=direction_str,
                fracs=fracs,
                outpath=outdir / f"{args.subject_id}_plot_{sfx}.png",
                dpi=args.dpi,
                view=view,
            )

        if all_metrics:
            plot_metrics_bar(
                metrics=all_metrics,
                outpath=outdir / f"{args.subject_id}_metrics_bar.png",
                dpi=args.dpi,
            )

        if all_metrics:
            best = min(all_metrics, key=lambda k: all_metrics[k].get("L1", 1e9))
        else:
            best = list(predictions.keys())[0]
        plot_wmh_detail(
            pred=predictions[best], target=target_norm, mask=mask_vol,
            model_label=f"{best} (best)",
            outpath=outdir / f"{args.subject_id}_wmh_detail_best.png",
            dpi=args.dpi,
            view=view,
        )

    print(f"\n{'='*60}")
    print(f"Subject: {args.subject_id}  |  Direction: {direction_str}")
    if all_metrics:
        print(f"{'Model':<20} {'L1':>8} {'PSNR':>8} {'SSIM':>8} {'WMH-L1':>10}")
        print("-" * 60)
        for nm, m in all_metrics.items():
            wmh_str = f"{m['WMH_L1']:.4f}" if not math.isnan(m['WMH_L1']) else "  n/a"
            print(f"{nm:<20} {m['L1']:>8.4f} {m['PSNR']:>8.2f} {m['SSIM']:>8.4f} {wmh_str:>10}")
    print(f"{'='*60}")
    print(f"Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
