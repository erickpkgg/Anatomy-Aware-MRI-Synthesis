#!/usr/bin/env python3
"""
figures_paper.py — Generates the supplementary figures for the ICBSP 2026 paper.

Figures produced:
  1. learning_curves.png      — Val L1 and Val WMH-L1 per epoch (3 models)
  2. intensity_profile.png    — Intensity profile along a line crossing WMH
  3. wmh_boxplot.png          — WMH-L1 stratified by lesion load (terciles)

Minimal usage (learning curves only):
    python figures_paper.py \
        --csv_aa    ./exp_anatomy_aware/metrics.csv \
        --csv_unif  ./exp_uniform/metrics.csv \
        --csv_base  ./exp_no_mask/metrics.csv \
        --outdir    ./figures

Full usage (all three figures):
    python figures_paper.py \
        --csv_aa    ./exp_anatomy_aware/metrics.csv \
        --csv_unif  ./exp_uniform/metrics.csv \
        --csv_base  ./exp_no_mask/metrics.csv \
        --ckpt_aa   ./exp_anatomy_aware/checkpoints/best.pt \
        --ckpt_unif ./exp_uniform/checkpoints/best.pt \
        --ckpt_base ./exp_no_mask/checkpoints/best.pt \
        --test_excel  ./tu_dataset.xlsx \
        --outdir    ./figures \
        --age 65 --sex M --domain 0 \
        --base_ch 32 --cond_dim 64 \
        --patch_size 96 96 96 \
        --dpi 300
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.inferers import SlidingWindowInferer
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

# ──────────────────────────────────────────────────────────────────────────────
# Palette consistent with the paper
# ──────────────────────────────────────────────────────────────────────────────
MODEL_STYLES = {
    "Anatomy-Aware":   {"color": "#1976D2", "ls": "-",  "lw": 2.2, "marker": "o"},
    "Uniform Weights": {"color": "#F57C00", "ls": "--", "lw": 1.8, "marker": "s"},
    "Baseline":        {"color": "#616161", "ls": ":",  "lw": 1.6, "marker": "^"},
}
GT_COLOR    = "#2E7D32"
SOURCE_COLOR = "#6A1B9A"


# ──────────────────────────────────────────────────────────────────────────────
# Architecture (copied from predict.py so this file is standalone)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_nifti(path: str) -> np.ndarray:
    arr = nib.load(path).get_fdata(dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {arr.shape}")
    return arr

def robust_normalize(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    vals = x[(mask > 0) & np.isfinite(x)] if mask is not None else x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    med   = np.median(vals)
    mad   = np.median(np.abs(vals - med))
    scale = 1.4826 * mad if mad > 1e-6 else (float(np.std(vals)) if np.std(vals) > 1e-6 else 1.0)
    return np.clip((x - med) / scale, -5.0, 5.0).astype(np.float32)

def display_normalize(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(vol)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)

def encode_sex(s: str) -> float:
    return 1.0 if str(s).strip().lower() in {"m", "male", "1", "man"} else 0.0

def file_ok(p) -> bool:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return False
    return Path(str(p)).exists()

def load_model(ckpt_path: str, base_ch: int, cond_dim: int,
               device: torch.device) -> RegionConditionedUNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
        n_dom = ckpt.get("n_domains", state["domain_emb.weight"].shape[0])
    else:
        state = ckpt
        n_dom = state["domain_emb.weight"].shape[0]
    model = RegionConditionedUNet(base_ch=base_ch, cond_dim=cond_dim,
                                  n_domains=n_dom, n_cov=3).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def run_inference(model, source_norm, cov, domain, patch_size, overlap, device):
    if not HAS_MONAI:
        raise ImportError("monai requerido para inferencia: pip install monai")
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=2,
                                   overlap=overlap, mode="gaussian")
    x = torch.from_numpy(source_norm[None, None]).to(device)
    pred = inferer(x, lambda t: model(t, cov, domain))
    return torch.clamp(pred, -5.0, 5.0)[0, 0].cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Learning curves
# ──────────────────────────────────────────────────────────────────────────────

def fig_learning_curves(csvs: Dict[str, str], outpath: Path, dpi: int = 300,
                         smooth: int = 3) -> None:
    """
    Val L1 and Val WMH-L1 per epoch for the three models.
    smooth: moving-average window used to smooth epoch-to-epoch noise.
    """
    def moving_avg(arr, w):
        if w <= 1 or len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for model_name, csv_path in csvs.items():
        if not Path(csv_path).exists():
            print(f"  [WARN] CSV not found: {csv_path} — skipping {model_name}")
            continue

        df  = pd.read_csv(csv_path)
        st  = MODEL_STYLES[model_name]
        eps = df["epoch"].values

        # Val L1
        l1 = df["val_l1"].values
        l1_sm = moving_avg(l1, smooth)
        eps_sm = eps[len(eps) - len(l1_sm):]
        ax1.plot(eps_sm, l1_sm, color=st["color"], ls=st["ls"], lw=st["lw"],
                 label=model_name)
        ax1.plot(eps, l1, color=st["color"], alpha=0.18, lw=0.8)

        # Val WMH-L1
        if "val_wmh_l1" in df.columns:
            wmh = pd.to_numeric(df["val_wmh_l1"], errors="coerce").values
            valid = np.isfinite(wmh)
            if valid.sum() > 2:
                wmh_sm = moving_avg(wmh[valid], smooth)
                eps_w  = eps[valid][len(eps[valid]) - len(wmh_sm):]
                ax2.plot(eps_w, wmh_sm, color=st["color"], ls=st["ls"], lw=st["lw"],
                         label=model_name)
                ax2.plot(eps[valid], wmh[valid], color=st["color"], alpha=0.18, lw=0.8)

    for ax, ylabel, title in [
        (ax1, "Val L1 ↓", "Overall Reconstruction (Val L1)"),
        (ax2, "Val WMH-L1 ↓", "WMH-Specific Reconstruction (Val WMH-L1)"),
    ]:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Learning Curves — Ablation Study", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [FIG] {outpath.name}")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Intensity profile over WMH
# ──────────────────────────────────────────────────────────────────────────────

def _find_wmh_line(mask: np.ndarray, source_norm: np.ndarray,
                   min_wmh_vox: int = 20
                   ) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Finds an axial slice with enough WMH and a horizontal line
    crossing the lesion center of mass.
    Returns (z, row, col_start, col_end, wmh_count) or None if no WMH is present.
    """
    # Slice with the most WMH
    wmh_per_z = [(mask[z] == 7).sum() for z in range(mask.shape[0])]
    z_best    = int(np.argmax(wmh_per_z))
    n_wmh     = wmh_per_z[z_best]

    if n_wmh < min_wmh_vox:
        return None

    wmh_slice = (mask[z_best] == 7).astype(np.uint8)
    # Lesion center of mass in that slice
    cy, cx = ndimage.center_of_mass(wmh_slice)
    row    = int(cy)

    # Horizontal line centered at cx with a margin
    H, W   = wmh_slice.shape
    margin = min(60, W // 4)
    c0     = max(0, int(cx) - margin)
    c1     = min(W, int(cx) + margin)

    return z_best, row, c0, c1, n_wmh


def fig_intensity_profile(
    source_norm:   np.ndarray,
    target_norm:   np.ndarray,
    mask:          np.ndarray,
    preds:         Dict[str, np.ndarray],
    outpath:       Path,
    dpi:           int = 300,
) -> None:
    """
    Two panels:
    Left: axial slice with the sampling line overlaid
    Right: intensity along the line for GT and 3 models
    """
    result = _find_wmh_line(mask, source_norm)
    if result is None:
        print("  [WARN] Not enough WMH found for the intensity profile — skipping")
        return

    z, row, c0, c1, n_wmh = result
    print(f"  [INFO] Intensity profile: z={z}, row={row}, cols=[{c0},{c1}], WMH_vox={n_wmh}")

    # Normalizar para display
    src_d = display_normalize(source_norm)
    tgt_d = display_normalize(target_norm)
    preds_d = {k: display_normalize(v) for k, v in preds.items()}

    src_sl  = src_d[z]
    tgt_sl  = tgt_d[z]
    mask_sl = mask[z]

    cols = np.arange(c0, c1)

    fig = plt.figure(figsize=(12, 4.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6], wspace=0.35)

    # ── Left panel: source slice with line ────────────────────────────────────
    ax_src = fig.add_subplot(gs[0])
    ax_src.imshow(src_sl, cmap="gray", vmin=0, vmax=1)
    ax_src.axhline(row, color=SOURCE_COLOR, lw=1.5, ls="--", alpha=0.8)
    ax_src.axvline(c0, color="white", lw=0.8, alpha=0.5)
    ax_src.axvline(c1, color="white", lw=0.8, alpha=0.5)
    # Overlay WMH
    h, w   = src_sl.shape
    rgba   = np.zeros((h, w, 4), dtype=np.float32)
    wmh_px = mask_sl == 7
    rgba[wmh_px] = [1.0, 0.15, 0.15, 0.75]
    ax_src.imshow(rgba, interpolation="nearest")
    ax_src.set_title("Source (T1w)\nwith sampling line", fontsize=10, fontweight="bold")
    ax_src.axis("off")

    # ── Middle panel: target slice with the same line ───────────────────────
    ax_tgt = fig.add_subplot(gs[1])
    ax_tgt.imshow(tgt_sl, cmap="gray", vmin=0, vmax=1)
    ax_tgt.axhline(row, color=GT_COLOR, lw=1.5, ls="--", alpha=0.8)
    ax_tgt.imshow(rgba, interpolation="nearest")  # mismo overlay WMH
    ax_tgt.set_title("Ground Truth (FLAIR)\nWMH highlighted", fontsize=10, fontweight="bold")
    ax_tgt.axis("off")

    # ── Panel derecho: perfiles de intensidad ────────────────────────────────
    ax_prof = fig.add_subplot(gs[2])

    # Ground truth
    gt_profile = tgt_sl[row, c0:c1]
    ax_prof.plot(cols, gt_profile, color=GT_COLOR, lw=2.2, ls="-",
                 label="Ground Truth", zorder=5)

    # Source (referencia)
    src_profile = src_sl[row, c0:c1]
    ax_prof.plot(cols, src_profile, color=SOURCE_COLOR, lw=1.4, ls="-.",
                 alpha=0.6, label="Source (T1w)", zorder=3)

    # Cada modelo
    for model_name, pred_d in preds_d.items():
        st = MODEL_STYLES[model_name]
        profile = pred_d[z][row, c0:c1]
        ax_prof.plot(cols, profile, color=st["color"], lw=st["lw"], ls=st["ls"],
                     label=model_name, zorder=4)

    # Shade the WMH region in the profile
    wmh_cols = np.where(mask_sl[row, c0:c1] == 7)[0] + c0
    if len(wmh_cols) > 0:
        wmin, wmax = wmh_cols.min(), wmh_cols.max()
        ax_prof.axvspan(wmin, wmax, alpha=0.12, color="red", label="WMH region")

    ax_prof.set_xlabel("Voxel position (column)", fontsize=10)
    ax_prof.set_ylabel("Normalized intensity", fontsize=10)
    ax_prof.set_title("Intensity Profile Along Sampling Line", fontsize=10, fontweight="bold")
    ax_prof.legend(fontsize=8.5, framealpha=0.9, loc="best")
    ax_prof.spines["top"].set_visible(False)
    ax_prof.spines["right"].set_visible(False)
    ax_prof.grid(axis="y", alpha=0.2, linestyle="--")
    ax_prof.set_xlim(c0, c1 - 1)

    fig.suptitle("Intensity Profile over WMH Region", fontsize=12, fontweight="bold", y=1.02)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [FIG] {outpath.name}")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — WMH-L1 box plot by lesion load
# ──────────────────────────────────────────────────────────────────────────────

def _compute_wmh_l1_subject(pred: np.ndarray, target: np.ndarray,
                             mask: np.ndarray) -> Tuple[float, float]:
    """
    Retorna (wmh_volume_mm3, wmh_l1) para un sujeto.
    wmh_volume = number of class-7 voxels (assuming 1 mm isotropic voxels = mm³).
    """
    wmh_vox = mask == 7
    vol = float(wmh_vox.sum())
    if vol == 0:
        return vol, float("nan")

    lo = np.percentile(target, 1)
    hi = np.percentile(target, 99)
    p_d = np.clip((pred   - lo) / (hi - lo + 1e-6), 0, 1)
    t_d = np.clip((target - lo) / (hi - lo + 1e-6), 0, 1)
    l1  = float(np.mean(np.abs(p_d[wmh_vox] - t_d[wmh_vox])))
    return vol, l1


def fig_wmh_boxplot(
    subject_results: Dict[str, List[Tuple[float, float]]],
    outpath: Path,
    dpi: int = 300,
    n_terciles: int = 3,
) -> None:
    """
    subject_results: {model_name: [(wmh_vol, wmh_l1), ...]} para todos los sujetos del test set.
    Stratifies into n_terciles according to wmh_vol and creates a boxplot of wmh_l1.
    """
    # Usar la lista del primer modelo para definir los terciles (mismos sujetos)
    first_key  = list(subject_results.keys())[0]
    all_vols   = np.array([r[0] for r in subject_results[first_key]])
    valid_mask = all_vols > 0

    if valid_mask.sum() < 6:
        print("  [WARN] Too few subjects with WMH for the box plot — skipping")
        return

    # Terciles sobre sujetos con WMH
    vols_valid  = all_vols[valid_mask]
    q33, q66    = np.percentile(vols_valid, [33, 66])
    labels_str  = [
        f"Low WMH\n(<{q33:.0f} mm³)",
        f"Medium WMH\n({q33:.0f}–{q66:.0f} mm³)",
        f"High WMH\n(>{q66:.0f} mm³)",
    ]

    def get_tercile(vol):
        if vol <= 0: return -1
        if vol < q33: return 0
        if vol < q66: return 1
        return 2

    model_names = list(subject_results.keys())
    n_models    = len(model_names)

    fig, axes = plt.subplots(1, n_terciles, figsize=(4 * n_terciles, 5),
                             sharey=True, squeeze=False)

    for t_idx in range(n_terciles):
        ax = axes[0, t_idx]
        data_per_model = []
        positions      = []

        for m_idx, model_name in enumerate(model_names):
            results = subject_results[model_name]
            wmh_l1s = [
                r[1] for i, r in enumerate(results)
                if valid_mask[i] and get_tercile(r[0]) == t_idx and np.isfinite(r[1])
            ]
            data_per_model.append(wmh_l1s)
            positions.append(m_idx + 1)

        bp = ax.boxplot(
            data_per_model,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2),
            flierprops=dict(marker="o", markersize=3.5, alpha=0.5),
            showfliers=True,
        )

        for patch, model_name in zip(bp["boxes"], model_names):
            st = MODEL_STYLES[model_name]
            patch.set_facecolor(st["color"])
            patch.set_alpha(0.75)

        # Puntos individuales
        for m_idx, (wmh_l1s, model_name) in enumerate(zip(data_per_model, model_names)):
            st = MODEL_STYLES[model_name]
            jitter = np.random.uniform(-0.12, 0.12, len(wmh_l1s))
            ax.scatter(
                [m_idx + 1 + j for j in jitter],
                wmh_l1s,
                color=st["color"], edgecolors="white", s=18, zorder=5, alpha=0.7,
            )

        ax.set_title(labels_str[t_idx], fontsize=10, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [nm.replace(" ", "\n") for nm in model_names], fontsize=8
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        # Number of subjects in each box
        for m_idx, wmh_l1s in enumerate(data_per_model):
            ax.text(m_idx + 1, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0,
                    f"n={len(wmh_l1s)}", ha="center", va="bottom",
                    fontsize=7, color="gray")

    axes[0, 0].set_ylabel("WMH-L1 ↓", fontsize=11)
    fig.suptitle("WMH-L1 by Lesion Load Tercile — Ablation Study",
                 fontsize=12, fontweight="bold", y=1.02)

    # Global legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=MODEL_STYLES[nm]["color"], alpha=0.75, label=nm)
        for nm in model_names
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=n_models,
               fontsize=9, bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [FIG] {outpath.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers para cargar el test set del Excel
# ──────────────────────────────────────────────────────────────────────────────

def load_test_split(excel_path: str, seed: int = 42) -> pd.DataFrame:
    """Reproduces the same 80/10/10 split used by the training script."""
    from sklearn.model_selection import train_test_split

    df = pd.read_excel(excel_path)
    df = df[df["T1_MNI"].apply(file_ok) &
            df["Flair_N4"].apply(file_ok) &
            df["Pseudolabel"].apply(file_ok)].copy()

    # Detectar columna de sujeto
    subj_col = None
    for c in ["Subject", "subject", "subject_id", "SubjectID", "RID", "rid", "ID", "id"]:
        if c in df.columns:
            subj_col = c
            break

    if subj_col:
        uniq = df[subj_col].astype(str).drop_duplicates().tolist()
        _, temp   = train_test_split(uniq, test_size=0.2, random_state=seed)
        _, te_ids = train_test_split(temp, test_size=0.5,  random_state=seed)
        return df[df[subj_col].astype(str).isin(te_ids)].copy()
    else:
        _, temp = train_test_split(df, test_size=0.2, random_state=seed)
        _, te   = train_test_split(temp, test_size=0.5, random_state=seed)
        return te.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generates figures for the paper: curves, intensity profile, WMH boxplot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Metric CSVs (required for the curves)
    p.add_argument("--csv_aa",   required=True, help="metrics.csv del modelo Anatomy-Aware")
    p.add_argument("--csv_unif", required=True, help="metrics.csv del modelo Uniform Weights")
    p.add_argument("--csv_base", required=True, help="metrics.csv del modelo Baseline")

    # Checkpoints (optional; needed for intensity profile and boxplot)
    p.add_argument("--ckpt_aa",   default=None, help="best.pt del modelo Anatomy-Aware")
    p.add_argument("--ckpt_unif", default=None, help="best.pt del modelo Uniform Weights")
    p.add_argument("--ckpt_base", default=None, help="best.pt del modelo Baseline")

    # Test subject for the intensity profile (alternative to test_excel)
    p.add_argument("--source",    default=None, help="T1 de un sujeto para el perfil de intensidad")
    p.add_argument("--target",    default=None, help="FLAIR correspondiente (ground truth)")
    p.add_argument("--mask",      default=None, help="8-class segmentation mask")

    # Full test set (for the lesion-load box plot)
    p.add_argument("--test_excel", default=None,
                   help="Excel del dataset para inferir sobre todo el test set (boxplot)")

    # Covariables (se usan si se pasa --source individual)
    p.add_argument("--age",    type=float, default=65.0)
    p.add_argument("--sex",    type=str,   default="M")
    p.add_argument("--domain", type=int,   default=0)

    # Arquitectura
    p.add_argument("--base_ch",  type=int, default=32)
    p.add_argument("--cond_dim", type=int, default=64)
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--overlap",  type=float, default=0.5)

    # Salida
    p.add_argument("--outdir", required=True)
    p.add_argument("--dpi",    type=int, default=300)
    p.add_argument("--seed",   type=int, default=42)

    # Control which figures to generate
    p.add_argument("--no_curves",  action="store_true", help="Omitir curvas de aprendizaje")
    p.add_argument("--no_profile", action="store_true", help="Omitir perfil de intensidad")
    p.add_argument("--no_boxplot", action="store_true", help="Omitir box plot WMH")

    # Boxplot parameter: maximum number of test-set subjects (for quick tests)
    p.add_argument("--max_subjects", type=int, default=None,
                   help="Maximum number of test-set subjects for the boxplot (None=all)")

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    csvs = {
        "Anatomy-Aware":   args.csv_aa,
        "Uniform Weights": args.csv_unif,
        "Baseline":        args.csv_base,
    }
    ckpts = {
        "Anatomy-Aware":   args.ckpt_aa,
        "Uniform Weights": args.ckpt_unif,
        "Baseline":        args.ckpt_base,
    }
    active_ckpts = {k: v for k, v in ckpts.items() if v is not None}

    # ── FIGURE 1: Learning curves ─────────────────────────────────────────────
    if not args.no_curves:
        print("\n[FIG 1] Learning curves...")
        fig_learning_curves(
            csvs=csvs,
            outpath=outdir / "learning_curves.png",
            dpi=args.dpi,
        )

    need_inference = (not args.no_profile or not args.no_boxplot) and active_ckpts

    if not need_inference:
        print("\nNo inference required. Done.")
        return

    # ── Cargar modelos ────────────────────────────────────────────────────────
    print(f"\n[INFO] Cargando {len(active_ckpts)} modelos...")
    models = {}
    for name, ckpt_path in active_ckpts.items():
        print(f"  → {name}")
        models[name] = load_model(ckpt_path, args.base_ch, args.cond_dim, device)

    def infer_subject(src_norm, cov_arr, domain_idx):
        preds = {}
        for name, model in models.items():
            cov    = torch.tensor([cov_arr], dtype=torch.float32).to(device)
            domain = torch.tensor([domain_idx], dtype=torch.long).to(device)
            if domain_idx >= model.domain_emb.num_embeddings:
                domain = torch.tensor([0], dtype=torch.long).to(device)
            preds[name] = run_inference(
                model, src_norm, cov, domain,
                tuple(args.patch_size), args.overlap, device
            )
        return preds

    # ── FIGURE 2: Intensity profile ──────────────────────────────────────────
    if not args.no_profile:
        print("\n[FIG 2] Intensity profile...")

        if args.source and args.target and args.mask:
            src_raw  = load_nifti(args.source)
            tgt_raw  = load_nifti(args.target)
            mask_vol = load_nifti(args.mask).astype(np.int64)
            support  = (np.abs(src_raw) > 1e-8).astype(np.uint8)
            src_norm = robust_normalize(src_raw, mask=support)
            tgt_norm = robust_normalize(tgt_raw, mask=(np.abs(tgt_raw) > 1e-8).astype(np.uint8))

            cov_arr = np.array([
                np.clip(args.age, 0, 120) / 100.0,
                encode_sex(args.sex),
                0.0,  # direction T1→FLAIR
            ], dtype=np.float32)

            preds = infer_subject(src_norm, cov_arr, args.domain)

            fig_intensity_profile(
                source_norm=src_norm,
                target_norm=tgt_norm,
                mask=mask_vol,
                preds=preds,
                outpath=outdir / "intensity_profile.png",
                dpi=args.dpi,
            )
        else:
            print("  [WARN] The intensity profile requires --source, --target, and --mask. Skipping.")

    # ── FIGURE 3: Box plot by lesion load ─────────────────────────────────────
    if not args.no_boxplot:
        print("\n[FIG 3] WMH box plot by lesion load...")

        if not args.test_excel:
            print("  [WARN] The box plot requires --test_excel. Skipping.")
        else:
            test_df = load_test_split(args.test_excel, seed=args.seed)
            if args.max_subjects:
                test_df = test_df.sample(min(args.max_subjects, len(test_df)),
                                         random_state=args.seed)
            print(f"  [INFO] Subjects in test set: {len(test_df)}")

            subject_results: Dict[str, List[Tuple[float, float]]] = {
                k: [] for k in models
            }

            for i, (_, row) in enumerate(test_df.iterrows()):
                try:
                    src_raw  = load_nifti(str(row["T1_MNI"]))
                    tgt_raw  = load_nifti(str(row["Flair_N4"]))
                    mask_vol = load_nifti(str(row["Pseudolabel"])).astype(np.int64)
                    support  = (np.abs(src_raw) > 1e-8).astype(np.uint8)
                    src_norm = robust_normalize(src_raw, mask=support)
                    tgt_norm = robust_normalize(tgt_raw,
                                               mask=(np.abs(tgt_raw) > 1e-8).astype(np.uint8))

                    age = float(row["Age"]) if "Age" in row and not pd.isna(row["Age"]) else 65.0
                    sex = str(row.get("Sex", "M"))
                    dom = int(row["cluster"]) if "cluster" in row and not pd.isna(row.get("cluster")) else 0

                    cov_arr = np.array([
                        np.clip(age, 0, 120) / 100.0,
                        encode_sex(sex),
                        0.0,
                    ], dtype=np.float32)

                    preds = infer_subject(src_norm, cov_arr, dom)

                    for model_name, pred in preds.items():
                        vol, wmh_l1 = _compute_wmh_l1_subject(pred, tgt_norm, mask_vol)
                        subject_results[model_name].append((vol, wmh_l1))

                    if (i + 1) % 10 == 0:
                        print(f"  [{i+1}/{len(test_df)}] processed...")

                except Exception as e:
                    print(f"  [WARN] Subject {i} failed: {e}")
                    for model_name in models:
                        subject_results[model_name].append((0.0, float("nan")))

            fig_wmh_boxplot(
                subject_results=subject_results,
                outpath=outdir / "wmh_boxplot.png",
                dpi=args.dpi,
            )

    print(f"\nFigures saved to: {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
