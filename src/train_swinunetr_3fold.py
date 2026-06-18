#!/usr/bin/env python3
"""
train_swinunetr_3fold.py — Training with hybrid SwinUNETR + AA-loss, 3-fold CV

Architecture:
  MONAI's SwinUNETR (Swin Transformer encoder + convolutional decoder with skip connections).
  With feature_size=24 and depths=(2,2,2,2) it fits in 8GB VRAM with 96³ patch, batch 2-4.

3-fold cross-validation:
  The dataset is divided into 3 folds at the subject level (no data leakage).
  Each fold uses 2/3 for train, 1/6 for val, 1/6 for test.
  Reports metrics per fold + mean ± std at the end.

Usage:
    python train_swinunetr_3fold.py \
        --excel    ./dataset.xlsx \
        --outdir   ./exp_swin_3fold \
        --epochs   100 \
        --batch_size 1 \
        --use_amp \
        --bidirectional

Ablation flags (same as UNet):
    --no_mask          Baseline without mask
    --uniform_weights  Uniform weights

Requires:
    pip install monai[all] timm einops
"""

from __future__ import annotations

import argparse
import math
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from monai.networks.nets import SwinUNETR
    from monai.inferers import SlidingWindowInferer
except ImportError:
    raise ImportError("Install monai: pip install 'monai[all]'")

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ──────────────────────────────────────────────────────────────────────────────
# Anatomical weighting
# ──────────────────────────────────────────────────────────────────────────────

REGION_WEIGHTS  = torch.tensor([0.1, 1.0, 1.5, 1.0, 1.0, 1.2, 1.2, 8.0], dtype=torch.float32)
UNIFORM_WEIGHTS = torch.ones(8, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def file_ok(p) -> bool:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return False
    return Path(str(p)).exists()

def load_nifti(path: str) -> np.ndarray:
    arr = nib.load(path).get_fdata(dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"3D input expected, obtained {arr.shape}")
    return arr

def robust_normalize(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    vals = x[(mask > 0) & np.isfinite(x)] if mask is not None else x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    med   = np.median(vals)
    mad   = np.median(np.abs(vals - med))
    scale = 1.4826 * mad if mad > 1e-6 else (float(np.std(vals)) if np.std(vals) > 1e-6 else 1.0)
    return np.clip((x - med) / scale, -5.0, 5.0).astype(np.float32)

def exists_nonempty(x) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    return len(str(x).strip()) > 0 and str(x).strip().lower() != "nan"


# ──────────────────────────────────────────────────────────────────────────────
# Anatomy-Aware Weight Map
# ──────────────────────────────────────────────────────────────────────────────

def build_weight_map(mask: torch.Tensor, region_weights: torch.Tensor) -> torch.Tensor:
    rw = region_weights.to(mask.device)
    w  = rw[mask]
    return w.unsqueeze(1).float()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def get_lesion_aware_crop(label: np.ndarray, patch_size: Tuple,
                           wmh_bias: float = 0.5) -> Tuple[int, int, int]:
    d, h, w = label.shape
    pd, ph, pw = patch_size
    rv = random.random()
    if rv < wmh_bias:
        target = 7
    elif rv < wmh_bias + 0.2:
        target = 2
    else:
        target = -1
    coords = np.argwhere(label == target) if target != -1 else []
    if len(coords) == 0:
        coords = np.argwhere(label > 0)
    if len(coords) > 0:
        cz, cy, cx = coords[np.random.randint(0, len(coords))]
        z0 = int(np.clip(cz - pd // 2, 0, d - pd))
        y0 = int(np.clip(cy - ph // 2, 0, h - ph))
        x0 = int(np.clip(cx - pw // 2, 0, w - pw))
    else:
        z0 = np.random.randint(0, max(1, d - pd + 1))
        y0 = np.random.randint(0, max(1, h - ph + 1))
        x0 = np.random.randint(0, max(1, w - pw + 1))
    return z0, y0, x0

def crop_3d(vol: np.ndarray, coords: Tuple, patch_size: Tuple) -> np.ndarray:
    z0, y0, x0 = coords
    pd, ph, pw = patch_size
    return vol[z0:z0+pd, y0:y0+ph, x0:x0+pw]

def random_augment(a, b, c):
    for axis in range(3):
        if random.random() < 0.5:
            a = np.flip(a, axis=axis).copy()
            b = np.flip(b, axis=axis).copy()
            c = np.flip(c, axis=axis).copy()
    if random.random() < 0.3:
        a = np.clip(a + np.random.normal(0, 0.05, a.shape).astype(np.float32), -5, 5)
    return a, b, c


class MRITranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, patch_size=(96, 96, 96),
                 train: bool = True, bidirectional: bool = False,
                 no_mask: bool = False):
        self.df = df.reset_index(drop=True)
        self.patch_size = patch_size
        self.train = train
        self.bidirectional = bidirectional
        self.no_mask = no_mask
        if bidirectional:
            self.indices = [(i, 0) for i in range(len(df))] + \
                           [(i, 1) for i in range(len(df))]
        else:
            self.indices = [(i, 0) for i in range(len(df))]

    def __len__(self):
        return len(self.indices)

    def _encode_sex(self, val) -> float:
        s = str(val).strip().lower()
        return 1.0 if s in {"m", "male", "1", "1.0", "man"} else 0.0

    def __getitem__(self, idx):
        row_idx, direction = self.indices[idx]
        row = self.df.iloc[row_idx]

        t1    = load_nifti(str(row["T1_MNI"]))
        flair = load_nifti(str(row["Flair_N4"]))
        label = load_nifti(str(row["Pseudolabel"]))

        support = ((np.abs(t1) > 1e-8) | (np.abs(flair) > 1e-8)).astype(np.uint8)
        t1    = robust_normalize(t1,    mask=support)
        flair = robust_normalize(flair, mask=support)

        age = float(row["Age"]) if exists_nonempty(row.get("Age")) else 50.0
        cov = np.array([
            np.clip(age, 0.0, 120.0) / 100.0,
            self._encode_sex(row.get("Sex", 0.0)),
            float(direction),
        ], dtype=np.float32)

        cluster_val = row.get("cluster", 0)
        domain = int(cluster_val) if not (isinstance(cluster_val, float) and np.isnan(cluster_val)) else 0

        src_vol = t1    if direction == 0 else flair
        tgt_vol = flair if direction == 0 else t1

        if self.train:
            if self.no_mask:
                d_, h_, w_ = src_vol.shape
                pd_, ph_, pw_ = self.patch_size
                z0 = np.random.randint(0, max(1, d_ - pd_ + 1))
                y0 = np.random.randint(0, max(1, h_ - ph_ + 1))
                x0 = np.random.randint(0, max(1, w_ - pw_ + 1))
                coords = (z0, y0, x0)
            else:
                coords = get_lesion_aware_crop(label, self.patch_size)
            sp = crop_3d(src_vol, coords, self.patch_size)
            tp = crop_3d(tgt_vol, coords, self.patch_size)
            lp = crop_3d(label,   coords, self.patch_size)
            sp, tp, lp = random_augment(sp, tp, lp)
            return {
                "source":    torch.from_numpy(sp[None].astype(np.float32)),
                "target":    torch.from_numpy(tp[None].astype(np.float32)),
                "mask":      torch.from_numpy(lp.astype(np.int64)),
                "cov":       torch.from_numpy(cov),
                "domain":    torch.tensor(domain, dtype=torch.long),
                "direction": torch.tensor(direction, dtype=torch.long),
            }
        else:
            return {
                "source":    torch.from_numpy(src_vol[None].astype(np.float32)),
                "target":    torch.from_numpy(tgt_vol[None].astype(np.float32)),
                "mask":      torch.from_numpy(label.astype(np.int64)),
                "cov":       torch.from_numpy(cov),
                "domain":    torch.tensor(domain, dtype=torch.long),
                "direction": torch.tensor(direction, dtype=torch.long),
            }


# ──────────────────────────────────────────────────────────────────────────────
# Model: Hybrid SwinUNETR with FiLM conditioning
# ──────────────────────────────────────────────────────────────────────────────

class FiLMConditioner(nn.Module):
    """
    Generates gammas and betas to modulate features of the SwinUNETR decoder. 
    Receives covariates + domain embedding and produces a conditioning vector.
    """
    def __init__(self, n_domains: int = 4, n_cov: int = 3, cond_dim: int = 64):
        super().__init__()
        self.domain_emb = nn.Embedding(n_domains, 16)
        self.cov_mlp    = nn.Sequential(
            nn.Linear(n_cov, 32), nn.GELU(), nn.Linear(32, 32)
        )
        self.proj = nn.Sequential(
            nn.Linear(48, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, cov: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([self.domain_emb(domain), self.cov_mlp(cov)], dim=-1))


class FiLMLayer(nn.Module):
    """Apply FiLM modulation to a 3D feature tensor."""
    def __init__(self, n_channels: int, cond_dim: int):
        super().__init__()
        self.film = nn.Linear(cond_dim, 2 * n_channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return x * (1.0 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]


class SwinUNETRTranslator(nn.Module):
    """
    SwinUNETR with:
      - Input:  1 channel (source modality)
      - Output: 1 channel (target modality)
      - FiLM conditioning in the decoder head
      - feature_size=24 for 8GB VRAM with 96³ patches

    The SwinUNETR was designed for segmentation (n classes), 
    here we adapt it for image regression by replacing the segmentation 
    head with a synthesis head.
    """
    def __init__(self, img_size: Tuple[int, int, int] = (96, 96, 96),
                 feature_size: int = 24,
                 n_domains: int = 4,
                 cond_dim: int = 64,
                 depths: Tuple[int, ...] = (2, 2, 2, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24),
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0):
        super().__init__()

        # ── Encoder/Decoder SwinUNETR (output = feature_size channels) ───────
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=feature_size,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=True,         # gradient checkpointing → VRAM
        )

        # ── FiLM conditioning ─────────────────────────────────────────────
        self.conditioner = FiLMConditioner(n_domains=n_domains, n_cov=3, cond_dim=cond_dim)
        self.film        = FiLMLayer(n_channels=feature_size, cond_dim=cond_dim)

        # ── Synthesis head (replaces segmentation one) ─────────────
        self.synth_head = nn.Sequential(
            nn.GroupNorm(min(8, feature_size), feature_size),
            nn.GELU(),
            nn.Conv3d(feature_size, feature_size // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(feature_size // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor,
                cov: torch.Tensor,
                domain: torch.Tensor) -> torch.Tensor:
        cond     = self.conditioner(cov, domain)
        features = self.backbone(x)          # [B, feature_size, D, H, W]
        features = self.film(features, cond) # FiLM modulation
        return self.synth_head(features)     # [B, 1, D, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# Losses 
# ──────────────────────────────────────────────────────────────────────────────

def stable_ssim_loss(pred: torch.Tensor, target: torch.Tensor,
                     window_size: int = 7) -> torch.Tensor:
    def _norm(t):
        mn = t.flatten(1).min(1)[0][:, None, None, None, None]
        mx = t.flatten(1).max(1)[0][:, None, None, None, None]
        return 2.0 * (t - mn) / (mx - mn).clamp(min=1e-5) - 1.0

    p, t = _norm(pred), _norm(target)
    mu1  = F.avg_pool3d(p, window_size, stride=1, padding=window_size // 2)
    mu2  = F.avg_pool3d(t, window_size, stride=1, padding=window_size // 2)
    s1   = (F.avg_pool3d(p*p, window_size, stride=1, padding=window_size//2) - mu1**2).clamp(0)
    s2   = (F.avg_pool3d(t*t, window_size, stride=1, padding=window_size//2) - mu2**2).clamp(0)
    s12  =  F.avg_pool3d(p*t, window_size, stride=1, padding=window_size//2) - mu1 * mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return 1.0 - ssim.mean()

def gradient_loss_3d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for d in (2, 3, 4):
        dp = torch.diff(pred,   dim=d)
        dt = torch.diff(target, dim=d)
        if dp.numel() > 0:
            losses.append(F.l1_loss(dp, dt))
    return sum(losses) / len(losses) if losses else pred.new_tensor(0.0)

def anatomy_aware_loss(pred, target, mask, region_weights,
                       lambda_ssim=1.5, lambda_grad=0.5):
    w       = build_weight_map(mask, region_weights)
    l1_map  = torch.abs(pred - target)
    l1_loss = (l1_map * w).sum() / w.sum().clamp(min=1.0)
    ssim_l  = stable_ssim_loss(pred, target)
    grad_l  = gradient_loss_3d(pred, target)
    total   = l1_loss + lambda_ssim * ssim_l + lambda_grad * grad_l
    return total, {"loss": total.item(), "l1": l1_loss.item(),
                   "ssim_loss": ssim_l.item(), "grad_loss": grad_l.item()}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target)**2))
    return 20 * math.log10(1.0 + 1e-8) - 10 * math.log10(mse + 1e-8)

def ssim2d_average(pred: np.ndarray, target: np.ndarray) -> float:
    if not HAS_SKIMAGE:
        return float("nan")
    vals = []
    for axis in range(3):
        mid  = pred.shape[axis] // 2
        sp   = np.take(pred,   mid, axis=axis)
        st   = np.take(target, mid, axis=axis)
        lo   = min(np.percentile(sp, 1), np.percentile(st, 1))
        hi   = max(np.percentile(sp, 99), np.percentile(st, 99))
        if hi - lo < 1e-6: continue
        sp = np.clip((sp - lo) / (hi - lo), 0, 1)
        st = np.clip((st - lo) / (hi - lo), 0, 1)
        ws = 7 if min(sp.shape) >= 7 else 3
        try:
            vals.append(float(skimage_ssim(sp, st, data_range=1.0, win_size=ws)))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, args, region_weights):
    model.eval()
    inferer = SlidingWindowInferer(
        roi_size=tuple(args.patch_size), sw_batch_size=1,
        overlap=0.5, mode="gaussian"
    )
    totals    = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0}
    wmh_accum = 0.0
    wmh_count = 0
    n = 0

    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        cov    = batch["cov"].to(device)
        domain = batch["domain"].to(device)
        mask   = batch["mask"].to(device)

        pred = inferer(source, lambda x: model(x, cov, domain))
        pred = torch.clamp(pred, -5.0, 5.0)

        totals["l1"] += F.l1_loss(pred, target).item()

        wmh_m = (mask == 7).unsqueeze(1).float()
        n_wmh = wmh_m.sum().item()
        if n_wmh > 0:
            wmh_accum += (torch.abs(pred - target) * wmh_m).sum().item() / n_wmh
            wmh_count += 1

        pn  = pred.cpu().numpy()
        tn  = target.cpu().numpy()
        for i in range(pn.shape[0]):
            lo = min(np.percentile(pn[i,0],1), np.percentile(tn[i,0],1))
            hi = max(np.percentile(pn[i,0],99), np.percentile(tn[i,0],99))
            if hi - lo >= 1e-6:
                p1 = np.clip((pn[i,0]-lo)/(hi-lo), 0, 1)
                t1 = np.clip((tn[i,0]-lo)/(hi-lo), 0, 1)
                totals["psnr"] += psnr(p1, t1)
                s = ssim2d_average(p1, t1)
                if np.isfinite(s): totals["ssim"] += s
        n += source.shape[0]

    n = max(n, 1)
    result = {k: v / n for k, v in totals.items()}
    result["wmh_l1"] = wmh_accum / wmh_count if wmh_count > 0 else float("nan")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# QC plots
# ──────────────────────────────────────────────────────────────────────────────

def save_qc(model, loader, device, epoch, out_dir, args):
    model.eval()
    inferer = SlidingWindowInferer(
        roi_size=tuple(args.patch_size), sw_batch_size=1, overlap=0.5, mode="gaussian"
    )
    for batch in loader:
        src = batch["source"].to(device)
        tgt = batch["target"].to(device)
        cov = batch["cov"].to(device)
        dom = batch["domain"].to(device)
        msk = batch["mask"][0].cpu().numpy()
        with torch.no_grad():
            pred = inferer(src, lambda x: model(x, cov, dom))
        s = src[0,0].cpu().numpy(); t = tgt[0,0].cpu().numpy(); p = pred[0,0].cpu().numpy()
        z = s.shape[0] // 2
        vmin, vmax = np.percentile(t[z], 1), np.percentile(t[z], 99)
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(s[z], cmap="gray");                           axes[0].set_title("Source")
        axes[1].imshow(t[z], cmap="gray", vmin=vmin, vmax=vmax);    axes[1].set_title("Target (GT)")
        axes[2].imshow(p[z], cmap="gray", vmin=vmin, vmax=vmax);    axes[2].set_title("Prediction")
        axes[3].imshow(np.abs(t[z]-p[z]), cmap="hot");              axes[3].set_title("Abs. Error")
        axes[4].imshow(msk[z], cmap="tab10", vmin=0, vmax=7);       axes[4].set_title("Mask")
        for ax in axes: ax.axis("off")
        plt.suptitle(f"Epoch {epoch:03d}", y=1.02)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"qc_epoch_{epoch:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()
        break


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_fold(fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
               test_df: pd.DataFrame, args, device: torch.device,
               n_domains: int) -> Dict[str, float]:

    fold_outdir = Path(args.outdir) / f"fold_{fold_idx}"
    ckpt_dir    = fold_outdir / "checkpoints"
    qc_dir      = fold_outdir / "qc"
    for d in [fold_outdir, ckpt_dir, qc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}  |  train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"{'='*60}")

    # Datasets
    train_ds = MRITranslationDataset(train_df, patch_size=tuple(args.patch_size),
                                     train=True, bidirectional=args.bidirectional,
                                     no_mask=args.no_mask)
    val_ds   = MRITranslationDataset(val_df, patch_size=tuple(args.patch_size),
                                     train=False, no_mask=args.no_mask)
    test_ds  = MRITranslationDataset(test_df, patch_size=tuple(args.patch_size),
                                     train=False, no_mask=args.no_mask)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Modelo
    model = SwinUNETRTranslator(
        img_size=tuple(args.patch_size),
        feature_size=args.feature_size,
        n_domains=n_domains,
        cond_dim=args.cond_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {n_params/1e6:.2f}M")

    # Weights
    if args.no_mask:
        region_weights = UNIFORM_WEIGHTS.to(device)
        print("[INFO] Mode: Baseline (no mask)")
    elif args.uniform_weights:
        region_weights = UNIFORM_WEIGHTS.to(device)
        print("[INFO] Mode: Uniform weights")
    else:
        region_weights = REGION_WEIGHTS.to(device)
        print("[INFO] Mode: Anatomy-Aware")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    warmup_epochs = max(5, args.epochs // 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    metrics_path = fold_outdir / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("epoch,lr,train_loss,train_l1,val_l1,val_psnr,val_ssim,val_wmh_l1\n")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_stats = {"loss": 0.0, "l1": 0.0}

        for batch in train_loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            mask   = batch["mask"].to(device)
            cov    = batch["cov"].to(device)
            domain = batch["domain"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                pred = model(source, cov, domain)
                loss, stats = anatomy_aware_loss(
                    pred, target, mask, region_weights,
                    args.lambda_ssim, args.lambda_grad
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for k in epoch_stats:
                epoch_stats[k] += stats.get(k, 0.0)

        scheduler.step()
        nb = max(len(train_loader), 1)
        epoch_stats = {k: v/nb for k, v in epoch_stats.items()}
        val_m = evaluate(model, val_loader, device, args, region_weights)
        lr    = scheduler.get_last_lr()[0]

        wmh_str = f"{val_m['wmh_l1']:.4f}" if np.isfinite(val_m['wmh_l1']) else "n/a"
        print(
            f"[Fold {fold_idx} Ep {epoch:03d}/{args.epochs}] "
            f"LR={lr:.2e} | TrainL1={epoch_stats['l1']:.4f} | "
            f"ValL1={val_m['l1']:.4f} PSNR={val_m['psnr']:.2f} "
            f"WMH-L1={wmh_str}"
        )

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{lr:.6f},{epoch_stats['loss']:.5f},{epoch_stats['l1']:.5f},"
                    f"{val_m['l1']:.5f},{val_m['psnr']:.4f},{val_m['ssim']:.4f},"
                    f"{val_m['wmh_l1'] if np.isfinite(val_m['wmh_l1']) else 'nan'}\n")

        if epoch % args.qc_every == 0 or epoch == args.epochs:
            save_qc(model, val_loader, device, epoch, qc_dir, args)

        # Save checkpoint
        torch.save({
            "epoch": epoch, "fold": fold_idx,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict() if scaler.is_enabled() else {},
            "best_val":        best_val,
            "n_domains":       n_domains,
            "feature_size":    args.feature_size,
        }, ckpt_dir / "last.pt")

        if val_m["l1"] < best_val:
            best_val = val_m["l1"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  ↳ Nuevo mejor modelo (val_l1={best_val:.4f})")

    # Test set evaluation with the best model
    print(f"\n[Fold {fold_idx}] Evaluating test set with best.pt...")
    best_state = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, device, args, region_weights)
    test_m["fold"] = fold_idx

    print(f"[Fold {fold_idx}] TEST → "
          f"L1={test_m['l1']:.4f} PSNR={test_m['psnr']:.2f} "
          f"SSIM={test_m['ssim']:.4f} WMH-L1={test_m.get('wmh_l1', float('nan')):.4f}")

    with open(fold_outdir / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)

    return test_m


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Hybrid SwinUNETR + AA-loss with 3-fold CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--excel",        required=True)
    p.add_argument("--cluster_csv",  default=None)
    p.add_argument("--cluster_col",  default="cluster")
    p.add_argument("--outdir",       default="./swin_3fold")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--use_amp",      action="store_true")

    # SwinUNETR
    p.add_argument("--feature_size", type=int,   default=24)
    p.add_argument("--cond_dim",     type=int,   default=64)
    p.add_argument("--depths",       type=int,   nargs=4, default=[2, 2, 2, 2])
    p.add_argument("--num_heads",    type=int,   nargs=4, default=[3, 6, 12, 24],
                   help="Attention heads")

    # Patches
    p.add_argument("--patch_size",   type=int,   nargs=3, default=[96, 96, 96])

    # Loss
    p.add_argument("--lambda_ssim",  type=float, default=1.5)
    p.add_argument("--lambda_grad",  type=float, default=0.5)

    # 3-fold CV
    p.add_argument("--n_folds",      type=int,   default=3)
    p.add_argument("--fold",         type=int,   default=None,
                   help="Just one fold training (None=all)")

    # Ablation
    p.add_argument("--no_mask",        action="store_true")
    p.add_argument("--uniform_weights",action="store_true")
    p.add_argument("--bidirectional",  action="store_true")
    p.add_argument("--qc_every",       type=int, default=10)

    args = p.parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Data loading ────────────────────────────────────────────────
    df = pd.read_excel(args.excel)
    df = df[df["T1_MNI"].apply(file_ok) &
            df["Flair_N4"].apply(file_ok) &
            df["Pseudolabel"].apply(file_ok)].copy()

    if args.cluster_csv:
        cdf = pd.read_csv(args.cluster_csv)[["Flair_N4", args.cluster_col]]
        cdf["Flair_N4"] = cdf["Flair_N4"].astype(str)
        df["Flair_N4"]  = df["Flair_N4"].astype(str)
        df = df.merge(cdf, on="Flair_N4", how="inner")
    else:
        df["cluster"] = 0

    n_domains = int(df["cluster"].max()) + 2

    # ── Filter unique subjects to prevent data leakage ──────────────────────────────────
    subj_col = None
    for c in ["Subject", "subject", "subject_id", "SubjectID", "RID", "rid", "ID", "id"]:
        if c in df.columns:
            subj_col = c
            break

    if subj_col:
        subjects = df[subj_col].astype(str).drop_duplicates().values
    else:
        subjects = df.index.astype(str).values

    print(f"[INFO] Unique subjects: {len(subjects)} | Domains: {n_domains}")

    # ── 3-Fold CV subject level ───────────────────────────────────────────
    kf      = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds   = list(kf.split(subjects))
    all_test_metrics = []

    for fold_idx, (dev_idx, test_idx) in enumerate(folds):
        if args.fold is not None and fold_idx != args.fold:
            continue

        dev_subjects  = subjects[dev_idx]
        test_subjects = subjects[test_idx]

        n_val  = max(1, len(dev_subjects) // 6)
        rng    = np.random.default_rng(args.seed + fold_idx)
        val_mask = np.zeros(len(dev_subjects), dtype=bool)
        val_mask[rng.choice(len(dev_subjects), n_val, replace=False)] = True

        train_subjects = dev_subjects[~val_mask]
        val_subjects   = dev_subjects[val_mask]

        if subj_col:
            train_df = df[df[subj_col].astype(str).isin(train_subjects)].copy()
            val_df   = df[df[subj_col].astype(str).isin(val_subjects)].copy()
            test_df  = df[df[subj_col].astype(str).isin(test_subjects)].copy()
        else:
            train_df = df.loc[df.index.astype(str).isin(train_subjects)].copy()
            val_df   = df.loc[df.index.astype(str).isin(val_subjects)].copy()
            test_df  = df.loc[df.index.astype(str).isin(test_subjects)].copy()

        test_m = train_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            args=args,
            device=device,
            n_domains=n_domains,
        )
        all_test_metrics.append(test_m)

    # ── Summary ─────────────────────────────────────────────────────────
    if len(all_test_metrics) > 1:
        print(f"\n{'='*60}")
        print(f"3-FOLD CV")
        print(f"{'='*60}")
        for metric in ["l1", "psnr", "ssim", "wmh_l1"]:
            vals = [m[metric] for m in all_test_metrics if np.isfinite(m.get(metric, float("nan")))]
            if vals:
                mean_v = np.mean(vals)
                std_v  = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                print(f"  {metric.upper():10s}: {mean_v:.4f} ± {std_v:.4f}")

        summary = {
            "folds": all_test_metrics,
            "mean": {k: float(np.nanmean([m.get(k, float("nan")) for m in all_test_metrics]))
                     for k in ["l1", "psnr", "ssim", "wmh_l1"]},
            "std":  {k: float(np.nanstd([m.get(k, float("nan")) for m in all_test_metrics], ddof=1))
                     for k in ["l1", "psnr", "ssim", "wmh_l1"]},
        }
        with open(outdir / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved in: {outdir / 'cv_summary.json'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
