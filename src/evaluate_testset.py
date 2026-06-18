#!/usr/bin/env python3
"""
evaluate_testset.py — Computes quantitative metrics on the full test set.

Produces:
  - metrics_summary.csv   : mean ± std per model (for the paper table)
  - metrics_per_subject.csv : individual metrics (for boxplots or additional analysis)
  - metrics_table.txt     : formatted table ready to copy into the paper

Usage:
    python evaluate_testset.py \
        --test_excel "C:/ruta/dataset.xlsx" \
        --ckpt_aa   "C:/t1_to_flair_v4_ft/checkpoints/best.pt" \
        --ckpt_unif "C:/t1_to_flair_v5/checkpoints/best.pt" \
        --ckpt_base "C:/t1_to_flair_v7/checkpoints/best.pt" \
        --outdir    ./metrics_results \
        --base_ch 32 --cond_dim 64 --patch_size 96 96 96

Optional arguments:
    --max_subjects N   Limit to N subjects (useful for quick tests)
    --seed 42          Seed for split reproducibility
    --overlap 0.5      Sliding-window overlap
    --no_amp           Disable AMP if precision issues arise
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn

try:
    from monai.inferers import SlidingWindowInferer
except ImportError:
    sys.exit("ERROR: instala monai → pip install monai")

try:
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[WARN] scikit-image not available — SSIM will be NaN")

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("ERROR: instala scikit-learn → pip install scikit-learn")


# ──────────────────────────────────────────────────────────────────────────────
# Architecture (identical to the training script)
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
        self.bottleneck = nn.Sequential(
            FiLMBlock3D(C*8, C*8, cond_dim),
            FiLMBlock3D(C*8, C*8, cond_dim)
        )
        self.up4 = UpBlock(C*8, C*8, C*8, cond_dim)
        self.up3 = UpBlock(C*8, C*4, C*4, cond_dim)
        self.up2 = UpBlock(C*4, C*2, C*2, cond_dim)
        self.up1 = UpBlock(C*2, C,   C,   cond_dim)
        self.head = nn.Sequential(
            nn.Conv3d(C, C, 3, padding=1), nn.GELU(), nn.Conv3d(C, 1, 1)
        )

    def make_cond(self, cov, domain):
        return self.cond_proj(
            torch.cat([self.domain_emb(domain), self.cov_mlp(cov)], dim=-1)
        )

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
# I/O and normalization
# ──────────────────────────────────────────────────────────────────────────────

def file_ok(p) -> bool:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return False
    return Path(str(p)).exists()

def load_nifti(path: str) -> np.ndarray:
    arr = nib.load(path).get_fdata(dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Esperaba 3D, obtuvo {arr.shape}")
    return arr

def robust_normalize(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    vals = x[(mask > 0) & np.isfinite(x)] if mask is not None else x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    med   = np.median(vals)
    mad   = np.median(np.abs(vals - med))
    scale = 1.4826 * mad if mad > 1e-6 else (float(np.std(vals)) if np.std(vals) > 1e-6 else 1.0)
    return np.clip((x - med) / scale, -5.0, 5.0).astype(np.float32)

def encode_sex(s: str) -> float:
    return 1.0 if str(s).strip().lower() in {"m", "male", "1", "man"} else 0.0

def load_model(ckpt_path: str, base_ch: int, cond_dim: int,
               device: torch.device) -> RegionConditionedUNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
        n_dom = ckpt.get("n_domains", state["domain_emb.weight"].shape[0])
    else:
        state = ckpt
        n_dom = state["domain_emb.weight"].shape[0]
    model = RegionConditionedUNet(
        base_ch=base_ch, cond_dim=cond_dim, n_domains=n_dom, n_cov=3
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    pred:   np.ndarray,
    target: np.ndarray,
    mask:   np.ndarray,
) -> Dict[str, float]:
    """
    Computes L1, PSNR, SSIM, and WMH-L1.

    Normalization: pred and target are mapped to [0,1] using the statistics
    of the TARGET (1st-99th percentile). This anchors both volumes to the same
    clinical reference scale and ensures comparability across subjects.
    """
    lo  = float(np.percentile(target, 1))
    hi  = float(np.percentile(target, 99))
    rng = hi - lo

    if rng < 1e-6:
        return {"L1": float("nan"), "PSNR": float("nan"),
                "SSIM": float("nan"), "WMH_L1": float("nan")}

    p_n = np.clip((pred   - lo) / rng, 0.0, 1.0).astype(np.float32)
    t_n = np.clip((target - lo) / rng, 0.0, 1.0).astype(np.float32)

    # L1 global
    l1 = float(np.mean(np.abs(p_n - t_n)))

    # PSNR
    mse  = float(np.mean((p_n - t_n) ** 2))
    psnr = 20.0 * math.log10(1.0 + 1e-8) - 10.0 * math.log10(mse + 1e-8)

    # SSIM: average of central slices along the 3 axes
    ssim = float("nan")
    if HAS_SKIMAGE:
        vals = []
        for axis in range(3):
            mid  = p_n.shape[axis] // 2
            sl_p = np.take(p_n, mid, axis=axis)
            sl_t = np.take(t_n, mid, axis=axis)
            ws   = 7 if min(sl_p.shape) >= 7 else 3
            try:
                vals.append(float(sk_ssim(sl_p, sl_t, data_range=1.0, win_size=ws)))
            except Exception:
                pass
        if vals:
            ssim = float(np.mean(vals))

    # WMH-L1: L1 restricted to class-7 voxels, same normalization
    wmh_vox = mask == 7
    wmh_l1  = float("nan")
    if wmh_vox.sum() > 0:
        wmh_l1 = float(np.mean(np.abs(p_n[wmh_vox] - t_n[wmh_vox])))

    return {"L1": l1, "PSNR": psnr, "SSIM": ssim, "WMH_L1": wmh_l1}


# ──────────────────────────────────────────────────────────────────────────────
# Reproducible split (identical to training)
# ──────────────────────────────────────────────────────────────────────────────

def load_test_split(excel_path: str, seed: int = 42) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    df = df[
        df["T1_MNI"].apply(file_ok) &
        df["Flair_N4"].apply(file_ok) &
        df["Pseudolabel"].apply(file_ok)
    ].copy().reset_index(drop=True)

    subj_col = None
    for c in ["Subject", "subject", "subject_id", "SubjectID", "RID", "rid", "ID", "id"]:
        if c in df.columns:
            subj_col = c
            break

    if subj_col:
        uniq = df[subj_col].astype(str).drop_duplicates().tolist()
        _, temp   = train_test_split(uniq, test_size=0.2, random_state=seed)
        _, te_ids = train_test_split(temp,  test_size=0.5, random_state=seed)
        return df[df[subj_col].astype(str).isin(te_ids)].copy().reset_index(drop=True)
    else:
        _, temp = train_test_split(df, test_size=0.2, random_state=seed)
        _, te   = train_test_split(temp, test_size=0.5, random_state=seed)
        return te.copy().reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:       RegionConditionedUNet,
    source_norm: np.ndarray,
    cov:         torch.Tensor,
    domain:      torch.Tensor,
    patch_size:  Tuple[int, int, int],
    overlap:     float,
    device:      torch.device,
    use_amp:     bool,
) -> np.ndarray:
    inferer = SlidingWindowInferer(
        roi_size=patch_size, sw_batch_size=2,
        overlap=overlap, mode="gaussian"
    )
    x = torch.from_numpy(source_norm[None, None]).to(device)
    with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        pred = inferer(x, lambda t: model(t, cov, domain))
    return torch.clamp(pred, -5.0, 5.0)[0, 0].cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "iqr": float("nan"), "n": 0}
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
        "iqr":    float(q75 - q25),
        "n":      int(len(arr)),
    }

def fmt_mean_std(values: List[float], dec: int = 4) -> str:
    s = summarize(values)
    if s["n"] == 0: return "n/a"
    f = f"{{:.{dec}f}}"
    return f"{f.format(s['mean'])} +/- {f.format(s['std'])}"

def fmt_median_iqr(values: List[float], dec: int = 4) -> str:
    s = summarize(values)
    if s["n"] == 0: return "n/a"
    f = f"{{:.{dec}f}}"
    return f"{f.format(s['median'])} +/- {f.format(s['iqr'])}"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Quantitative metrics on the full test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test_excel",   required=True)
    p.add_argument("--ckpt_aa",      required=True, help="best.pt Anatomy-Aware")
    p.add_argument("--ckpt_unif",    required=True, help="best.pt Uniform Weights")
    p.add_argument("--ckpt_base",    required=True, help="best.pt Baseline")
    p.add_argument("--outdir",       required=True)
    p.add_argument("--base_ch",      type=int,   default=32)
    p.add_argument("--cond_dim",     type=int,   default=64)
    p.add_argument("--patch_size",   type=int,   nargs=3, default=[96, 96, 96])
    p.add_argument("--overlap",      type=float, default=0.5)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--max_subjects", type=int,   default=None,
                   help="Limit N subjects for quick tests")
    p.add_argument("--no_amp",       action="store_true")
    args = p.parse_args()

    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp
    print(f"[INFO] Device: {device} | AMP: {use_amp}")

    # ── Test set ──────────────────────────────────────────────────────────────
    print("\n[1/3] Loading test set...")
    test_df = load_test_split(args.test_excel, seed=args.seed)
    if args.max_subjects:
        test_df = test_df.sample(
            min(args.max_subjects, len(test_df)), random_state=args.seed
        ).reset_index(drop=True)
    print(f"  Subjects: {len(test_df)}")

    # ── Models ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Loading models...")
    model_specs = {
        "Anatomy-Aware":   args.ckpt_aa,
        "Uniform Weights": args.ckpt_unif,
        "Baseline":        args.ckpt_base,
    }
    models = {}
    for name, path in model_specs.items():
        print(f"  -> {name}")
        models[name] = load_model(path, args.base_ch, args.cond_dim, device)

    # ── Evaluation ───────────────────────────────────────────────────────────
    print(f"\n[3/3] Evaluating {len(test_df)} subjects x {len(models)} models...")

    all_metrics: Dict[str, Dict[str, List[float]]] = {
        name: {"L1": [], "PSNR": [], "SSIM": [], "WMH_L1": []}
        for name in models
    }
    rows  = []
    errs  = 0

    for i, (_, row) in enumerate(test_df.iterrows()):
        try:
            src_raw  = load_nifti(str(row["T1_MNI"]))
            tgt_raw  = load_nifti(str(row["Flair_N4"]))
            mask_vol = load_nifti(str(row["Pseudolabel"])).astype(np.int64)
            support  = (np.abs(src_raw) > 1e-8).astype(np.uint8)
            src_norm = robust_normalize(src_raw, mask=support)

            age = float(row["Age"]) if "Age" in row.index and not pd.isna(row.get("Age")) else 65.0
            sex = str(row.get("Sex", "M"))
            dom = int(row["cluster"]) if "cluster" in row.index and not pd.isna(row.get("cluster", np.nan)) else 0

            cov_t  = torch.tensor([[
                np.clip(age, 0, 120) / 100.0,
                encode_sex(sex),
                0.0,  # T1->FLAIR
            ]], dtype=torch.float32).to(device)

            subj_row = {"subject_idx": i}

            for model_name, model in models.items():
                n_dom  = model.domain_emb.num_embeddings
                dom_t  = torch.tensor([dom if dom < n_dom else 0], dtype=torch.long).to(device)

                pred = run_inference(
                    model, src_norm, cov_t, dom_t,
                    tuple(args.patch_size), args.overlap, device, use_amp
                )
                m = compute_all_metrics(pred, tgt_raw, mask_vol)

                for k, v in m.items():
                    all_metrics[model_name][k].append(v)

                sfx = model_name.replace(" ", "_")
                subj_row.update({
                    f"{sfx}_L1": m["L1"], f"{sfx}_PSNR": m["PSNR"],
                    f"{sfx}_SSIM": m["SSIM"], f"{sfx}_WMH_L1": m["WMH_L1"],
                })

            rows.append(subj_row)

            if (i + 1) % 10 == 0 or (i + 1) == len(test_df):
                aa   = all_metrics["Anatomy-Aware"]
                nval = sum(1 for v in aa["L1"] if np.isfinite(v))
                print(
                    f"  [{i+1:4d}/{len(test_df)}] AA -> "
                    f"L1={np.nanmean(aa['L1']):.4f} "
                    f"PSNR={np.nanmean(aa['PSNR']):.2f} "
                    f"SSIM={np.nanmean(aa['SSIM']):.4f} "
                    f"WMH-L1={np.nanmean(aa['WMH_L1']):.4f} "
                    f"(n={nval})"
                )

        except Exception as e:
            print(f"  [WARN] Sujeto {i}: {e}")
            for name in models:
                for k in ["L1", "PSNR", "SSIM", "WMH_L1"]:
                    all_metrics[name][k].append(float("nan"))
            errs += 1

    print(f"  Errors: {errs}/{len(test_df)}")

    # ── CSVs ─────────────────────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv(outdir / "metrics_per_subject.csv", index=False)

    summary_rows = []
    for name in models:
        m = all_metrics[name]
        s = {k: summarize(m[k]) for k in ["L1", "PSNR", "SSIM", "WMH_L1"]}
        summary_rows.append({
            "Model":         name,
            "N_valid":       s["L1"]["n"],
            "L1_mean":       s["L1"]["mean"],   "L1_std":       s["L1"]["std"],
            "PSNR_mean":     s["PSNR"]["mean"],  "PSNR_std":     s["PSNR"]["std"],
            "SSIM_mean":     s["SSIM"]["mean"],  "SSIM_std":     s["SSIM"]["std"],
            "WMH_L1_mean":   s["WMH_L1"]["mean"],"WMH_L1_std":   s["WMH_L1"]["std"],
            "WMH_L1_median": s["WMH_L1"]["median"],"WMH_L1_iqr": s["WMH_L1"]["iqr"],
            "N_with_WMH":    s["WMH_L1"]["n"],
        })
    pd.DataFrame(summary_rows).to_csv(outdir / "metrics_summary.csv", index=False)

    # ── Table for the paper ──────────────────────────────────────────────────
    sep  = "=" * 82
    sep2 = "-" * 82
    hdr  = f"{'Method':<22} {'L1 (mean+/-std)':>18} {'PSNR (mean+/-std)':>18} {'SSIM (mean+/-std)':>18} {'WMH-L1 (mean+/-std)':>20}"

    lines = [
        sep,
        f"TABLE 1 — Ablation Study  |  T1->FLAIR  |  Test set n={len(test_df)}",
        sep, hdr, sep2,
    ]
    for name in models:
        m = all_metrics[name]
        lines.append(
            f"{name:<22} "
            f"{fmt_mean_std(m['L1'],   4):>18} "
            f"{fmt_mean_std(m['PSNR'], 2):>18} "
            f"{fmt_mean_std(m['SSIM'], 4):>18} "
            f"{fmt_mean_std(m['WMH_L1'], 4):>20}"
        )
    lines += [sep, "", "WMH-L1 as median +/- IQR (recommended for paper table):", sep2]
    for name in models:
        m    = all_metrics[name]
        nwmh = sum(1 for v in m["WMH_L1"] if np.isfinite(v))
        lines.append(f"  {name:<22} {fmt_median_iqr(m['WMH_L1'], 4)}  (n_with_WMH={nwmh})")
    lines += [
        "", sep,
        "Copy mean+/-std into Table 1 columns L1 / PSNR / SSIM.",
        "Use median+/-IQR for WMH-L1 (more robust to outlier subjects).",
        sep,
    ]

    table_str = "\n".join(lines)
    print("\n" + table_str + "\n")
    (outdir / "metrics_table.txt").write_text(table_str, encoding="utf-8")

    print(f"Files saved in: {outdir.resolve()}")
    print("  metrics_per_subject.csv — per-subject metrics")
    print("  metrics_summary.csv     — statistical summary")
    print("  metrics_table.txt       — paper-ready table")


if __name__ == "__main__":
    main()
