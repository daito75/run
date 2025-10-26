#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B方式：左右を毎回読み、入替をその場で補正してから解析する簡潔版


python3 test.py \
  --json_dir "/mnt/c/brlab/2025/mizuno/openpose/3.0m/60cm/json" \
  --out_dir  "/mnt/c/brlab/2025/mizuno/openpose/3.0m/60cm/gaitevents_steps" \
  --side right \
  --conf_th 0.3 \
  --smooth_win 7 \
  --min_frames 10 \
  --neg_th_px 2.0 \
  --margin_px 3.0

"""

import os, json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# BODY_25
MID_HIP = 8
R_BIGTOE, R_HEEL = 22, 24
L_BIGTOE, L_HEEL = 19, 21

# ---------- I/O: 左右+腰 を読む ----------
def load_both(json_dir, conf_th=0.3):
    files = sorted(Path(json_dir).glob("*.json"))
    if not files: raise SystemExit(f"[ERR] JSONなし: {json_dir}")
    xr, yr, xl, yl, xm = [], [], [], [], []
    for p in files:
        d = json.loads(p.read_text())
        if not d.get("people"):
            xr += [np.nan]; yr += [np.nan]
            xl += [np.nan]; yl += [np.nan]
            xm += [np.nan]; continue
        pts = np.array(d["people"][0]["pose_keypoints_2d"], np.float32).reshape(-1,3)
        if pts.shape[0] < 25:
            xr += [np.nan]; yr += [np.nan]
            xl += [np.nan]; yl += [np.nan]
            xm += [np.nan]; continue

        def avg_foot(toe, heel):
            xs, ys = [], []
            if pts[toe,2]  >= conf_th: xs.append(pts[toe,0]);  ys.append(pts[toe,1])
            if pts[heel,2] >= conf_th: xs.append(pts[heel,0]); ys.append(pts[heel,1])
            return (float(np.mean(xs)), float(np.mean(ys))) if xs else (np.nan, np.nan)

        rx, ry = avg_foot(R_BIGTOE, R_HEEL)
        lx, ly = avg_foot(L_BIGTOE, L_HEEL)
        mx = float(pts[MID_HIP,0]) if pts[MID_HIP,2] >= conf_th else np.nan

        xr.append(rx); yr.append(ry)
        xl.append(lx); yl.append(ly)
        xm.append(mx)

    return {
        "x_r": np.array(xr, float), "y_r": np.array(yr, float),
        "x_l": np.array(xl, float), "y_l": np.array(yl, float),
        "x_mid": np.array(xm, float), "n": len(files), "frames": [p.name for p in files]
    }

# ---------- 入替補正（prev採用値基準） ----------
def fix_swaps_by_x_direction(x_main, x_other, dir="increasing", neg_th_px=2.0):
    """
    x座標だけで判定：
    - increasing: xが増えるのが正しい（後退 = dx < -neg_th_px）
    - decreasing: xが減るのが正しい（後退 = dx > +neg_th_px）
    後退したフレームを other の値で置き換える。
    """
    x_main = np.asarray(x_main, float).copy()
    x_other = np.asarray(x_other, float)
    n = len(x_main)
    swap_flags = np.zeros(n, dtype=bool)

    # 最初の有限点（スタート位置）
    prev = next((i for i in range(n) if np.isfinite(x_main[i])), None)
    if prev is None:
        return x_main, swap_flags

    def is_backward(dx):
        if dir == "increasing":
            return dx < -neg_th_px
        else:  # decreasing
            return dx > +neg_th_px

    for t in range(prev+1, n):
        px = x_main[prev]
        if not np.isfinite(px):
            prev = t
            continue
        if not np.isfinite(x_main[t]) and np.isfinite(x_other[t]):
            # main欠損時はother採用
            x_main[t] = x_other[t]
            swap_flags[t] = True
            prev = t
            continue

        dx_m = x_main[t] - px

        # 後退したら入れ替え
        if is_backward(dx_m):
            if np.isfinite(x_other[t]):
                dx_o = x_other[t] - px
                # otherが正方向なら採用
                if not is_backward(dx_o):
                    x_main[t] = x_other[t]
                    swap_flags[t] = True
        prev = t

    return x_main, swap_flags


# ---------- 補助（補間・平滑・検出・描画） ----------
def nan_interp(a):
    a = a.astype(float); n = a.size; idx = np.arange(n)
    mask = np.isfinite(a)
    if mask.sum()==0: return np.zeros_like(a)
    first, last = np.argmax(mask), n-1-np.argmax(mask[::-1])
    a[:first] = a[first]; a[last+1:] = a[last]
    mask = np.isfinite(a)
    return np.interp(idx, idx[mask], a[mask])

def moving_avg(x, win=7):
    win = int(max(1, (win//2)*2+1));  pad = win//2
    if win<=1: return x.copy()
    xp = np.pad(x, (pad,pad), mode="edge")
    return np.convolve(xp, np.ones(win)/win, mode="valid")

def derivative(x): return np.gradient(x.astype(float))

def find_cycles_max_min_max(sig, min_frames=10, min_amp=None):
    x = sig; n = len(x); s = np.sign(derivative(x)); s[s==0]=1
    zc = np.where(np.diff(s)!=0)[0] + 1
    maxima, minima = [], []
    for i in zc:
        if 1<=i<=n-2:
            if x[i]>=x[i-1] and x[i]>=x[i+1]: maxima.append(i)
            if x[i]<=x[i-1] and x[i]<=x[i+1]: minima.append(i)
    maxima, minima = np.array(maxima), np.array(minima)
    if min_amp is None: min_amp = 0.2*np.nanstd(x)
    cycles = []
    for i in range(len(maxima)-1):
        i0, i1 = maxima[i], maxima[i+1]
        mb = minima[(minima>i0) & (minima<i1)]
        if len(mb)==0 or (i1-i0)<min_frames: continue
        j = int(mb[np.argmin(x[mb])]); amp = (x[i0]-x[j] + x[i1]-x[j])*0.5
        if amp < min_amp: continue
        cycles.append((int(i0), int(j), int(i1)))
    return cycles

def resample_to_100(x):
    m = len(x); 
    if m<2: return np.full(101, np.nan)
    return np.interp(np.linspace(0,1,101), np.linspace(0,1,m), x)

def plot_step_series(steps, out_png, title, ylabel):
    plt.figure(figsize=(8,4))
    for k,y in enumerate(steps,1):
        t = np.linspace(0,100,len(y)); plt.plot(t,y,label=f"step{k}")
    plt.xlabel("Gait cycle [%]"); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True, alpha=0.3); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="B方式：その場補正 → 解析")
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--side", default="right", choices=["right","left"])
    ap.add_argument("--conf_th", type=float, default=0.3)
    ap.add_argument("--smooth_win", type=int, default=7)
    ap.add_argument("--min_frames", type=int, default=10)
    ap.add_argument("--min_amp", type=float, default=None)
    ap.add_argument("--neg_th_px", type=float, default=2.0)
    ap.add_argument("--margin_px",  type=float, default=3.0)   # ★調整2
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ft = load_both(args.json_dir, conf_th=args.conf_th)

        # 右主・左従で右列を補正、左主・右従で左列を補正（対称にかける）
    xr, flags_r = fix_swaps_by_x_direction(
        ft["x_r"], ft["x_l"], dir="increasing", neg_th_px=2.0)
    xl, flags_l = fix_swaps_by_x_direction(
        ft["x_l"], ft["x_r"], dir="increasing", neg_th_px=2.0)

    
    # 解析対象の脚＆フラグを選択
    if args.side.startswith("r"):
        x_foot = xr
        swap_flags = flags_r
    else:
        x_foot = xl
        swap_flags = flags_l

    relx   = nan_interp(x_foot - ft["x_mid"])
    relx_s = moving_avg(relx, args.smooth_win)
    vrelx  = derivative(relx_s)

    # ---- ここから「入れ替えフレームの表示＆保存」 ----
    swap_idx = np.where(swap_flags)[0]
    if swap_idx.size > 0:
        print("[INFO] swap frames (入れ替え採用フレーム):", swap_idx.tolist())
        # CSV保存
        import csv
        swap_csv = Path(args.out_dir)/"swap_frames.csv"
        with open(swap_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx","filename","side"])
            side_used = "right" if args.side.startswith("r") else "left"
            for i in swap_idx:
                name = ft["frames"][i] if "frames" in ft and i < len(ft["frames"]) else ""
                w.writerow([int(i), name, side_used])
        print(f"[INFO] swap frames csv -> {swap_csv}")
    else:
        print("[INFO] swap frames (入れ替え採用フレーム): none")

    cycles = find_cycles_max_min_max(relx_s, min_frames=args.min_frames, min_amp=args.min_amp)

    # CSV
    import csv
    csv_path = Path(args.out_dir)/"steps_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["step_id","max_start","min_mid","max_end","length_frames"])
        for si,(a,b,c) in enumerate(cycles,1): w.writerow([si,a,b,c,c-a])
    print(f"[INFO] steps -> {csv_path}")

    # ステップ可視化
    steps_relx, steps_v = [], []
    steps_dir = Path(args.out_dir)/"steps"; steps_dir.mkdir(exist_ok=True, parents=True)
    for si,(a,b,c) in enumerate(cycles,1):
        y_relx = resample_to_100(relx_s[a:c+1]); y_v = resample_to_100(vrelx[a:c+1])
        steps_relx.append(y_relx); steps_v.append(y_v)
        t = np.linspace(0,100,len(y_relx))
        plt.figure(figsize=(8,3)); plt.plot(t,y_relx,label="relative x"); plt.plot(t,y_v,label="velocity",alpha=0.7)
        plt.title(f"Step {si} ({a}→{c})"); plt.xlabel("Gait cycle [%]"); plt.ylabel("px / px/frame")
        plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(steps_dir/f"step_{si:02d}.png", dpi=200); plt.close()

    plot_step_series(steps_relx, Path(args.out_dir)/"compare_relx.png",
                     f"Foot relative X vs. gait cycle ({args.side})", "relative x [px]")
    plot_step_series(steps_v, Path(args.out_dir)/"compare_velocity.png",
                     f"Foot velocity vs. gait cycle ({args.side})", "velocity [px/frame]")

    # 全体波形
    plt.figure(figsize=(10,4))
    t = np.arange(len(relx_s)); plt.plot(t, relx_s, label="relative x (smoothed)")

    # ★入れ替えフレームにマゼンタ点線
    for i in swap_idx:
        plt.axvline(i, color="m", linestyle="--", alpha=0.5)

    for si,(a,b,c) in enumerate(cycles,1):
        for x, col in [(a,"g"),(b,"y"),(c,"r")]: plt.axvline(x,color=col,alpha=0.5)
        plt.text(a, relx_s[a], f"S{si}", color="g", fontsize=8)
    plt.xlabel("frame"); plt.ylabel("relative x [px]"); plt.title("Whole series with step boundaries")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(Path(args.out_dir)/"series_with_boundaries.png", dpi=200); plt.close()

    print(f"[DONE] out -> {args.out_dir}")
    print(" - steps_index.csv, steps/*.png, compare_relx.png, compare_velocity.png, series_with_boundaries.png")

if __name__ == "__main__":
    main()
