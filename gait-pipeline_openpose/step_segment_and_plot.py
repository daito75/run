#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 step_segment_and_plot.py \
  --json_dir "/mnt/c/brlab/2025/mizuno/openpose/3.0m/60cm/json" \
  --out_dir  "/mnt/c/brlab/2025/mizuno/openpose/3.0m/60cm/gaitevents_steps" \
  --side right \
  --conf_th 0.3 \
  --smooth_win 7 \
  --min_frames 10

"""

import os, json, argparse, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # サーバでも描ける
import matplotlib.pyplot as plt
import numpy as np, json

# BODY_25 のインデックス
MID_HIP = 8
R_BIGTOE, R_HEEL = 22, 24
L_BIGTOE, L_HEEL = 19, 21

# === 右/左のToe+Heel平均をまとめて読む（x_r, x_l, x_mid を返す） ===
def load_both_feet_from_json(json_dir, conf_th=0.3):
    json_dir = Path(json_dir)
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"[ERR] JSONなし: {json_dir}")

    xr, yr, xl, yl, xm, ym = [], [], [], [], [], []
    for p in files:
        data = json.loads(p.read_text())
        if not data.get("people"):
            xr += [np.nan]; yr += [np.nan]
            xl += [np.nan]; yl += [np.nan]
            xm += [np.nan]; ym += [np.nan]
            continue
        pts = np.array(data["people"][0]["pose_keypoints_2d"], dtype=np.float32).reshape(-1,3)
        if pts.shape[0] < 25:
            xr += [np.nan]; yr += [np.nan]
            xl += [np.nan]; yl += [np.nan]
            xm += [np.nan]; ym += [np.nan]
            continue

        def avg_foot(toe_idx, heel_idx):
            toe, heel = pts[toe_idx], pts[heel_idx]
            xs, ys = [], []
            if toe[2]  >= conf_th: xs.append(toe[0]);  ys.append(toe[1])
            if heel[2] >= conf_th: xs.append(heel[0]); ys.append(heel[1])
            if xs: return float(np.mean(xs)), float(np.mean(ys))
            return np.nan, np.nan

        rx, ry = avg_foot(R_BIGTOE, R_HEEL)
        lx, ly = avg_foot(L_BIGTOE, L_HEEL)

        mid = pts[MID_HIP]
        mx = float(mid[0]) if mid[2] >= conf_th else np.nan
        my = float(mid[1]) if mid[2] >= conf_th else np.nan

        xr.append(rx); yr.append(ry)
        xl.append(lx); yl.append(ly)
        xm.append(mx); ym.append(my)

    return dict(
        x_r=np.array(xr, dtype=float), y_r=np.array(yr, dtype=float),
        x_l=np.array(xl, dtype=float), y_l=np.array(yl, dtype=float),
        x_mid=np.array(xm, dtype=float), y_mid=np.array(ym, dtype=float),
        n=len(files), frames=[p.name for p in files]
    )

def fix_swaps_by_jump_threshold(
    x_main: np.ndarray, x_other: np.ndarray,
    y_main: np.ndarray | None = None, y_other: np.ndarray | None = None,
    jump_th_px: float = 10.0,
    margin_px: float = 1.0,
    mode: str = "swap_if_better",
):
    """
    直前に“採用”した値（prev_ref）との距離で keep/swap を比較する版。
    これにより 142 で swap したら、143/144 も「swap後の値」を基準に継続評価され、
    “異常区間が連続する間は連続で swap” できる。
    """
    # 出力（採用系列）を作る：最初は main のコピー
    ax = np.asarray(x_main, dtype=float).copy()
    use_y = y_main is not None and y_other is not None
    if use_y:
        ay = np.asarray(y_main,  dtype=float).copy()
        xo = np.asarray(x_other, dtype=float)
        yo = np.asarray(y_other, dtype=float)
    else:
        ay = None
        xo = np.asarray(x_other, dtype=float)
        yo = None

    n = len(ax)
    if n == 0:
        return (ax, ay)

    def dist_xy(px, py, cx, cy):
        """2D距離（yがNoneなら|Δx|）。どれかNaNならinf。"""
        if not np.isfinite(px) or not np.isfinite(cx):
            return np.inf
        if ay is None:
            return abs(cx - px)
        if not np.isfinite(py) or not np.isfinite(cy):
            return np.inf
        return float(np.hypot(cx - px, cy - py))

    # prev_ref（直前の“採用”値）のインデックスを探すヘルパ
    def find_prev_idx(k):
        i = k - 1
        while i >= 0:
            if np.isfinite(ax[i]) and (ay is None or np.isfinite(ay[i])):
                return i
            i -= 1
        return -1

    for t in range(0, n):
        # t=0 はそのまま（初期値）。欠損ならスキップ。
        if t == 0:
            continue

        prev_i = find_prev_idx(t)
        if prev_i < 0:
            # まだ prev_ref が決まってない → ここでは何もしない
            continue

        # prev_ref（直前に採用した値）
        px = ax[prev_i]
        py = ay[prev_i] if ay is not None else None

        # いまの観測（元の足 / 逆足）
        mx = x_main[t]
        my = y_main[t] if ay is not None else None
        ox = x_other[t]
        oy = y_other[t] if ay is not None else None

        # 欠損処理：main が欠損なら、other が十分近ければ other を採用
        if not np.isfinite(mx) or (ay is not None and not np.isfinite(my)):
            if mode == "swap_if_better":
                d_swap = dist_xy(px, py, ox, oy)
                if d_swap <= (jump_th_px - margin_px):
                    ax[t] = ox
                    if ay is not None: ay[t] = oy
            # drop/keep のときは何もしない（後段の補間に任せる）
            continue

        # 距離を同じ prev_ref から比較
        d_keep = dist_xy(px, py, mx, my)
        d_swap = dist_xy(px, py, ox, oy)

        # 正常なら keep
        if d_keep <= jump_th_px:
            ax[t] = mx
            if ay is not None: ay[t] = my
            continue

        # ここから“飛び”
        if mode == "drop":
            ax[t] = np.nan
            if ay is not None: ay[t] = np.nan
            continue

        if mode == "swap_if_better":
            # 逆足の方が連続的なら採用（marginでヒステリシス）
            if (d_swap + margin_px) < d_keep:
                ax[t] = ox
                if ay is not None: ay[t] = oy
            else:
                ax[t] = np.nan
                if ay is not None: ay[t] = np.nan
        elif mode == "keep":
            # 何もしない
            ax[t] = mx
            if ay is not None: ay[t] = my

    return (ax, ay)

def load_series_from_json(json_dir, side="right", conf_th=0.3):
    json_dir = Path(json_dir)
    files = sorted([p for p in json_dir.glob("*.json")])
    if not files:
        raise SystemExit(f"[ERR] JSONなし: {json_dir}")

    xs_foot, ys_foot, conf_foot = [], [], []
    xs_mid, ys_mid, conf_mid = [], [], []

    if side.lower().startswith("r"):
        toe_idx, heel_idx = R_BIGTOE, R_HEEL
    else:
        toe_idx, heel_idx = L_BIGTOE, L_HEEL

    for p in files:
        data = json.loads(p.read_text())
        # OpenPose JSON: people: [{pose_keypoints_2d: [x,y,c, x,y,c, ...]}]
        if not data.get("people"):
            # いないフレームは NaN
            xs_foot.append(np.nan); ys_foot.append(np.nan); conf_foot.append(0.0)
            xs_mid.append(np.nan); ys_mid.append(np.nan); conf_mid.append(0.0)
            continue
        kp = data["people"][0]["pose_keypoints_2d"]
        pts = np.array(kp, dtype=np.float32).reshape(-1,3)
        # 基本の安全確認
        if pts.shape[0] < 25:
            xs_foot.append(np.nan); ys_foot.append(np.nan); conf_foot.append(0.0)
            xs_mid.append(np.nan); ys_mid.append(np.nan); conf_mid.append(0.0)
            continue
        # 足（つま先＋踵）の平均を足位置とする
        foot = pts[toe_idx]; heel = pts[heel_idx]; mid = pts[MID_HIP]
        # 信頼度で弾く
        if foot[2] < conf_th and heel[2] < conf_th:
            x_f = np.nan; y_f = np.nan; c_f = max(foot[2], heel[2])
        else:
            # 片方欠けてもある方を採用
            xs = []; ys = []; cs=[]
            if foot[2] >= conf_th: xs.append(foot[0]); ys.append(foot[1]); cs.append(foot[2])
            if heel[2] >= conf_th: xs.append(heel[0]); ys.append(heel[1]); cs.append(heel[2])
            x_f = float(np.mean(xs)) if xs else np.nan
            y_f = float(np.mean(ys)) if ys else np.nan
            c_f = float(np.mean(cs)) if cs else 0.0

        x_m = mid[0] if mid[2] >= conf_th else np.nan
        y_m = mid[1] if mid[2] >= conf_th else np.nan
        c_m = mid[2]

        xs_foot.append(x_f); ys_foot.append(y_f); conf_foot.append(c_f)
        xs_mid.append(x_m);  ys_mid.append(y_m);  conf_mid.append(c_m)

    return {
        "x_foot": np.array(xs_foot, dtype=np.float32),
        "y_foot": np.array(ys_foot, dtype=np.float32),
        "c_foot": np.array(conf_foot, dtype=np.float32),
        "x_mid":  np.array(xs_mid,  dtype=np.float32),
        "y_mid":  np.array(ys_mid,  dtype=np.float32),
        "c_mid":  np.array(conf_mid, dtype=np.float32),
        "n": len(files),
        "frames": [p.name for p in files],
    }

def nan_interp(a: np.ndarray):
    """NaN を線形補間（端は最近値で埋め）"""
    a = a.astype(float)
    n = a.size
    idx = np.arange(n)
    mask = np.isfinite(a)
    if mask.sum() == 0:
        return np.zeros_like(a)
    # 端埋め
    first = np.argmax(mask)
    last  = len(mask) - 1 - np.argmax(mask[::-1])
    a[:first] = a[first]
    a[last+1:] = a[last]
    mask = np.isfinite(a)
    return np.interp(idx, idx[mask], a[mask])

def moving_avg(x, win=7):
    win = int(max(1, win//2*2+1))  # 奇数化
    if win <= 1: return x.copy()
    pad = win//2
    xp = np.pad(x, (pad,pad), mode="edge")
    kern = np.ones(win)/win
    return np.convolve(xp, kern, mode="valid")

def derivative(x):
    d = np.gradient(x.astype(float))
    return d

def find_cycles_max_min_max(sig, min_frames=10, min_amp=None):
    """
    右足(相対x)の波形から max→min→max を1歩として抽出
    min_amp: 振幅しきい値（なければ 0.2*std）
    """
    x = sig
    n = len(x)
    # 局所極値の粗検出
    dx = derivative(x)
    # 極大: dx[i-1]>0 & dx[i+1]<0 と近似（符号反転利用）
    s = np.sign(dx)
    s[s==0] = 1
    zc = np.where(np.diff(s) != 0)[0] + 1  # 勾配符号の変わり目
    # 候補から極大/極小を絞る
    maxima = []
    minima = []
    for i in zc:
        # 端は無視
        if i <= 1 or i >= n-2: continue
        # 近傍比較
        if x[i] >= x[i-1] and x[i] >= x[i+1]:
            maxima.append(i)
        if x[i] <= x[i-1] and x[i] <= x[i+1]:
            minima.append(i)
    maxima = np.array(maxima); minima = np.array(minima)

    if min_amp is None:
        min_amp = 0.2*np.nanstd(x)

    # max-min-max の並びを抽出
    cycles = []
    for i in range(len(maxima)-1):
        i0 = maxima[i]
        i1 = maxima[i+1]
        # 間にある最小を1つ選ぶ
        mins_between = minima[(minima > i0) & (minima < i1)]
        if len(mins_between) == 0: continue
        j = int(mins_between[np.argmin(x[mins_between])])  # 一番低い谷
        if (i1 - i0) < min_frames: continue
        amp = (x[i0] - x[j]) + (x[i1] - x[j])
        if amp*0.5 < min_amp:  # 平均片振幅しきい値
            continue
        cycles.append((int(i0), int(j), int(i1)))
    return cycles  # list of (max, min, next_max)

def resample_to_100(x):
    """区間データを 0-100% に補間"""
    m = len(x)
    if m < 2:
        return np.full(101, np.nan)
    xi = np.linspace(0, 1, m)
    xo = np.linspace(0, 1, 101)
    return np.interp(xo, xi, x)

def plot_step_series(steps, out_png, title, ylabel, fps=None):
    plt.figure(figsize=(8,4))
    for k, y in enumerate(steps, 1):
        if fps is None:
            t = np.linspace(0, 100, len(y))
            plt.plot(t, y, label=f"step{k}")
        else:
            # 既に 0-100% に正規化されている前提のときは t は%軸でOK
            t = np.linspace(0, 100, len(y))
            plt.plot(t, y, label=f"step{k}")
    plt.xlabel("Gait cycle [%]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Step segmentation & plotting from OpenPose JSON")
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--side", default="right", choices=["right","left"])
    ap.add_argument("--conf_th", type=float, default=0.3)
    ap.add_argument("--smooth_win", type=int, default=7)
    ap.add_argument("--min_frames", type=int, default=10)
    ap.add_argument("--min_amp", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None, help="動画fps（任意）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 読み出し
    feet = load_both_feet_from_json(args.json_dir, conf_th=args.conf_th)

    # 右脚を補正（x/y両方で判定＆置換）
    xr_guard, yr_guard = fix_swaps_by_jump_threshold(
        feet["x_r"], feet["x_l"],
        feet["y_r"], feet["y_l"],
        jump_th_px=10.0,
        margin_px=1.0,
        mode="swap_if_better"
    )

    # 左脚も同様に補正
    xl_guard, yl_guard = fix_swaps_by_jump_threshold(
        feet["x_l"], feet["x_r"],
        feet["y_l"], feet["y_r"],
        jump_th_px=10.0,
        margin_px=1.0,
        mode="swap_if_better"
    )

    # 解析側を選択
    if args.side.lower().startswith("r"):
        x_foot = xr_guard
    else:
        x_foot = xl_guard
    x_mid = feet["x_mid"]

    relx = x_foot - x_mid  # 右足（or左足）の相対X
    # 欠損を補間 → 平滑化
    relx = nan_interp(relx)
    relx_s = moving_avg(relx, args.smooth_win)

    # 速度（差分）
    vrelx = derivative(relx_s)

    # ステップ区切り（max-min-max）
    cycles = find_cycles_max_min_max(relx_s, min_frames=args.min_frames, min_amp=args.min_amp)

    # 保存：インデックス一覧
    import csv
    csv_path = Path(args.out_dir)/"steps_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step_id","max_start","min_mid","max_end","length_frames"])
        for si,(a,b,c) in enumerate(cycles,1):
            w.writerow([si,a,b,c,c-a])
    print(f"[INFO] steps -> {csv_path}")

    # 各ステップを0-100%正規化して可視化
    steps_relx = []
    steps_v = []
    steps_dir = Path(args.out_dir)/"steps"
    steps_dir.mkdir(exist_ok=True, parents=True)

    for si,(a,b,c) in enumerate(cycles,1):
        seg_relx = relx_s[a:c+1]
        seg_v    = vrelx[a:c+1]
        y_relx = resample_to_100(seg_relx)
        y_v    = resample_to_100(seg_v)
        steps_relx.append(y_relx)
        steps_v.append(y_v)

        # 個別プロット
        plt.figure(figsize=(8,3))
        t = np.linspace(0,100,len(y_relx))
        plt.plot(t, y_relx, label="relative x (foot-midHip)")
        plt.plot(t, y_v,    label="velocity", alpha=0.7)
        plt.title(f"Step {si} ({a}→{c})")
        plt.xlabel("Gait cycle [%]")
        plt.ylabel("value [px or px/frame]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        png = steps_dir / f"step_{si:02d}.png"
        plt.savefig(png, dpi=200); plt.close()

    # 重ね描き（比較）
    plot_step_series(steps_relx, Path(args.out_dir)/"compare_relx.png",
                     f"Foot relative X vs. gait cycle ({args.side})",
                     "relative x [px]")
    plot_step_series(steps_v, Path(args.out_dir)/"compare_velocity.png",
                     f"Foot velocity vs. gait cycle ({args.side})",
                     "velocity [px/frame]")

    # 全体の波形＋区切りも保存
    plt.figure(figsize=(10,4))
    t = np.arange(len(relx_s))
    plt.plot(t, relx_s, label="relative x (smoothed)")
    for si,(a,b,c) in enumerate(cycles,1):
        plt.axvline(a, color="g", alpha=0.5)
        plt.axvline(b, color="y", alpha=0.5)
        plt.axvline(c, color="r", alpha=0.5)
        plt.text(a, relx_s[a], f"S{si}", color="g", fontsize=8)
    plt.xlabel("frame")
    plt.ylabel("relative x [px]")
    plt.title("Whole series with step boundaries")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(args.out_dir)/"series_with_boundaries.png", dpi=200)
    plt.close()

    print(f"[DONE] out -> {args.out_dir}")
    print(" - steps_index.csv")
    print(" - steps/step_XX.png")
    print(" - compare_relx.png, compare_velocity.png")
    print(" - series_with_boundaries.png")

if __name__ == "__main__":
    main()
