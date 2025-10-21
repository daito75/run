#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPose BODY_25 JSON → 関節角度（股・膝・足首）の時系列を算出 & プロット
- 側面視を想定（画像 x 方向が進行方向、y は下向き）
- 膝角度: 大腿と下腿のなす角 → 解剖学的屈曲角 = 180 - 角度
- 股角度: 大腿ベクトルと「鉛直上向き」のなす角（屈曲を正）
- 足関節: 下腿と足部（踵→母趾）のなす角 → 背屈を正に近づく向きで 180 - 角度
- NaN補間 + Savitzky–Golay で平滑
- events.csv を渡すと HS/TO を縦線で重ね描き

使い方例:
python3 analyze_joint_angles.py \
    --json_dir /mnt/d/BRLAB/2025/openpose_out/0LR/1/json \
    --out_dir  /mnt/d/BRLAB/2025/openpose_out/0LR/1/angles \
    --fps 30 --conf_th 0.3 \
    --events_csv /mnt/d/BRLAB/2025/openpose_out/0LR/1/gaitevents/events.csv
"""
import os, glob, json, argparse, math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# BODY_25 index
NOSE=0; NECK=1; MIDHIP=8
RHIP, RKNEE, RANK = 9,10,11
LHIP, LKNEE, LANK = 12,13,14
LBIG, LSMALL, LHEEL = 19,20,21
RBIG, RSMALL, RHEEL = 22,23,24

def load_seq(json_dir, conf_th):
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    seq=[]
    for f in files:
        J = json.load(open(f,"r"))
        if not J.get("people"):
            seq.append(None); continue
        # もっとも総信頼度の高い person を採用
        def score(p): 
            k = np.array(p["pose_keypoints_2d"]).reshape(-1,3)
            return float(np.nansum(k[:,2]))
        best = max(J["people"], key=score)
        k = np.array(best["pose_keypoints_2d"]).reshape(-1,3)  # (25,3)
        # 低信頼は NaN
        k[k[:,2] < conf_th,:2] = np.nan
        seq.append(k)
    return seq

def interp_nan_1d(a):
    a = np.asarray(a, float)
    n = len(a); idx = np.arange(n)
    mask = np.isfinite(a)
    if mask.sum() < 2: return a
    a[~mask] = np.interp(idx[~mask], idx[mask], a[mask])
    return a

def smooth_1d(a, win=9, poly=2):
    a = np.asarray(a, float)
    if len(a) < win or win<3 or win%2==0:
        return a
    return savgol_filter(a, window_length=win, polyorder=poly)

def angle_between(u, v):
    """2Dベクトルのなす角 [deg], 0..180"""
    un = u/ (np.linalg.norm(u) + 1e-9)
    vn = v/ (np.linalg.norm(v) + 1e-9)
    c = np.clip(np.dot(un, vn), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def signed_angle_to_vertical(v):
    """鉛直上向き(0,-1) からの有向角 [deg]。右向き(+)を前方とし、前方屈曲を正に近い符号付け。"""
    # 画像座標で y 下向きなので、上向きは (0,-1)
    up = np.array([0.0, -1.0], float)
    ang = angle_between(v, up)  # 0..180
    # 右向き成分で符号づけ（vのx>0で正、x<0で負）
    return ang if v[0] >= 0 else -ang

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--conf_th", type=float, default=0.3)
    ap.add_argument("--events_csv", default=None)
    ap.add_argument("--smooth_win", type=int, default=9)
    ap.add_argument("--smooth_poly", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seq = load_seq(args.json_dir, args.conf_th)
    nF = len(seq)
    t = np.arange(nF) / float(args.fps)

    # 座標時系列（不足フレームはNaN）
    def kxy(idx):
        xs, ys = [], []
        for k in seq:
            if k is None: xs.append(np.nan); ys.append(np.nan)
            else: xs.append(k[idx,0]); ys.append(k[idx,1])
        xs, ys = np.array(xs,float), np.array(ys,float)
        xs, ys = interp_nan_1d(xs), interp_nan_1d(ys)
        xs, ys = smooth_1d(xs,args.smooth_win,args.smooth_poly), smooth_1d(ys,args.smooth_win,args.smooth_poly)
        return xs, ys

    # L/R 関節の2D座標
    xLhip,yLhip = kxy(LHIP); xLkne,yLkne = kxy(LKNEE); xLank,yLank = kxy(LANK)
    xRhip,yRhip = kxy(RHIP); xRkne,yRkne = kxy(RKNEE); xRank,yRank = kxy(RANK)
    xLbig,yLbig = kxy(LBIG); xLhee,yLhee = kxy(LHEEL)
    xRbig,yRbig = kxy(RBIG); xRhee,yRhee = kxy(RHEEL)

    # ベクトル生成ヘルパ
    def vec(ax,ay, bx,by):  # a->b
        return np.stack([bx-ax, by-ay], axis=1)

    # --- 角度算出 ---
    # 膝屈曲: なす角(大腿, 下腿) → flex = 180 - angle_between
    thigh_L = vec(xLhip,yLhip, xLkne,yLkne)
    shank_L = vec(xLank,yLank, xLkne,yLkne)  # ひざ頂点に向かうベクトル
    knee_L_raw = np.array([angle_between(thigh_L[i], shank_L[i]) for i in range(nF)])
    knee_L_flex = 180.0 - knee_L_raw

    thigh_R = vec(xRhip,yRhip, xRkne,yRkne)
    shank_R = vec(xRank,yRank, xRkne,yRkne)
    knee_R_raw = np.array([angle_between(thigh_R[i], shank_R[i]) for i in range(nF)])
    knee_R_flex = 180.0 - knee_R_raw

    # 股屈曲: 大腿ベクトル vs 鉛直上向き → 前方(進行方向+)屈曲を正に
    hip_L_flex = np.array([signed_angle_to_vertical(thigh_L[i]) for i in range(nF)])
    hip_R_flex = np.array([signed_angle_to_vertical(thigh_R[i]) for i in range(nF)])

    # 足関節（背屈+）: 下腿 vs 足部（踵→母趾） → 180 - 角度
    foot_L = vec(xLhee,yLhee, xLbig,yLbig)
    foot_R = vec(xRhee,yRhee, xRbig,yRbig)
    ankle_L = np.array([180.0 - angle_between(shank_L[i], foot_L[i]) for i in range(nF)])
    ankle_R = np.array([180.0 - angle_between(shank_R[i], foot_R[i]) for i in range(nF)])

    # 出力CSV
    df = pd.DataFrame({
        "time_s": t,
        "hip_L_deg": hip_L_flex, "hip_R_deg": hip_R_flex,
        "knee_L_deg": knee_L_flex, "knee_R_deg": knee_R_flex,
        "ankle_L_deg": ankle_L, "ankle_R_deg": ankle_R
    })
    csv_path = os.path.join(args.out_dir, "joint_angles.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] wrote {csv_path}")

    # 図: 3段（股・膝・足首）。HS/TO があれば縦線を重ねる
    hsL, toL, hsR, toR = [],[],[],[]
    if args.events_csv and os.path.exists(args.events_csv):
        ev = pd.read_csv(args.events_csv)
        hsL = ev[(ev.side=="L")&(ev.event=="HS")]["time_sec"].values
        toL = ev[(ev.side=="L")&(ev.event=="TO")]["time_sec"].values
        hsR = ev[(ev.side=="R")&(ev.event=="HS")]["time_sec"].values
        toR = ev[(ev.side=="R")&(ev.event=="TO")]["time_sec"].values

    def vlines(ax, ts, color, label):
        for i,tt in enumerate(ts):
            ax.axvline(tt, linestyle="--", linewidth=1, color=color, alpha=0.5)
        if len(ts): ax.plot([],[], linestyle="--", color=color, label=label)

    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(3,1,1); ax1.set_title("Hip Flexion (+flex)")
    ax1.plot(t, hip_L_flex, label="L"); ax1.plot(t, hip_R_flex, label="R")
    vlines(ax1, hsL, "tab:green", "L-HS"); vlines(ax1, toL, "tab:olive", "L-TO")
    vlines(ax1, hsR, "tab:orange", "R-HS"); vlines(ax1, toR, "tab:red", "R-TO")
    ax1.set_ylabel("deg"); ax1.legend(loc="upper right"); ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3,1,2); ax2.set_title("Knee Flexion (+flex)")
    ax2.plot(t, knee_L_flex, label="L"); ax2.plot(t, knee_R_flex, label="R")
    vlines(ax2, hsL, "tab:green", "L-HS"); vlines(ax2, toL, "tab:olive", "L-TO")
    vlines(ax2, hsR, "tab:orange", "R-HS"); vlines(ax2, toR, "tab:red", "R-TO")
    ax2.set_ylabel("deg"); ax2.legend(loc="upper right"); ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3,1,3); ax3.set_title("Ankle Dorsi(+)")
    ax3.plot(t, ankle_L, label="L"); ax3.plot(t, ankle_R, label="R")
    vlines(ax3, hsL, "tab:green", "L-HS"); vlines(ax3, toL, "tab:olive", "L-TO")
    vlines(ax3, hsR, "tab:orange", "R-HS"); vlines(ax3, toR, "tab:red", "R-TO")
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("deg"); ax3.legend(loc="upper right"); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "joint_angles.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[OK] wrote {fig_path}")

if __name__ == "__main__":
    main()
