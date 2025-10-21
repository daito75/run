#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
左膝・右膝を1枚の図にまとめ、各々で Composite(合成) と Single(単体) の膝角度を重ね描き
- 入力: 各動画の joint_angles.csv（analyze_joint_angles.py の出力）
- 任意: 各動画の events.csv（あれば歩行周期(0–100%)基準の図も出力）
- 出力: knee_overlay_both_time.png（必ず出す）
        knee_overlay_both_cycle.png（events が両方あるとき）

python3 compare_v2.py \
  --a_csv /mnt/d/BRLAB/2025/openpose_out/0LR/1/angles/joint_angles.csv \
  --b_csv /mnt/d/BRLAB/2025/openpose_out/LR/1/angles/joint_angles.csv \
  --a_events /mnt/d/BRLAB/2025/openpose_out/0LR/1/gaitevents/events.csv \
  --b_events /mnt/d/BRLAB/2025/openpose_out/LR/1/gaitevents/events.csv \
  --out_dir /mnt/d/BRLAB/2025/openpose_out/compare/1

        
"""

import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_angles(p):
    df = pd.read_csv(p)
    need = {"time_s","knee_L_deg","knee_R_deg"}
    if not need.issubset(df.columns):
        raise ValueError(f"missing columns in {p}: need {need}")
    return df

def load_hs(ev_path, side):
    if not ev_path or not os.path.exists(ev_path): return None
    ev = pd.read_csv(ev_path)
    if not {"time_sec","side","event"}.issubset(ev.columns): return None
    hs = ev[(ev.side==side) & (ev.event=="HS")]["time_sec"].values
    return np.sort(hs) if len(hs) else None

def cycle_mean(times, values, hs_times, n_points=100):
    if hs_times is None or len(hs_times) < 2: return None, 0
    curves=[]
    for i in range(len(hs_times)-1):
        t0, t1 = hs_times[i], hs_times[i+1]
        if t1<=t0: continue
        m = (times>=t0)&(times<=t1)
        if m.sum()<5: continue
        u = (times[m]-t0)/(t1-t0)
        f = interp1d(u, values[m], kind="linear", fill_value="extrapolate", bounds_error=False)
        U = np.linspace(0,1,n_points)
        curves.append(f(U))
    if not curves: return None, 0
    C = np.vstack(curves)
    return np.nanmean(C,axis=0), C.shape[0]

def plot_time_both(dfA, dfB, out_png, title_suffix=""):
    fig, axes = plt.subplots(2,1,sharex=True,figsize=(10,7))
    for ax, col, title in zip(axes, ["knee_L_deg","knee_R_deg"], ["Knee L","Knee R"]):
        ax.plot(dfA["time_s"], dfA[col], label="Composite", linewidth=2)
        ax.plot(dfB["time_s"], dfB[col], label="Single", linewidth=2, alpha=0.85)
        ax.set_ylabel("knee flexion [deg]")
        ax.set_title(f"{title} (time domain){title_suffix}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_cycle_both(dfA, dfB, evA, evB, out_png):
    UL = np.linspace(0,100,100)
    fig, axes = plt.subplots(2,1,sharex=True,figsize=(10,7))
    for ax, side, col in zip(axes, ["L","R"], ["knee_L_deg","knee_R_deg"]):
        mA, nA = cycle_mean(dfA["time_s"].values, dfA[col].values, load_hs(evA, side))
        mB, nB = cycle_mean(dfB["time_s"].values, dfB[col].values, load_hs(evB, side))
        if (mA is not None) and (mB is not None):
            ax.plot(UL, mA, label=f"Composite mean (n={nA})", linewidth=2)
            ax.plot(UL, mB, label=f"Single mean (n={nB})", linewidth=2, alpha=0.85)
        else:
            ax.text(0.5,0.5,"No HS normalization (events missing)", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("knee flexion [deg]"); ax.grid(True, alpha=0.3)
        ax.set_title(f"Knee {side} (HS-normalized gait cycle)")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("gait cycle [%]")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_csv", required=True, help="joint_angles.csv of Composite")
    ap.add_argument("--b_csv", required=True, help="joint_angles.csv of Single")
    ap.add_argument("--a_events", default=None, help="events.csv of Composite (optional)")
    ap.add_argument("--b_events", default=None, help="events.csv of Single (optional)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dfA = load_angles(args.a_csv)
    dfB = load_angles(args.b_csv)

    # 1) 時間基準の1枚図（上:左膝 / 下:右膝）
    out_time = os.path.join(args.out_dir, "knee_overlay_both_time.png")
    plot_time_both(dfA, dfB, out_time)
    print("[OK]", out_time)

    # 2) 歩行周期基準の1枚図（両方に events がある場合）
    if args.a_events and args.b_events and os.path.exists(args.a_events) and os.path.exists(args.b_events):
        out_cycle = os.path.join(args.out_dir, "knee_overlay_both_cycle.png")
        plot_cycle_both(dfA, dfB, args.a_events, args.b_events, out_cycle)
        print("[OK]", out_cycle)
    else:
        print("[INFO] events.csv が足りないため、周期基準図はスキップしました。")

if __name__ == "__main__":
    main()
