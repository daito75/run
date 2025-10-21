#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成動画の「繋ぎ目」健全性を検証するスクリプト。
- 入力: A(Composite) と B(Normal) の joint_angles.csv / events.csv / (任意) json_dir
- 指定した seam 時刻の前後 ±window_s で、膝角度の連続性/誤差、HS/TO数、(任意)信頼度を比較
- 出力: 図(PNG) + 指標まとめCSV

合成動画（Composite＝A）と通常動画（Single＝B）を比べて、指定した「繋ぎ目時間」の前後で関節角度や歩行イベントがどれだけ“なめらか”かを定量＋可視化

python3 validate_seam_robustness.py \
  --a_angles /mnt/d/BRLAB/2025/openpose_out/Composite/1/angles/joint_angles.csv \
  --a_events /mnt/d/BRLAB/2025/openpose_out/Composite/1/gaitevents/events.csv \
  --b_angles /mnt/d/BRLAB/2025/openpose_out/Normal/1/angles/joint_angles.csv \
  --b_events /mnt/d/BRLAB/2025/openpose_out/Normal/1/gaitevents/events.csv \
  --a_json_dir /mnt/d/BRLAB/2025/openpose_out/Composite/1/json \
  --b_json_dir /mnt/d/BRLAB/2025/openpose_out/Normal/1/json \
  --seams 2.5 \
  --window_s 0.5 \
  --fps_est 30 \
  --out_dir /mnt/d/BRLAB/2025/openpose_out/compare/seam_validation

  "D:\BRLAB\2025\mizuno\done\deta\kaiseki2\concat\a0\0Composite\1.mp4"

  
ffprobe -v error -show_entries format=duration \
  -show_entries stream=avg_frame_rate,r_frame_rate \
  -of default=nw=1:nk=1 "/mnt/d/BRLAB/2025/mizuno/done/deta/kaiseki2/concat/a0/0Composite/1.mp4"

ffprobe -v error -show_entries format=duration \
  -show_entries stream=avg_frame_rate,r_frame_rate \
  -of default=nw=1:nk=1 "/mnt/d/BRLAB/2025/mizuno/done/deta/kaiseki2/concat/a0/Normal/1.mp4"


"""
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, glob, json

# -------------------- I/O --------------------
def load_angles(p):
    df = pd.read_csv(p)
    need = {"time_s","knee_L_deg","knee_R_deg"}
    if not need.issubset(df.columns):
        raise ValueError(f"missing columns in {p}: need {need}")
    return df

def load_events(p):
    if p and os.path.exists(p):
        df = pd.read_csv(p)
        need = {"time_sec","side","event"}
        if need.issubset(df.columns):
            return df
    return None

def load_conf_series(json_dir, conf_idx_list=(10,11,13,14)):
    """
    json_dir から各フレームの平均信頼度(右膝・右足首・左膝・左足首)を計算して時系列化。
    戻り: DataFrame[frame, mean_conf, t]  ※ tは不明なので frame を返す（秒にしたいときは外で合わせる）
    """
    if not json_dir or not os.path.isdir(json_dir):
        return None
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not files:
        return None
    rows=[]
    for i,f in enumerate(files):
        try:
            J = json.load(open(f,"r"))
            ppl = J.get("people",[])
            if not ppl:
                rows.append((i, np.nan)); continue
            # 最もスコア高い人
            best = max(ppl, key=lambda p: np.nansum(np.array(p["pose_keypoints_2d"],float)[2::3]))
            k = np.array(best["pose_keypoints_2d"], float).reshape(-1,3)
            confs = [k[idx,2] for idx in conf_idx_list if idx < k.shape[0]]
            rows.append((i, float(np.nanmean(confs))))
        except Exception:
            rows.append((i, np.nan))
    return pd.DataFrame(rows, columns=["frame","mean_conf"])

# -------------------- metrics --------------------
def window_mask(t, center, w):
    return (t >= center - w) & (t <= center + w)

def jump_metrics(t, y, seam, w):
    """繋ぎ目中心の L/R 角度列から不連続性を測る: 
       ・直前直後の平均差 (pre_mean, post_mean, delta_mean) 
       ・最大1階差(速度)のピーク 
       ・RMSE(前半vs後半)
    """
    m = window_mask(t, seam, w)
    if m.sum() < 6:
        return dict(valid=False)
    tt = t[m]; yy = y[m]
    mid = np.argmin(np.abs(tt - seam))
    pre = yy[:mid]; post = yy[mid:]
    if len(pre)<3 or len(post)<3:
        return dict(valid=False)
    delta_mean = float(np.nanmean(post) - np.nanmean(pre))
    # 1階差（擬似速度）
    dy = np.diff(yy) / np.maximum(np.diff(tt), 1e-6)
    vel_peak = float(np.nanmax(np.abs(dy)))
    # RMSE between mirrored segments（長さ合わせ）
    n = min(len(pre), len(post))
    rmse = float(np.sqrt(np.nanmean((post[:n] - pre[-n:])**2)))
    return dict(valid=True, delta_mean_deg=delta_mean, vel_peak_deg_per_s=vel_peak, rmse_deg=rmse)

def event_metrics(ev, seam, w, side):
    """繋ぎ目前後の HS/TO カウントとタイミング"""
    if ev is None:
        return dict(valid=False)
    m = (ev["time_sec"] >= seam - w) & (ev["time_sec"] <= seam + w) & (ev["side"]==side)
    seg = ev[m]
    n_hs = int((seg["event"]=="HS").sum())
    n_to = int((seg["event"]=="TO").sum())
    return dict(valid=True, HS=n_hs, TO=n_to)

def conf_metrics(conf_df, fps_est, seam, w):
    """jsonの平均信頼度がある場合、繋ぎ目前後の平均/落ち込みを見る"""
    if conf_df is None or fps_est is None or fps_est<=0:
        return dict(valid=False)
    # frame -> time 近似
    t = conf_df["frame"].values / float(fps_est)
    c = conf_df["mean_conf"].values
    m = window_mask(t, seam, w)
    if m.sum() < 4: 
        return dict(valid=False)
    pre = c[(t >= seam - w) & (t < seam)]
    post= c[(t >  seam) & (t <= seam + w)]
    if len(pre)==0 or len(post)==0:
        return dict(valid=False)
    return dict(valid=True,
                pre_mean=float(np.nanmean(pre)),
                post_mean=float(np.nanmean(post)),
                delta_mean=float(np.nanmean(post) - np.nanmean(pre)))

# -------------------- plotting --------------------
# 置き換え：AとBで別々の時間軸を扱う
def plot_overlay(ax, tA, yA, tB, yB, seam, title):
    ax.plot(tA, yA, label="Composite", linewidth=2)
    ax.plot(tB, yB, label="Single", linewidth=2, alpha=0.85)
    ax.axvline(seam, color="k", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("knee flexion [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_diff(ax, tA, yA, tB, yB, seam, title):
    import numpy as np
    from scipy.interpolate import interp1d
    # 共通の時間範囲
    t0 = max(np.nanmin(tA), np.nanmin(tB))
    t1 = min(np.nanmax(tA), np.nanmax(tB))
    if not np.isfinite([t0,t1]).all() or t1 <= t0:
        ax.text(0.5,0.5,"insufficient overlap", ha="center", va="center", transform=ax.transAxes)
        return
    # 共通グリッド（密度は小さめでOK）
    T = np.linspace(t0, t1, min(max(len(tA), len(tB)), 800))
    fA = interp1d(tA, yA, kind="linear", fill_value="extrapolate", bounds_error=False)
    fB = interp1d(tB, yB, kind="linear", fill_value="extrapolate", bounds_error=False)
    D = fA(T) - fB(T)
    ax.plot(T, D, linewidth=1.5)
    ax.axhline(0, color="k", linewidth=1)
    ax.axvline(seam, color="k", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("A - B [deg]")
    ax.grid(True, alpha=0.3)


def draw_events(ax, ev_path, color, linestyle, t_min, t_max, side):
    import pandas as pd
    if not ev_path: return
    ev = pd.read_csv(ev_path)
    seg = ev[(ev.side==side) & (ev.time_sec>=t_min) & (ev.time_sec<=t_max)]
    for _,r in seg.iterrows():
        c = color
        if r.event == "HS":
            ax.axvline(r.time_sec, color=c, linestyle=linestyle, linewidth=1.2, alpha=0.9)
        elif r.event == "TO":
            ax.axvline(r.time_sec, color=c, linestyle=linestyle, linewidth=1.2, alpha=0.45)

def phase_metrics(ev, t_min, t_max, side):
    import numpy as np, pandas as pd
    ev = ev[(ev.side==side) & (ev.time_sec>=t_min) & (ev.time_sec<=t_max)].sort_values("time_sec")
    hs = ev[ev.event=="HS"]["time_sec"].values
    to = ev[ev.event=="TO"]["time_sec"].values
    if len(hs) < 2: return dict(n_stride=0)
    # 1周期ずつ
    strides = []
    for i in range(len(hs)-1):
        t0, t1 = hs[i], hs[i+1]
        # t0–t1の間のTO（最初のTOを採用）
        to_i = to[(to>t0) & (to<t1)]
        if len(to_i)==0: continue
        T = t1 - t0
        stance = (to_i[0]-t0)/T
        strides.append((T, stance))
    if not strides: return dict(n_stride=0)
    S = np.array(strides)
    return dict(n_stride=len(S), mean_stride_s=S[:,0].mean(), mean_stance_ratio=float(S[:,1].mean()))



# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_angles", required=True)     # A: Composite joint_angles.csv
    ap.add_argument("--a_events", required=True)     # A: events.csv
    ap.add_argument("--b_angles", required=True)     # B: Normal joint_angles.csv
    ap.add_argument("--b_events", required=True)     # B: events.csv
    ap.add_argument("--a_json_dir", default=None)    # 任意: A json dir
    ap.add_argument("--b_json_dir", default=None)    # 任意: B json dir
    ap.add_argument("--seams", required=True, help="繋ぎ目の秒をカンマ区切り 例: 12.3,28.7")
    ap.add_argument("--window_s", type=float, default=1.0, help="繋ぎ目前後の評価幅 [s]")
    ap.add_argument("--fps_est", type=float, default=30.0, help="json→time 変換の仮fps")
    ap.add_argument("--out_dir", required=True)
    # argparse 定義のところに追加
    ap.add_argument("--show_events", action="store_true",
                help="Enable drawing HS/TO event lines on plots (off by default)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seams = [float(s) for s in args.seams.split(",")]

    A = load_angles(args.a_angles)
    B = load_angles(args.b_angles)
    evA = load_events(args.a_events)
    evB = load_events(args.b_events)
    confA = load_conf_series(args.a_json_dir)
    confB = load_conf_series(args.b_json_dir)

    rows=[]
    for side, col in [("L","knee_L_deg"), ("R","knee_R_deg")]:
        tA = A["time_s"].values; yA = A[col].values
        tB = B["time_s"].values; yB = B[col].values
        for seam in seams:
            # 図：重ね＋差
            import matplotlib.pyplot as plt
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,7), sharex=False)
            # Composite 側を +0.05 秒シフト
            shift = -0.05
            tA_shifted = tA + shift

            mA = window_mask(tA_shifted, seam, args.window_s)
            mB = window_mask(tB, seam, args.window_s)

            # 切り出し
            tA_seg, yA_seg = tA_shifted[mA], yA[mA]
            tB_seg, yB_seg = tB[mB],          yB[mB]

            # グラフ描画
            plot_overlay(ax1, tA_shifted[mA], yA[mA],tB[mB], yB[mB], seam,
                f"Knee {side} around seam {seam:.2f}s (±{args.window_s}s)")
            plot_diff(ax2, tA_shifted[mA], yA[mA],tB[mB], yB[mB], seam,
                f"Difference (Composite - Single)")


            if args.show_events:
                # 可視化する時間範囲（シフト後の切り出し区間でOK）
                t0 = float(min(tA_seg.min(), tB_seg.min()))
                t1 = float(max(tA_seg.max(), tB_seg.max()))
                if evA is not None:
                    draw_events(ax1, args.a_events, "tab:blue",  "-",  t0, t1, side)   # Composite: 実線
                if evB is not None:
                    draw_events(ax1, args.b_events, "tab:orange","--", t0, t1, side)   # Single: 破線

                # 凡例の追加（イベント線を描くときだけ）
                ax1.plot([], [], color="tab:blue",  linestyle="-",  label="HS/TO (Composite)")
                ax1.plot([], [], color="tab:orange",linestyle="--", label="HS/TO (Single)")
                ax1.legend(loc="upper right")

            # 差分プロット
            plot_diff(ax2, tA[mA], yA[mA],tB[mB], yB[mB], seam, "Difference (Composite - Single)")
            ax2.set_xlabel("time [s]")
            fig.tight_layout()
            out_png = os.path.join(args.out_dir, f"seam_{seam:.2f}_knee_{side}.png")
            fig.savefig(out_png, dpi=150); plt.close(fig)

            # 数値メトリクス
            jmA = jump_metrics(tA, yA, seam, args.window_s)
            jmB = jump_metrics(tB, yB, seam, args.window_s)
            emA = event_metrics(evA, seam, args.window_s, side)
            emB = event_metrics(evB, seam, args.window_s, side)
            cmA = conf_metrics(confA, args.fps_est, seam, args.window_s)
            cmB = conf_metrics(confB, args.fps_est, seam, args.window_s)

            rows.append({
                "side":side, "seam_s":seam,
                # A(合成)の連続性
                "A_delta_mean_deg": jmA.get("delta_mean_deg", np.nan),
                "A_vel_peak_deg_per_s": jmA.get("vel_peak_deg_per_s", np.nan),
                "A_rmse_deg_pre_vs_post": jmA.get("rmse_deg", np.nan),
                "A_HS": emA.get("HS", np.nan), "A_TO": emA.get("TO", np.nan),
                "A_conf_pre": cmA.get("pre_mean", np.nan), "A_conf_post": cmA.get("post_mean", np.nan),
                "A_conf_delta": cmA.get("delta_mean", np.nan),
                # B(普通)の連続性（比較基準）
                "B_delta_mean_deg": jmB.get("delta_mean_deg", np.nan),
                "B_vel_peak_deg_per_s": jmB.get("vel_peak_deg_per_s", np.nan),
                "B_rmse_deg_pre_vs_post": jmB.get("rmse_deg", np.nan),
                "B_HS": emB.get("HS", np.nan), "B_TO": emB.get("TO", np.nan),
                "B_conf_pre": cmB.get("pre_mean", np.nan), "B_conf_post": cmB.get("post_mean", np.nan),
                "B_conf_delta": cmB.get("delta_mean", np.nan),
            })

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "seam_validation_summary.csv"), index=False)
    print("[DONE] results ->", args.out_dir)

if __name__ == "__main__":
    main()
