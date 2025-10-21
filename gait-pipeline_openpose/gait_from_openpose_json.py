#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPose (BODY_25) の --write_json 出力から歩行指標を計算し、
HS/TO を検出して CSV に書き出し、イベントを動画に重畳するスクリプト。
- ケイデンス、ステップ時間、立脚率、（任意で）ステップ長[m]
- イベントCSV: frame, time_sec, side(L/R), event(HS/TO), x_px, y_px
使い方例:
python3 gait_from_openpose_json.py \
    --json_dir "/mnt/d/BRLAB/2025/openpose_out/0LR/1/json" \
    --video_in "/mnt/d/BRLAB/2025/mizuno/done/deta/kaiseki2/concat/a0/0LR/1.mp4" \
    --video_overlay "/mnt/d/BRLAB/2025/openpose_out/0LR/1/HSTO/mywalk_pose_overlay.mp4" \
    --out_dir "/mnt/d/BRLAB/2025/openpose_out/0LR/1/HSTO/gaitevents" \
    --scale_m_per_px 0.005
"""
import argparse, json, glob, os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import cv2

# BODY_25 foot indices
L_BIG, L_SML, L_HEEL = 19, 20, 21
R_BIG, R_SML, R_HEEL = 22, 23, 24

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--video_in", required=True, help="元の歩行動画（fps取得用/オーバーレイ用）")
    ap.add_argument("--video_overlay", default=None, help="OpenPoseが出した骨格オーバーレイ動画（これに重畳したい場合）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conf_th", type=float, default=0.3)
    ap.add_argument("--speed_th", type=float, default=3.0, help="接地判定の速度閾値 [px/frame]")
    ap.add_argument("--min_hold", type=int, default=4, help="立脚/遊脚の最小持続フレーム")
    ap.add_argument("--scale_m_per_px", type=float, default=None, help="m/px（わかる場合のみ。ステップ長[m]計算に使用）")
    return ap.parse_args()

def load_seq(json_dir):
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    seq = []
    for f in files:
        J = json.load(open(f, "r"))
        if not J.get("people"):
            seq.append(None); continue
        # 最も信頼度が高いpersonを採用
        best = max(J["people"], key=lambda p: sum(p["pose_keypoints_2d"][2::3]))
        k = np.array(best["pose_keypoints_2d"]).reshape(-1,3)  # (25,3)
        seq.append(k)
    return seq

def foot_xy(pt, side, conf_th):
    if side=='L':
        pts = [pt[L_BIG], pt[L_SML], pt[L_HEEL]]
    else:
        pts = [pt[R_BIG], pt[R_SML], pt[R_HEEL]]
    cs = [p[2] for p in pts]
    if np.median(cs) < conf_th: return np.nan, np.nan
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return float(np.median(xs)), float(np.median(ys))

def interp_nan(a):
    a = np.asarray(a, dtype=float)
    n = len(a); idx = np.arange(n)
    mask = np.isfinite(a)
    if mask.sum() < 2: return a
    a[~mask] = np.interp(idx[~mask], idx[mask], a[mask])
    return a

def smooth(arr, win=7, poly=2):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < win:
        return arr
    # 修正ポイント: win_length → window_length
    return savgol_filter(arr, window_length=win, polyorder=poly)

def enforce_min_run(mask, min_len):
    m = mask.astype(int)
    i=0
    while i < len(m):
        if m[i]==1:
            j=i
            while j<len(m) and m[j]==1: j+=1
            if (j-i) < min_len: m[i:j]=0
            i=j
        else:
            i+=1
    return m.astype(bool)

def edges(stance):
    s = stance.astype(int)
    hs = np.where((s[1:]==1)&(s[:-1]==0))[0]+1  # rising -> HS
    to = np.where((s[1:]==0)&(s[:-1]==1))[0]+1  # falling -> TO
    return hs, to

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # FPS取得
    cap0 = cv2.VideoCapture(args.video_in)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    seq = load_seq(args.json_dir)
    nF  = len(seq)

    # 足中心軌跡抽出
    xs_L, ys_L, xs_R, ys_R = [], [], [], []
    for k in seq:
        if k is None:
            xs_L.append(np.nan); ys_L.append(np.nan)
            xs_R.append(np.nan); ys_R.append(np.nan)
        else:
            xL,yL = foot_xy(k,'L',args.conf_th)
            xR,yR = foot_xy(k,'R',args.conf_th)
            xs_L.append(xL); ys_L.append(yL)
            xs_R.append(xR); ys_R.append(yR)

    # 補間＋平滑
    xs_L = smooth(interp_nan(xs_L))
    ys_L = smooth(interp_nan(ys_L))
    xs_R = smooth(interp_nan(xs_R))
    ys_R = smooth(interp_nan(ys_R))

    # 速度ベクトル
    vL = np.hypot(np.diff(xs_L,prepend=xs_L[0]),
                  np.diff(ys_L,prepend=ys_L[0]))
    vR = np.hypot(np.diff(xs_R,prepend=xs_R[0]),
                  np.diff(ys_R,prepend=ys_R[0]))

    # 接地マスク
    stanceL = enforce_min_run(vL < args.speed_th, args.min_hold)
    stanceR = enforce_min_run(vR < args.speed_th, args.min_hold)

    # HS/TO
    HS_L, TO_L = edges(stanceL)
    HS_R, TO_R = edges(stanceR)

    # 指標計算
    def intervals(ev):  # 同側HS-HS
        return np.diff(ev)/fps if len(ev)>=2 else np.array([])
    strideL = intervals(HS_L)
    strideR = intervals(HS_R)

    # ステップ時間（L->R）
    step_times = []
    for hL in HS_L:
        nxtR = HS_R[HS_R>hL]
        if len(nxtR): step_times.append( (nxtR[0]-hL)/fps )
    step_times = np.array(step_times)

    cadence = float(60.0 / np.nanmean(step_times)) if len(step_times) else np.nan

    def stance_ratio(HS, TO, strides):
        if len(HS)==0 or len(TO)==0 or len(strides)==0: return np.nan
        m = min(len(HS), len(TO), len(strides))
        nums = []
        for i in range(m-1):  # HS[i]~HS[i+1]の周期
            on  = HS[i]
            off = TO[i] if TO[i]>on else TO[i+1] if i+1<len(TO) else None
            if off is None: continue
            T = (HS[i+1]-HS[i])/fps
            nums.append(((off-on)/fps)/T)
        return float(np.nanmean(nums)) if nums else np.nan

    stance_ratio_L = stance_ratio(HS_L, TO_L, strideL)
    stance_ratio_R = stance_ratio(HS_R, TO_R, strideR)

    # ステップ長（ピクセル → 任意でm）
    # 画像x方向を歩行方向と仮定（側面視）
    step_px = []
    for hL in HS_L:
        nxtR = HS_R[HS_R>hL]
        if len(nxtR):
            step_px.append(abs(xs_R[nxtR[0]] - xs_L[hL]))
    step_px = np.array(step_px, dtype=float)
    step_m = step_px*args.scale_m_per_px if (args.scale_m_per_px and len(step_px)) else None

    # 結果サマリ
    summary = {
        "fps": fps,
        "frames": nF,
        "mean_step_time_s": float(np.nanmean(step_times)) if len(step_times) else np.nan,
        "cadence_steps_per_min": cadence,
        "stride_time_L_s": float(np.nanmean(strideL)) if len(strideL) else np.nan,
        "stride_time_R_s": float(np.nanmean(strideR)) if len(strideR) else np.nan,
        "stance_ratio_L": stance_ratio_L,
        "stance_ratio_R": stance_ratio_R,
        "mean_step_length_px": float(np.nanmean(step_px)) if len(step_px) else np.nan,
        "scale_m_per_px": args.scale_m_per_px if args.scale_m_per_px else None,
        "mean_step_length_m": float(np.nanmean(step_m)) if step_m is not None else None
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, "metrics_summary.csv"), index=False)

    # イベントCSV
    rows = []
    for f in HS_L: rows.append([int(f), f/fps, "L", "HS", float(xs_L[f]), float(ys_L[f])])
    for f in TO_L: rows.append([int(f), f/fps, "L", "TO", float(xs_L[f]), float(ys_L[f])])
    for f in HS_R: rows.append([int(f), f/fps, "R", "HS", float(xs_R[f]), float(ys_R[f])])
    for f in TO_R: rows.append([int(f), f/fps, "R", "TO", float(xs_R[f]), float(ys_R[f])])
    df = pd.DataFrame(rows, columns=["frame","time_sec","side","event","x_px","y_px"]).sort_values("frame")
    df.to_csv(os.path.join(args.out_dir, "events.csv"), index=False)

    print("=== Summary ===")
    for k,v in summary.items():
        print(f"{k}: {v}")

    # 動画へのオーバーレイ（OpenPoseのoverlay動画が指定されていればそっちを優先）
    vis_src = args.video_overlay if (args.video_overlay and os.path.exists(args.video_overlay)) else args.video_in
    cap = cv2.VideoCapture(vis_src)
    out_path = os.path.join(args.out_dir, "overlay_with_events.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # フレーム→イベント辞書
    ev_by_f = {}
    for _,row in df.iterrows():
        ev_by_f.setdefault(int(row["frame"]), []).append(row)

    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        # イベントがあるフレームにマーカー/テキスト
        if fidx in ev_by_f:
            for row in ev_by_f[fidx]:
                x,y = int(row["x_px"]), int(row["y_px"])
                color = (0,255,0) if row["side"]=="L" else (0,128,255)
                text  = f'{row["side"]}-{row["event"]}'
                cv2.circle(frame, (x,y), 7, color, -1)
                cv2.putText(frame, text, (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            # タイムライン目印
            cv2.line(frame, (0,40), (frame.shape[1],40), (255,255,255), 1)
            cv2.putText(frame, f"t={fidx/fps:.2f}s", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        out.write(frame)
        fidx += 1

    cap.release(); out.release()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
