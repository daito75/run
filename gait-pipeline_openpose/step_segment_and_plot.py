#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 step_segment_and_plot.py \
  --json_dir "/mnt/d/BRLAB/2025/mizuno/done/deta/kaiseki4/openpose/3.0m/60cm/json" \
  --out_dir  "/mnt/d/BRLAB/2025/mizuno/done/deta/kaiseki4/openpose/3.0m/60cm/gaitevents_steps" \
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

def robust_assign_feet_from_jsondir(
    json_dir,
    conf_th=0.3,
    k_swap=3,
    max_jump_px=50.0,
    margin=5.0,
    speed_w=0.5,
    conf_w=0.0,
    prefer_consistency=True,
):
    """
    OpenPose JSON ディレクトリから、右足/左足のトラックを頑健に復元し、
    右左の入れ替わりを抑える。

    Parameters
    ----------
    json_dir : str or Path
        OpenPoseの JSON が入っているディレクトリ。
    conf_th : float
        つま先/踵/MID_HIP を「使う」かの信頼度しきい値。
    k_swap : int
        「左右スワップの方が自然」という判定が連続で起きた回数の閾値。
        大きいほど“確信が持てたときだけ”スワップし、誤スワップを抑える。
        反面、反応は鈍くなる（2～4あたりから調整推奨）。
    max_jump_px : float
        1フレームで許容する移動距離(px)。これを超える移動にはペナルティを課す。
        歩行速度・撮影条件・fpsに応じて 50～120px の範囲で調整が目安。
    margin : float
        “そのまま”と“スワップ”のコスト差が僅差の時のブレ止めマージン(px)。
        ノイズが多い映像ほど 5～10px 程度に上げると安定。
    speed_w : float
        速度連続性ペナルティの重み。0なら速度ペナルティなし。
        0.3～1.0 程度で調整すると暴れを抑制できる。
    conf_w : float
        信頼度の低い観測へペナルティを与える重み。
        0.0～0.5 程度。0のままでもよい（OpenPoseが安定なら0でOK）。
    prefer_consistency : bool
        “直前フレームの割当を維持すること”を弱く優先する小バイアスを付ける。
        True 推奨（Falseでも動作はする）。

    Returns
    -------
    dict
        {
          "x_r","y_r","x_l","y_l",   # 右/左 足の時系列（NaN を含む）
          "x_mid","y_mid",           # MID_HIP の時系列（NaN を含む）
          "frames", "n"              # 参照フレーム名一覧、総フレーム数
        }
    """
    from pathlib import Path
    import numpy as np, json

    json_dir = Path(json_dir)
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"[ERR] JSONなし: {json_dir}")

    def _avg_foot(pts, toe_idx, heel_idx, th):
        """つま先/踵のうち、信頼度が閾値以上の点だけ平均して足位置を作る"""
        toe = pts[toe_idx]; heel = pts[heel_idx]
        xs, ys, cs = [], [], []
        if toe[2] >= th:  xs.append(toe[0]); ys.append(toe[1]); cs.append(toe[2])
        if heel[2] >= th: xs.append(heel[0]); ys.append(heel[1]); cs.append(heel[2])
        if xs:
            return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(cs))
        return np.nan, np.nan, 0.0

    def _dist(a, b):
        """ユークリッド距離（どちらか NaN を含めば np.inf を返す）"""
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
            return np.inf
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _cost(prev_pos, curr_pos, prev_vel, conf=1.0):
        """
        単点の割当コスト：
         - 位置の移動距離
         - 大ジャンプ(>max_jump_px)のペナルティ
         - 速度連続性（前フレーム速度との差）ペナルティ（重み speed_w）
         - 低信頼度ペナルティ（重み conf_w）
        """
        # 位置距離
        d = _dist(prev_pos, curr_pos)
        if not np.isfinite(d):
            return 1e6  # 観測欠損や prev欠損は超高コスト

        # 大ジャンプペナルティ（距離を増幅）
        if d > max_jump_px:
            d *= 4.0

        # 速度連続性：v_t = curr - prev_pos
        if np.all(np.isfinite(prev_pos)) and np.all(np.isfinite(curr_pos)) and np.all(np.isfinite(prev_vel)):
            v = np.array([curr_pos[0]-prev_pos[0], curr_pos[1]-prev_pos[1]], dtype=float)
            dv = v - prev_vel
            d += float(speed_w) * float(np.hypot(dv[0], dv[1]))

        # 信頼度ペナルティ（conf ∈ [0,1] を想定、低いほどペナルティ）
        if conf_w > 0.0:
            d += float(conf_w) * (1.0 - float(conf))

        return d

    # 観測の収集 ----------------------------------------------------------
    obs_r = []  # (x,y,conf)
    obs_l = []
    mids  = []  # (x,y,conf)

    for p in files:
        data = json.loads(p.read_text())
        if not data.get("people"):
            obs_r.append((np.nan,np.nan,0.0))
            obs_l.append((np.nan,np.nan,0.0))
            mids.append((np.nan,np.nan,0.0))
            continue
        pts = np.array(data["people"][0]["pose_keypoints_2d"], dtype=np.float32).reshape(-1,3)
        if pts.shape[0] < 25:
            obs_r.append((np.nan,np.nan,0.0))
            obs_l.append((np.nan,np.nan,0.0))
            mids.append((np.nan,np.nan,0.0))
            continue

        xr, yr, cr = _avg_foot(pts, R_BIGTOE, R_HEEL, conf_th)
        xl, yl, cl = _avg_foot(pts, L_BIGTOE, L_HEEL, conf_th)
        mid = pts[MID_HIP]
        xm = float(mid[0]) if mid[2] >= conf_th else np.nan
        ym = float(mid[1]) if mid[2] >= conf_th else np.nan
        cm = float(mid[2])

        obs_r.append((xr, yr, cr))
        obs_l.append((xl, yl, cl))
        mids.append((xm, ym, cm))

    obs_r = np.array(obs_r, dtype=float)   # shape (N,3)
    obs_l = np.array(obs_l, dtype=float)
    mids  = np.array(mids,  dtype=float)

    N = len(files)
    x_r = np.full(N, np.nan); y_r = np.full(N, np.nan)
    x_l = np.full(N, np.nan); y_l = np.full(N, np.nan)

    # 初期化：始動フレーム（両足が一応読めた最初のフレーム）を探す
    t0 = 0
    while t0 < N and (not np.isfinite(obs_r[t0,0]) or not np.isfinite(obs_l[t0,0])):
        t0 += 1
    if t0 == N:
        # 全欠損なら空で返す
        return dict(x_r=x_r,y_r=y_r,x_l=x_l,y_l=y_l,
                    x_mid=mids[:,0], y_mid=mids[:,1],
                    frames=[p.name for p in files], n=N)

    # 観測ラベルをそのまま採用して開始
    x_r[t0], y_r[t0] = obs_r[t0,0], obs_r[t0,1]
    x_l[t0], y_l[t0] = obs_l[t0,0], obs_l[t0,1]

    # 直前速度（最初はゼロベクトル）
    v_r_prev = np.array([0.0, 0.0], dtype=float)
    v_l_prev = np.array([0.0, 0.0], dtype=float)

    swap_streak = 0  # 連続で「入れ替え案が有利」のカウント

    for t in range(t0+1, N):
        r_meas = obs_r[t,:2]; l_meas = obs_l[t,:2]
        cr, cl = obs_r[t,2], obs_l[t,2]
        prev_r = np.array([x_r[t-1], y_r[t-1]])
        prev_l = np.array([x_l[t-1], y_l[t-1]])

        # 直前速度の更新（前フレームの確定位置から）
        if np.all(np.isfinite(prev_r)) and np.all(np.isfinite([x_r[t-2] if t-2>=0 else np.nan, y_r[t-2] if t-2>=0 else np.nan])):
            v_r_prev = prev_r - np.array([x_r[t-2], y_r[t-2]])
        if np.all(np.isfinite(prev_l)) and np.all(np.isfinite([x_l[t-2] if t-2>=0 else np.nan, y_l[t-2] if t-2>=0 else np.nan])):
            v_l_prev = prev_l - np.array([x_l[t-2], y_l[t-2]])

        # 欠損ケースを先に処理 -------------------------------------------
        both_nan = not np.isfinite(r_meas[0]) and not np.isfinite(l_meas[0])
        if both_nan:
            # 何も観測がない → そのまま NaN
            continue

        only_r = np.isfinite(r_meas[0]) and (not np.isfinite(l_meas[0]))
        only_l = np.isfinite(l_meas[0]) and (not np.isfinite(r_meas[0]))
        if only_r:
            # 右のみ観測 → 右or左のどちらに繋ぐのが自然かで割当
            keep_r_cost = _cost(prev_r, r_meas, v_r_prev, conf=cr)
            swap_l_cost = _cost(prev_l, r_meas, v_l_prev, conf=cr)
            if swap_l_cost + margin < keep_r_cost:
                # 左に来た方が自然 → （見かけ上の入れ替え発生中とみなす）
                x_l[t], y_l[t] = r_meas
            else:
                x_r[t], y_r[t] = r_meas
            continue

        if only_l:
            keep_l_cost = _cost(prev_l, l_meas, v_l_prev, conf=cl)
            swap_r_cost = _cost(prev_r, l_meas, v_r_prev, conf=cl)
            if swap_r_cost + margin < keep_l_cost:
                x_r[t], y_r[t] = l_meas
            else:
                x_l[t], y_l[t] = l_meas
            continue

        # どちらも観測あり → “そのまま”vs“スワップ”の合計コスト比較 ------
        cost_keep = (
            _cost(prev_r, r_meas, v_r_prev, conf=cr) +
            _cost(prev_l, l_meas, v_l_prev, conf=cl)
        )
        cost_swap = (
            _cost(prev_r, l_meas, v_r_prev, conf=cl) +
            _cost(prev_l, r_meas, v_l_prev, conf=cr)
        )

        # 連続優位判定（僅差は margin で無視）
        if cost_swap + margin < cost_keep:
            swap_streak += 1
        else:
            # 一度でも keep が優位になればリセット
            swap_streak = 0

        # “直前の割当を維持”の弱いバイアス：微妙な局面でのフリップ抑制
        if prefer_consistency and abs(cost_swap - cost_keep) < margin:
            swap_streak = 0

        if swap_streak >= k_swap:
            # スワップ確定（このフレームのみ）
            x_r[t], y_r[t] = l_meas
            x_l[t], y_l[t] = r_meas
            swap_streak = 0
        else:
            # 現状維持
            x_r[t], y_r[t] = r_meas
            x_l[t], y_l[t] = l_meas

    return dict(
        x_r=x_r, y_r=y_r, x_l=x_l, y_l=y_l,
        x_mid=mids[:,0], y_mid=mids[:,1],
        frames=[p.name for p in files], n=N
    )

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

    # robust foot assignment (入れ替わり補正付き)
    feet = robust_assign_feet_from_jsondir(
    args.json_dir,
    conf_th=args.conf_th,
    k_swap=3,         # ★調整ノブ：2～4 が目安
    max_jump_px=80.0, # ★調整ノブ：50～120px
    margin=5.0,       # ★調整ノブ：3～10px
    speed_w=0.5,      # ★調整ノブ：0.0～1.0（0で無効）
    conf_w=0.0,       # ★調整ノブ：0.0～0.5（低信頼度ペナルティ）
    prefer_consistency=True
)

    if args.side.lower().startswith("r"):
        x_foot = feet["x_r"]    
    else:
        x_foot = feet["x_l"]
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
