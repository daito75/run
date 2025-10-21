#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#動画上にわかりやすく，点を表示．その表示とリアルの距離関係を入力することにより射影変換を実行している．

"""
python D:\BRLAB\2025\mizuno\done\run\syaeihenkann\batch_warp_plane_manual.py `
  --pair "D:\BRLAB\2025\mizuno\done\deta\kaiseki3\frame\cam2\cam2_1080p.yaml|D:\BRLAB\2025\mizuno\done\deta\kaiseki3\walk\normal\cam2_walk.mp4|D:\BRLAB\2025\mizuno\done\deta\kaiseki3\syaeihenkann\normal\cam2_on_plane.mp4" `
  --px-per-m 200 `
  --alpha 1.0 `
  --superres 1.0 `
  --calib-size 1920x1080 `
  --select-frame 0 `
  --debug-dir "D:\BRLAB\2025\mizuno\done\deta\kaiseki3\syaeihenkann\manual" `
  --progress-interval-sec 15 `
  --fisheye `
  --reuse-clicks

  python D:\BRLAB\2025\mizuno\done\run\syaeihenkann\manual.py `
  --pair "D:\BRLAB\2025\mizuno\done\deta\kaiseki4\frame\cam1\cam1_1080p.yaml|D:\BRLAB\2025\mizuno\done\deta\kaiseki4\walk\3.0m\1m\cam1_walk.mp4|cam1_on_plane.mp4" `
  --px-per-m 200 --alpha 1.0 --superres 1.0 --calib-size 1920x1080 `
  --select-frame 0 --fisheye --reuse-clicks `
  --out-dir "D:\BRLAB\2025\mizuno\done\deta\kaiseki4\syaeihenkann\3.0m\1m" `
  --clicks-root "D:\BRLAB\2025\mizuno\done\deta\kaiseki4\syaeihenkann\clicks"

  --reuse-clicks `



cam1
[0.00, 0.00],   #左下LB
[2.00, 0.00],   #右下RB
[2.00, 1.00],   #右上RT
[0.00, 1.00],   #左上LT

"""

import argparse
from pathlib import Path
from typing import Tuple, Optional, List
import time
import yaml
import cv2
import numpy as np

# ===== 実世界[m]座標（クリック順に対応） =====
WORLD_POINTS_M = np.array([
[0.00, 0.00],   #左下LB
[2.00, 0.00],   #右下RB
[2.00, 1.00],   #右上RT
[0.00, 1.00],   #左上LT
], dtype=np.float64)

# ===== 進捗ログ =====
class _ProgressTicker:
    def __init__(self, total: int, interval_sec: float = 2.0):
        self.total = max(int(total), 0)
        self.interval = float(interval_sec)
        self.start = time.time()
        self.last = self.start
    def tick(self, done: int, prefix: str = ""):
        now = time.time()
        if (now - self.last) < self.interval:
            return None
        self.last = now
        elapsed = now - self.start
        done = max(0, int(done))
        pct = (100.0 * done / self.total) if self.total > 0 else 0.0
        fps = (done / elapsed) if elapsed > 0 else 0.0
        eta = ((self.total - done) / fps) if (fps > 0 and self.total > 0) else float("inf")
        eta_txt = "ETA: --" if np.isinf(eta) else time.strftime("ETA: %H:%M:%S", time.gmtime(int(eta+0.5)))
        el_txt  = time.strftime("elapsed: %H:%M:%S", time.gmtime(int(elapsed+0.5)))
        return f"{prefix} {done}/{self.total} ({pct:5.1f}%) | {fps:5.1f} fps | {el_txt} | {eta_txt}"

# ===== 基本ユーティリティ =====
def set_opencv_quiet(quiet: bool):
    if not quiet:
        return
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass

def load_cam_yaml(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    K = np.array(y["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(y["dist_coeffs"], dtype=np.float64).ravel()
    return K, D

def parse_wh(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x")
    return (int(w), int(h))

def scale_intrinsics(K: np.ndarray, calib_wh: Tuple[int, int], video_wh: Tuple[int, int]) -> np.ndarray:
    sx = video_wh[0] / calib_wh[0]
    sy = video_wh[1] / calib_wh[1]
    K2 = K.copy()
    K2[0,0] *= sx; K2[1,1] *= sy
    K2[0,2] *= sx; K2[1,2] *= sy
    return K2

def build_undistort_maps(K, D, size_wh, alpha=0.75, fisheye=False):
    w, h = size_wh
    if not fisheye:
        # 通常モデル：32Fマップ + getOptimalNewCameraMatrix(alpha)
        newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=alpha)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, newK, (w, h), cv2.CV_32FC1
        )
    else:
        # 魚眼モデル：fisheye 用のマップを使う（alpha の概念は異なるので newK=K を基本に）
        newK = K.copy()
        R = np.eye(3)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, R, newK, (w, h), cv2.CV_32FC1
        )
        roi = (0, 0, w, h)  # fisheye では getOptimalNewCameraMatrix の ROI は無いので全面
    return map1, map2, newK, roi


def adjust_K_for_crop(K: np.ndarray, roi):
    x, y, _, _ = roi
    K2 = K.copy()
    K2[0,2] -= x; K2[1,2] -= y
    return K2

def undistort_with_maps(img, map1, map2):
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

def make_canvas_and_H_px(H_m: np.ndarray, img_wh: Tuple[int,int],
                         px_per_m: float, margin_m=0.2, y_up: bool = True):
    w, h = img_wh
    img_corners = np.array([[0,0],[w,0],[w,h],[0,h]], np.float64)
    plane_corners_m = project_points(H_m, img_corners)
    xmin, ymin = plane_corners_m.min(axis=0) - margin_m
    xmax, ymax = plane_corners_m.max(axis=0) + margin_m

    W = max(64, int(np.ceil((xmax-xmin) * px_per_m)))
    Hh= max(64, int(np.ceil((ymax-ymin) * px_per_m)))

    if not y_up:
        # （従来）画像と同じ y↓
        S = np.array([[ px_per_m, 0,        -xmin * px_per_m],
                      [ 0,        px_per_m, -ymin * px_per_m],
                      [ 0,        0,         1             ]], np.float64)
    else:
        # 世界は y↑ → 出力で上下反転して画像の y↓に合わせる
        # y_pix = -px_per_m * (y_m - ymax) = -px*y_m + px*ymax
        S = np.array([[ px_per_m,  0,        -xmin * px_per_m],
                      [ 0,        -px_per_m,  ymax * px_per_m],
                      [ 0,         0,          1             ]], np.float64)

    H_px = S @ H_m
    return H_px, (W, Hh)


def project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.hstack([pts_xy, np.ones((pts_xy.shape[0], 1), np.float64)])
    out = (H @ pts.T).T
    return out[:, :2] / out[:, 2:3]

def open_video_writer_with_fallback(out_path: Path, fps: float, size_wh: tuple):
    trials = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("H264", ".mp4"), ("MJPG", ".avi")]
    for fourcc, ext in trials:
        use_path = out_path.with_suffix(ext) if ext != out_path.suffix.lower() else out_path
        writer = cv2.VideoWriter(str(use_path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (int(size_wh[0]), int(size_wh[1])), True)
        if writer.isOpened():
            return writer, use_path, fourcc
    raise RuntimeError("VideoWriterを開けませんでした")

# ===== クリック UI =====
class ClickCollector:
    def __init__(self, img_bgr, title="Click corresponding points (Enter=confirm, r=reset, Esc=cancel)"):
        self.img = img_bgr
        self.title = title
        self.pts = []

    def _on_mouse(self, ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))

    def run(self, need_n: int):
        disp = self.img.copy()
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self._on_mouse)
        guide = "Order: Bottom-Left → Bottom-Right → Top-Right → Top-Left (Enter=confirm, r=reset, Esc=cancel)"

        while True:
            canvas = np.ascontiguousarray(disp.copy())  # prevent memory layout errors
            for i, (x, y) in enumerate(self.pts):
                cv2.circle(canvas, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(canvas, f"P{i}", (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, guide, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(self.title, canvas)
            k = cv2.waitKey(16) & 0xFF
            if k in (13, 10):  # Enter
                if len(self.pts) >= need_n:
                    break
            elif k in (ord('r'), ord('R')):
                self.pts.clear()
            elif k == 27:  # Esc
                self.pts = None
                break

        cv2.destroyWindow(self.title)
        return None if self.pts is None else np.array(self.pts, np.float64)



# ===== H 推定（手動対応点→実世界[m]） =====
"""
def find_H_manual(img_pts: np.ndarray, world_pts_m: np.ndarray) -> np.ndarray:
    assert img_pts.shape[0] == world_pts_m.shape[0] and world_pts_m.shape[0] >= 4
    world_pts = world_pts_m.astype(np.float64)
    H, _ = cv2.findHomography(img_pts.astype(np.float64), world_pts, 0 if len(img_pts)==4 else cv2.RANSAC, 1.5)
    if H is None:
        raise RuntimeError("findHomography 失敗（点配置を見直してください）")
    return H

"""
def find_H_manual(img_pts: np.ndarray, world_pts_m: np.ndarray) -> np.ndarray:
    """
    img_pts: 形状 (N,2) の画像座標（undistort+ROI後）。クリック順が WORLD_POINTS_M と対応。
    world_pts_m: 形状 (N,2) の実世界[m]座標。少なくとも4点。ここでは N=4 を想定して厳密解。
    """
    assert img_pts.shape[0] >= 4 and img_pts.shape[0] == world_pts_m.shape[0]
    N = img_pts.shape[0]
    if N != 4:
        # 4点以外なら従来の findHomography を使う（もしくは最小二乗で拡張）
        H, _ = cv2.findHomography(img_pts.astype(np.float64), world_pts_m.astype(np.float64), 0, 0)
        if H is None:
            raise RuntimeError("findHomography 失敗（点配置を見直してください）")
        return H

    # 4点の別名
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = img_pts.astype(np.float64)
    (x0t, y0t), (x1t, y1t), (x2t, y2t), (x3t, y3t) = world_pts_m.astype(np.float64)

    # あなたの数式と同じ 8元連立
    A = np.array([
        [x0, y0, 1, 0, 0, 0, -x0*x0t, -y0*x0t],
        [0, 0, 0, x0, y0, 1, -x0*y0t, -y0*y0t],
        [x1, y1, 1, 0, 0, 0, -x1*x1t, -y1*x1t],
        [0, 0, 0, x1, y1, 1, -x1*y1t, -y1*y1t],
        [x2, y2, 1, 0, 0, 0, -x2*x2t, -y2*x2t],
        [0, 0, 0, x2, y2, 1, -x2*y2t, -y2*y2t],
        [x3, y3, 1, 0, 0, 0, -x3*x3t, -y3*x3t],
        [0, 0, 0, x3, y3, 1, -x3*y3t, -y3*y3t]
    ], dtype=np.float64)
    b = np.array([x0t, y0t, x1t, y1t, x2t, y2t, x3t, y3t], dtype=np.float64)

    # 逆行列ではなく solve で安定に解く
    sol = np.linalg.solve(A, b)

    # H（src→dst）を組む（最後の h22=1）
    H = np.array([
        [sol[0], sol[1], sol[2]],
        [sol[3], sol[4], sol[5]],
        [sol[6], sol[7], 1.0]
    ], dtype=np.float64)
    return H


# ===== メイン処理（1ペア） =====
def process_one_pair_manual(
    yaml_path: Path, walk_path: Path, out_path: Path,
    px_per_m: float, alpha: float, superres: float,
    calib_size_str: str, select_frame: int,
    reuse_clicks: bool,
    clicks_root: Optional[Path],  # クリック保存ルート（cam別）
    out_dir: Optional[Path],      # 動画保存ルート（条件フォルダ）
    trial_dir: Optional[Path],    # out-dir未指定時のみ使用（互換）
    group_by_cam: bool,           # out-dirの下をcam別サブフォルダにする
    progress_interval_sec: float, quiet: bool=False,
    fisheye: bool=False,
) -> str:
        # ===== 保存レイアウト =====

    # カメラ名（例: "cam3_walk" -> "cam3"）
    cam_name = Path(walk_path).stem.split("_")[0]

    # クリックtxt: clicks_root/camX/clicks.txt
    click_txt = None
    if clicks_root:
        cam_dir = clicks_root / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        click_txt = cam_dir / "clicks.txt"

    # 出力動画の保存先:
    #   - --out-dir があればそこを最優先
    #   - なければ trial_dir（従来互換）
    #   - 最後の手段として pairの3番目の親パス
    out_name = Path(out_path).name
    base_dir = None
    if out_dir is not None:
        base_dir = out_dir
    elif trial_dir is not None:
        base_dir = trial_dir
    else:
        parent = Path(out_path).parent
        base_dir = parent if str(parent) not in (".", "") else Path.cwd()

    if group_by_cam:
        base_dir = base_dir / cam_name

    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = (base_dir / out_name).resolve()


    # 1) 動画情報
    cap0 = cv2.VideoCapture(str(walk_path))
    if not cap0.isOpened():
        raise FileNotFoundError(f"cannot open: {walk_path}")
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 2) キャリブ & undistort
    K, D = load_cam_yaml(str(yaml_path))
    if calib_size_str:
        K = scale_intrinsics(K, parse_wh(calib_size_str), (w, h))
    map1, map2, newK, roi = build_undistort_maps(K, D, (w, h), alpha=alpha, fisheye=fisheye)
    if roi is None:
        roi = (0, 0, w, h)
    x, y, rw, rh = roi
    if rw <= 1 or rh <= 1:
        # ROI が無効ならフル画像を使う
        roi = (0, 0, w, h)


    img_pts = None
    if reuse_clicks and click_txt and click_txt.exists():
        vals = []
        for ln in click_txt.read_text(encoding="utf-8").strip().splitlines():
            x, y = ln.strip().split(",")
            vals.append([float(x), float(y)])
        img_pts = np.asarray(vals, np.float64)


    if img_pts is None:
        cap0.set(cv2.CAP_PROP_POS_FRAMES, int(np.clip(select_frame, 0, max(total_frames-1,0))))
        ok, f0 = cap0.read()
        if not ok: raise RuntimeError("選択フレーム取得に失敗")
        f0_ud = undistort_with_maps(f0, map1, map2)
        # ROI クロップ（getOptimalNewCameraMatrixの有効領域）
        x, y, w_roi, h_roi = roi
        # ROI が無効な場合（幅または高さが0以下）→ フル画像に置き換え
        if w_roi <= 1 or h_roi <= 1:        
            x, y, w_roi, h_roi = 0, 0, f0_ud.shape[1], f0_ud.shape[0]

        # クロップ＋copyで連続メモリ化
        f0_ud = f0_ud[y:y+h_roi, x:x+w_roi].copy()

        # クリックUI起動
        clicker = ClickCollector(f0_ud, title="Pick corresponding points")
        img_pts = clicker.run(need_n=WORLD_POINTS_M.shape[0])
        if img_pts is None:     
            raise RuntimeError("クリックがキャンセルされました")

        # クリック座標の保存
        if click_txt:
            with open(click_txt, "w", encoding="utf-8") as f:
                for (px, py) in img_pts:
                    f.write(f"{px:.3f},{py:.3f}\n")


    cap0.release()

    # 4) H[m] 推定（undistort+ROI 後の画像座標 → 実世界[m]）
    H_m = find_H_manual(img_pts, WORLD_POINTS_M)

    # 5) キャンバス設計 & H_px
    px_per_m_eff = px_per_m * max(1.0, float(superres))
    img_wh_used = (roi[2], roi[3])
    H_px, (W_hr, H_hr) = make_canvas_and_H_px(H_m, img_wh_used, px_per_m_eff)

    # 6) 出力 Writer
    W_out = int(round(W_hr / superres)) if superres > 1.0 else int(W_hr)
    H_out = int(round(H_hr / superres)) if superres > 1.0 else int(H_hr)
    if W_out % 2: W_out += 1
    if H_out % 2: H_out += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, used_path, used_fourcc = open_video_writer_with_fallback(out_path, fps, (W_out, H_out))

    # 7) 全フレーム warp（射影変換の本体）
    cap = cv2.VideoCapture(str(walk_path))
    progress = _ProgressTicker(total_frames if total_frames>0 else 1, progress_interval_sec)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        img_ud = undistort_with_maps(frame, map1, map2)
        x,y,w_roi,h_roi = roi; img_ud = img_ud[y:y+h_roi, x:x+w_roi]

        warped_hr = cv2.warpPerspective(
            img_ud, H_px, (W_hr, H_hr),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
        )
        out_img = warped_hr if (W_hr, H_hr)==(W_out, H_out) else cv2.resize(warped_hr, (W_out, H_out), interpolation=cv2.INTER_AREA)
        writer.write(out_img)
        i += 1
        msg = progress.tick(i, prefix="[warp]")
        if msg and not quiet: print(msg)

    writer.release(); cap.release()
    if not quiet:
        print(f"[Writer] fourcc={used_fourcc}, file={used_path.name}, size={W_out}x{H_out}, frames={i}")
    return f"OK[Manual]: {yaml_path.name} -> {Path(used_path).name} ({i} frames)"

# ===== CLI =====
def parse_pairs(pairs):
    out = []
    for p in pairs:
        s = p.strip().strip('"').strip("'")
        if "|" in s:
            y, v, o = s.split("|")
        else:
            y, v, o = s.rsplit(":", 2)
        out.append((Path(y), Path(v), Path(o)))
    return out

def main():
    ap = argparse.ArgumentParser(description="手動対応点 + 固定ワールド座標で射影変換（ArUco不使用）")
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--px-per-m", type=float, default=200.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--superres", type=float, default=1.0)
    ap.add_argument("--calib-size", default="")
    ap.add_argument("--select-frame", type=int, default=0)
    ap.add_argument("--reuse-clicks", action="store_true")
    ap.add_argument("--debug-dir", default="debug")
    ap.add_argument("--progress-interval-sec", type=float, default=10.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--fisheye", action="store_true")
    ap.add_argument("--trial", type=int, default=0)

    # ★ここに追加する！！
    ap.add_argument("--out-dir", default="", help="射影変換動画の保存先フォルダ（例: ...\\syaeihenkann\\normal）")
    ap.add_argument("--clicks-root", default="", help="クリック点（txt）の保存ルート（例: ...\\syaeihenkann\\clicks）")
    ap.add_argument("--group-by-cam", action="store_true", help="出力動画をカメラ別サブフォルダ（camX/）に振り分ける")

    args = ap.parse_args()


    set_opencv_quiet(args.quiet)
    pairs = parse_pairs(args.pair)

    if args.clicks_root:
        clicks_root = Path(args.clicks_root).resolve()
    elif args.debug_dir:
        clicks_root = Path(args.debug_dir).resolve()
    else:
        clicks_root = Path("manual").resolve()
    clicks_root.mkdir(parents=True, exist_ok=True)

    # 動画出力ルート（--out-dir 優先）
    out_dir = None
    trial_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] out dir    : {out_dir}")
    else:
        # 従来の trial 自動採番
        manual_root = Path(args.debug_dir).resolve() if args.debug_dir else Path("manual").resolve()
        if args.trial and args.trial > 0:
            trial_dir = manual_root / str(args.trial)
            trial_dir.mkdir(parents=True, exist_ok=True)
        else:
            trial_dir = next_trial_dir(manual_root)
        print(f"[INFO] manual root: {manual_root}")
        print(f"[INFO] trial dir  : {trial_dir}")

    # ★ 保存先が決まってから、処理ループへ
    for (y, v, o) in pairs:
        # out名だけ使うので、ここでは trial_dir を使っても out-dir を使ってもOK（中で分岐）
        o_trial = Path(o).name  # ← out-dir を使うので名前だけで十分

    # manual ルート（--debug-dir を manual 指定に流用）
    manual_root = Path(args.debug_dir).resolve() if args.debug_dir else Path("manual").resolve()

    # trial フォルダ決定
    if args.trial and args.trial > 0:
        trial_dir = manual_root / str(args.trial)
        trial_dir.mkdir(parents=True, exist_ok=True)
    else:
        trial_dir = next_trial_dir(manual_root)

    print(f"[INFO] manual root: {manual_root}")
    print(f"[INFO] trial dir  : {trial_dir}")

    for (y, v, o) in pairs:
        o_trial = trial_dir / Path(o).name  # 出力は試行フォルダへ

        msg = process_one_pair_manual(
            yaml_path=y, walk_path=v, out_path=o_trial,
            px_per_m=args.px_per_m,
            alpha=args.alpha, superres=args.superres,
            calib_size_str=args.calib_size, select_frame=args.select_frame,
            reuse_clicks=args.reuse_clicks,
            clicks_root=clicks_root,
            out_dir=out_dir,                 # --out-dir があればこちら
            trial_dir=trial_dir,             # なければ従来trial
            group_by_cam=args.group_by_cam,  # 任意
            progress_interval_sec=args.progress_interval_sec,
            quiet=args.quiet,
            fisheye=args.fisheye,
        )
        print(msg)




def next_trial_dir(base_dir: Path) -> Path:
    """
    base_dir 直下に 1,2,3,... の連番フォルダを自動生成して返す。
    例: base_dir=.../manual -> .../manual/1, .../manual/2 など
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in base_dir.iterdir():
        if p.is_dir():
            try:
                nums.append(int(p.name))
            except ValueError:
                pass
    n = max(nums) + 1 if nums else 1
    trial = base_dir / str(n)
    trial.mkdir(parents=True, exist_ok=True)
    return trial



if __name__ == "__main__":
    main()
