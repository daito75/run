#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HQ version: YAML(K,D) + 歩行動画 → (ArUco複数フレーム) → 実世界平面[m]へのHを頑健推定 → pxへスケール → 各動画を高品質warp
- 歪み補正: CV_32FC1マップ + Lanczos4 (alpha可変)
- H推定: 複数フレーム / サブピクセル / solvePnP(+LM) / 姿勢平均(SVD直交化)
- warp: Lanczos4, 必要に応じてスーパーサンプリング→AREAダウンサンプル


python batch_warp_plane_hq.py --cfg /mnt/d/BRLAB/2025/mizuno/done/run/syaeihenkann/rakusiyoutosita/config_warp.yaml

"""


import argparse
from pathlib import Path 
from typing import Tuple, Optional, List, Dict
import csv
import cv2
import numpy as np
import yaml
import concurrent.futures as futures
from pathlib import Path
import os

# 先頭の import 群のすぐ下に置き換え
import platform, re, yaml

try:
    # もし同ディレクトリ or PYTHONPATH 上に cfgutil.py があればそれを使う
    from cfgutil import load_cfg as _external_load_cfg
    def load_cfg(path): return _external_load_cfg(path)
except Exception:
    # フォールバック: YAMLの {base} 展開 & WSLパス変換を内蔵実装
    _IS_WSL = "microsoft" in platform.uname().release.lower()
    _DRIVE_RE = re.compile(r"^[A-Za-z]:/")  # 例: D:/foo

    def _win_abs_to_wsl(s: str) -> str:
        s = s.replace("\\", "/")
        return f"/mnt/{s[0].lower()}/{s[3:]}"  # "D:/a/b" -> "/mnt/d/a/b"

    def _normalize(s: str) -> str:
        s = s.replace("\\", "/")
        if _IS_WSL and _DRIVE_RE.match(s):
            s = _win_abs_to_wsl(s)
        return s

    def _walk_norm(x):
        if isinstance(x, str): return _normalize(x)
        if isinstance(x, list): return [_walk_norm(v) for v in x]
        if isinstance(x, dict): return {k: _walk_norm(v) for k, v in x.items()}
        return x

    def load_cfg(path):
        path = _normalize(str(path))
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        base = _normalize(cfg.get("base_dir", "."))
        def expand(v):
            if isinstance(v, str): return _normalize(v.replace("{base}", base))
            if isinstance(v, list): return [expand(z) for z in v]
            if isinstance(v, dict): return {k: expand(z) for k, z in v.items()}
            return v
        cfg = expand(cfg)
        cfg.setdefault("common", {})
        cfg.setdefault("cams", [])
        return cfg


# 先頭付近のimportの後あたりに追記
def set_opencv_quiet(quiet: bool):#警告をなくす定義
    if not quiet: #入力で--quietと入れてないと何もなし
        return
    try:#入力で--quietと入れていると以下の処理が行われる
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)#SILENTが完全に黙らせる文
    except Exception:#上記で適応されない場合
        pass  # 古いOpenCVでも無視

def open_video_writer_with_fallback(out_path: Path, fps: float, size_wh: tuple):
    """複数コーデックで順に試す。最後はMJPG(AVI)にフォールバック。"""
    trials = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("H264", ".mp4"),
        ("MJPG", ".avi"),
    ]
    for fourcc, ext in trials:# 各候補を順にチェック
        use_path = out_path
        if ext != out_path.suffix.lower():# 拡張子が違えば差し替え
            use_path = out_path.with_suffix(ext)
        writer = cv2.VideoWriter(# VideoWriterを生成
            str(use_path),
            cv2.VideoWriter_fourcc(*fourcc),# fourccコード指定
            float(fps),
            (int(size_wh[0]), int(size_wh[1])),# 出力サイズ (width,height)
            True
        )
        if writer.isOpened():                 # 正しく開けたら
            return writer, use_path, fourcc   # Writer, 実際のパス, コーデックを返す
    raise RuntimeError(                       # 全部ダメなら例外を出す
        "VideoWriterを開けませんでした（利用可能なコーデックが見つからない）"
    )

# ========== 基本ユーティリティ ==========


def load_cam_yaml(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:#入力したyamlファイルから内部行列Kと歪み係数Dを読み込む
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    K = np.array(y["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(y["dist_coeffs"], dtype=np.float64).ravel()
    return K, D

def build_undistort_maps_hq(K: np.ndarray, D: np.ndarray, size_wh: Tuple[int,int], alpha: float = 0.75):
    """高精度歪み補正用のマップ(newK含む)を作る（CV_32FC1 + alpha>0推奨）"""
    w, h = size_wh
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=alpha)#newK補正後の見え方を反映した新しい内部行列K,ROI黒帯無しの有効領域|ゆがみ補正後は、切り取りや拡大縮小の影響でみかけの内部行列が変化する
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, newK, (w, h), m1type=cv2.CV_32FC1
    )#,map1,map2補正用の座標マップ（元画像のどこから画素を拾うのか）
    return map1, map2, newK, roi

def build_undistort_maps_fisheye(K, D, size_wh, alpha=0.0):
    w,h = size_wh
    # fisheyeはalphaの概念が違う。newKはKを好みに調整
    newK = K.copy()
    R = np.eye(3)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, newK, (w,h), cv2.CV_32FC1
    )
    roi = (0,0,w,h)
    return map1, map2, newK, roi

def parse_wh(s: str) -> Tuple[int,int]:
    w,h = s.lower().split("x")
    return (int(w), int(h))

def parse_roi(s: str) -> Optional[Tuple[int,int,int,int]]:
    if not s:
        return None
    x,y,w,h = [int(v) for v in s.replace(" ", "").split(",")]
    return (x,y,w,h)


def scale_intrinsics(K: np.ndarray, calib_wh: Tuple[int,int], video_wh: Tuple[int,int]) -> np.ndarray:
    sx = video_wh[0] / calib_wh[0]
    sy = video_wh[1] / calib_wh[1]
    K2 = K.copy()
    K2[0,0] *= sx; K2[1,1] *= sy
    K2[0,2] *= sx; K2[1,2] *= sy
    return K2

def _fit_to_screen(img, max_w=1600, max_h=900):
    h, w = img.shape[:2]
    s = min(max_w / max(w,1), max_h / max(h,1), 1.0)
    if s < 1.0:
        disp = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    else:
        disp = img
    return disp, s

class _RoiSelector:
    def __init__(self, img_bgr, win="Select ROI"):
        self.img = img_bgr
        self.disp, self.s = _fit_to_screen(img_bgr)
        self.win = win
        self.pt0 = None
        self.pt1 = None
        self.drawing = False
        self.result = None

    def _on_mouse(self, ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt0 = (x, y)
            self.pt1 = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.pt1 = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.pt1 = (x, y)

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)
        while True:
            canvas = self.disp.copy()
            if self.pt0 and self.pt1:
                x0, y0 = self.pt0; x1, y1 = self.pt1
                cv2.rectangle(canvas, (x0,y0), (x1,y1), (0,255,255), 2)
            cv2.putText(canvas, "Drag to select ROI  |  Enter=OK, C=clear, Esc=cancel",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(self.win, canvas)
            k = cv2.waitKey(16) & 0xFF
            if k in (13, 10):  # Enter
                if self.pt0 and self.pt1:
                    x0, y0 = self.pt0; x1, y1 = self.pt1
                    x, y = min(x0,x1), min(y0,y1)
                    w, h = abs(x1-x0), abs(y1-y0)
                    if w >= 4 and h >= 4:
                        # 画面座標 → 元画像座標へ逆スケール
                        sx = 1.0 / self.s; sy = 1.0 / self.s
                        self.result = (int(round(x*sx)), int(round(y*sy)),
                                       int(round(w*sx)), int(round(h*sy)))
                        break
            elif k in (ord('c'), ord('C')):
                self.pt0 = self.pt1 = None
            elif k == 27:  # Esc
                self.result = None
                break
        cv2.destroyWindow(self.win)
        return self.result

def select_detect_roi_interactive(frame_bgr, title="Select Detect ROI"):
    """frame_bgr: undistort/crop後の画像（検出に使う見え方そのまま）"""
    sel = _RoiSelector(frame_bgr, win=title)
    return sel.run()


def adjust_K_for_crop(K: np.ndarray, roi: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,_,_ = roi
    K2 = K.copy()
    K2[0,2] -= x
    K2[1,2] -= y
    return K2


def undistort_with_maps(img: np.ndarray, map1, map2) -> np.ndarray:
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)#精度優先ならINTER_LANCZSO4,速度優先ならINTER_LINEAR

def get_aruco_dict(name: str):
    table = {
        "4X4_50":  cv2.aruco.DICT_4X4_50,
        "4X4_100": cv2.aruco.DICT_4X4_100,
        "4X4_250": cv2.aruco.DICT_4X4_250,
        "4X4_1000": cv2.aruco.DICT_4X4_1000,
        "5X5_50":  cv2.aruco.DICT_5X5_50,
        "5X5_100": cv2.aruco.DICT_5X5_100,
        "5X5_250": cv2.aruco.DICT_5X5_250,
        "5X5_1000": cv2.aruco.DICT_5X5_1000,
        "6X6_50":  cv2.aruco.DICT_6X6_50,
        "6X6_250": cv2.aruco.DICT_6X6_250,
        "6X6_1000": cv2.aruco.DICT_6X6_1000,
    }
    key = name.strip().upper()
    if key not in table:
        raise ValueError(f"Unsupported ArUco dict: {name}")
    return cv2.aruco.getPredefinedDictionary(table[key])

def detect_aruco(gray, aruco_dict):
    """OpenCV 4.7+ / 旧API 両対応"""
    try:
        det = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
    except Exception:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids

def marker_area(corners) -> float:
    """4点コーナーの投影面積"""
    pts = corners.reshape(-1, 2).astype(np.float32)
    return float(cv2.contourArea(pts))

def pick_largest_marker(corners, ids):
    areas = [marker_area(c) for c in corners]
    idx = int(np.argmax(areas))
    return corners[idx].reshape(-1, 2), int(ids[idx][0]), areas[idx]

def corner_subpix_refine(gray, corners_list, win=(5,5), iters=100, eps=1e-4) -> List[np.ndarray]:
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, eps)
    out = []
    for c in corners_list:
        pts = c.reshape(-1,1,2).astype(np.float32)
        cv2.cornerSubPix(gray, pts, win, (-1,-1), term)
        out.append(pts.reshape(-1,2))
    return out

def H_from_pose(newK: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """画像(undistorted,newK) → 平面[m] のH（スケール正規化）"""
    R, _ = cv2.Rodrigues(rvec)
    r1, r2 = R[:,0], R[:,1]
    G = newK @ np.column_stack([r1, r2, tvec.reshape(3)])  # 3x3
    H = np.linalg.inv(G)
    return H / H[2,2]

def project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.hstack([pts_xy, np.ones((pts_xy.shape[0], 1), np.float64)])
    out = (H @ pts.T).T
    return out[:, :2] / out[:, 2:3]

def make_canvas_and_H_px(H_m: np.ndarray, img_wh: Tuple[int,int], px_per_m: float, margin_m=0.2):
    w, h = img_wh
    img_corners = np.array([[0,0],[w,0],[w,h],[0,h]], np.float64)
    plane_corners_m = project_points(H_m, img_corners)
    xmin, ymin = plane_corners_m.min(axis=0) - margin_m
    xmax, ymax = plane_corners_m.max(axis=0) + margin_m
    W = max(64, int(np.ceil((xmax - xmin) * px_per_m)))
    H = max(64, int(np.ceil((ymax - ymin) * px_per_m)))
    S = np.array([[px_per_m, 0, -xmin*px_per_m],
                  [0, px_per_m, -ymin*px_per_m],
                  [0, 0, 1]], np.float64)
    H_px = S @ H_m
    return H_px, (W, H)

# ========== HQ H推定（複数フレーム, サブピクセル, PnP, 姿勢平均） ==========

def estimate_H_from_video_hq(
    walk_path: Path,
    map1, map2, newK,
    aruco_dict,
    marker_size_m: float,
    scan_step=1,
    scan_max=800,
    candidates_top=10,
    reproj_thresh_px=0.8,
    debug_dir: Optional[Path]=None,
    csv_log: Optional[Path]=None,
    save_best_k: int = 0,  # ★ 追加：良フレームの保存枚数（0で保存しない）
    roi: Optional[Tuple[int,int,int,int]] = None,
    detect_roi: Optional[Tuple[int,int,int,int]] = None,  # ★ 追加
    expected_id: int = -1                                  # ★ 追加
) -> np.ndarray:
    cap = cv2.VideoCapture(str(walk_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open: {walk_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detections = []
    scanned, idx = 0, 0

        # 1) スキャンして候補を集める（画像保存はしない）
    while scanned < scan_max and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        img_ud = undistort_with_maps(frame, map1, map2)
        if roi is not None:
            x0, y0, w_roi, h_roi = roi
            img_ud = img_ud[y0:y0+h_roi, x0:x0+w_roi]

        # --- 検出用ROI（さらに絞る）。座標は img_ud 基準 ---
        det_view = img_ud
        offx = offy = 0
        if detect_roi is not None:
            dx, dy, dw, dh = detect_roi
            det_view = img_ud[dy:dy+dh, dx:dx+dw]
            offx, offy = dx, dy

        gray_full = cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY)  # サブピクセルは元座標で
        gray = cv2.cvtColor(det_view, cv2.COLOR_BGR2GRAY)     # 検出は det_view

        # while ループ内で img_ud / det_view が決まった直後に
        if debug_dir and scanned == 0:
            visdir = (debug_dir / f"{walk_path.stem}_roi_check")
            visdir.mkdir(parents=True, exist_ok=True)
            # undistort(+crop)後のフル画像
            cv2.imwrite(str(visdir / "full_ud.png"), img_ud)
            if detect_roi is not None:
                x,y,w,h = detect_roi
                det_rect = img_ud.copy()
                cv2.rectangle(det_rect, (x,y), (x+w,y+h), (0,255,255), 2)
                cv2.imwrite(str(visdir / "det_view_rect.png"), det_rect)  # 枠だけ描いた全体
                cv2.imwrite(str(visdir / "det_view_crop.png"), det_view)  # 実際に検出に使う切り出し


        corners, ids = detect_aruco(gray, aruco_dict)
        if ids is None or len(ids) == 0:
            scanned += 1; idx += scan_step; continue

        # 期待IDでフィルタ（>=0 のときのみ採用）
        if expected_id >= 0:
            mask = (ids.reshape(-1) == expected_id)
            if not np.any(mask):
                scanned += 1; idx += scan_step; continue
            corners = [corners[i] for i, m in enumerate(mask) if m]
            ids = ids[mask]

        pts_local, _id, area = pick_largest_marker(corners, ids)
        # ROI切り出し分を元座標へ戻す
        pts = pts_local + np.array([offx, offy], np.float32)

        # 小さすぎる四角は除外
        side_lens = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        if float(np.min(side_lens)) < 40.0:
            scanned += 1; idx += scan_step; continue

        detections.append({"idx": idx, "area": area, "pts": pts, "gray": gray_full})

        scanned += 1
        idx += scan_step


    if len(detections) == 0:
        cap.release()
        raise RuntimeError(f"ArUco not found in {walk_path}")

    # 2) 面積上位を選抜
    detections.sort(key=lambda d: d["area"], reverse=True)
    cand = detections[:min(candidates_top, len(detections))]

    # 3) PnP → reprojection誤差
    objp = np.array([
        [0,0,0],
        [marker_size_m,0,0],
        [marker_size_m,marker_size_m,0],
        [0,marker_size_m,0]
    ], np.float32)

    refined = []
    for d in cand:
        gray = d["gray"]
        pts_ref = corner_subpix_refine(gray, [d["pts"]], win=(9,9), iters=300, eps=1e-6)[0].astype(np.float32)
        imgp = pts_ref.reshape(-1,1,2)

        ok = False
        rvec = tvec = None
        err = 1e9

        # まずは平面正方形に強い IPPE_SQUARE を試す
        try:
            ok_ippe, rvec_ippe, tvec_ippe = cv2.solvePnP(objp, imgp, newK, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok_ippe:
                reproj, _ = cv2.projectPoints(objp, rvec_ippe, tvec_ippe, newK, None)
                reproj = reproj.reshape(-1,2)
                err_ippe = float(np.sqrt(np.mean(np.sum((reproj - pts_ref)**2, axis=1))))
                ok, rvec, tvec, err = True, rvec_ippe, tvec_ippe, err_ippe
        except Exception:
            pass

        # IPPEでダメ/誤差大きいときは従来のITERATIVE(+LM)で再挑戦
        if (not ok) or (err > reproj_thresh_px):
            ok_iter, rvec_iter, tvec_iter = cv2.solvePnP(objp, imgp, newK, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok_iter:
                try:
                    rvec_iter, tvec_iter = cv2.solvePnPRefineLM(objp, imgp, newK, None, rvec_iter, tvec_iter)
                except Exception:
                    pass
                reproj2, _ = cv2.projectPoints(objp, rvec_iter, tvec_iter, newK, None)
                reproj2 = reproj2.reshape(-1,2)
                err2 = float(np.sqrt(np.mean(np.sum((reproj2 - pts_ref)**2, axis=1))))
                # どちらが良いかで選択
                if not ok or err2 < err:
                    ok, rvec, tvec, err = True, rvec_iter, tvec_iter, err2

        if ok and err <= reproj_thresh_px:
            d2 = dict(d)
            d2.update({"rvec": rvec.reshape(3), "tvec": tvec.reshape(3), "err": err, "imgp": pts_ref})
            refined.append(d2) 

    # CSV
    if csv_log:
        import csv
        with open(csv_log, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx","area","reproj_err_px"])
            for d in refined:
                w.writerow([d["idx"], f"{d['area']:.3f}", f"{d['err']:.4f}"])

    if len(refined) == 0:
        cap.release()
        raise RuntimeError(f"robust candidates not found (all reproj_err > {reproj_thresh_px}px): {walk_path}")

    # 4) 重み付き姿勢平均
    eps = 1e-9
    weights = np.array([d["area"] / (d["err"]**2 + eps) for d in refined], dtype=np.float64)
    Wsum = float(np.sum(weights))
    weights = weights / Wsum

    R_acc = np.zeros((3,3), np.float64)
    t_acc = np.zeros(3, np.float64)
    for d, w in zip(refined, weights):
        R, _ = cv2.Rodrigues(d["rvec"])
        R_acc += w * R
        t_acc += w * d["tvec"]

    U, S, Vt = np.linalg.svd(R_acc)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    t_avg = t_acc

    # H
    r1, r2 = R_avg[:,0], R_avg[:,1]
    G = newK @ np.column_stack([r1, r2, t_avg])
    H_m = np.linalg.inv(G)
    H_m = H_m / H_m[2,2]

    # ★ 良フレームだけ保存（オプション）
    if debug_dir and save_best_k > 0:
        best_dir = debug_dir / f"{walk_path.stem}_best"
        best_dir.mkdir(parents=True, exist_ok=True)
        # 上位wのフレームidxでソート
        order = np.argsort(weights)[::-1][:save_best_k]
        for rank, i in enumerate(order, start=1):
            d = refined[i]
            vis = cv2.cvtColor(d["gray"], cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, [d["imgp"].astype(int)], True, (0,255,0), 2)
            cv2.putText(vis, f"rank={rank} idx={d['idx']} area={d['area']:.1f} err={d['err']:.3f}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2, cv2.LINE_AA)
            cv2.imwrite(str(best_dir / f"best_{rank:02d}_idx{d['idx']:06d}.png"), vis)

    cap.release()
    return H_m


# ========== 1カメラ処理（HQ） ==========

def process_one_pair_hq(
    yaml_path: Path, walk_path: Path, out_path: Path,
    aruco_dict_name: str, marker_size_cm: float,
    px_per_m: float, alpha: float,
    scan_step: int, scan_max: int, candidates_top: int, reproj_thresh_px: float,
    superres: float,
    debug_dir: Optional[Path], save_csv: bool,
    quiet: bool=False, save_best_k: int=0,
    fisheye: bool=False,
    calib_size_str: str="",        # ★ 追加
    crop_roi: bool=False,           # ★ 追加
    detect_roi_str: str="",          # ★ 追加
    select_detect_roi: bool=False,   # ★ 追加
    select_roi_frame: int=0,         # ★ 追加
    expected_id: int=-1              # ★ 追加  
) -> str:
    K, D = load_cam_yaml(str(yaml_path))

    cap0 = cv2.VideoCapture(str(walk_path))
    if not cap0.isOpened():
        raise FileNotFoundError(f"cannot open: {walk_path}")
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    cap0.release()
        # キャリブ時の画像サイズが指定されていれば、Kを動画サイズへスケール
    if calib_size_str:
        calib_wh = parse_wh(calib_size_str)    # 例: "1920x1080"
        K = scale_intrinsics(K, calib_wh, (w, h))


        # --- ここまで: K,D 読み込み & 動画の w,h,fps 取得 & Kスケール（必要なら） ---
    if fisheye:
        map1, map2, newK, roi = build_undistort_maps_fisheye(K, D, (w, h))
    else:
        map1, map2, newK, roi = build_undistort_maps_hq(K, D, (w, h), alpha=alpha)

    aruco_dict = get_aruco_dict(aruco_dict_name)
    marker_size_m = marker_size_cm / 100.0

    # ---- ここから“必ず” newK_use を決める（ガード付き）----
    # roi が None の場合にも備える
    if roi is None:
        roi = (0, 0, w, h)

    # 引数にデフォルト値を持たせた上でも、ここで再度初期化しておくと安心
    use_roi = bool(crop_roi and roi[2] > 0 and roi[3] > 0)

    # まず newK_use を newK で初期化（最後の保険）
    newK_use = newK
    img_wh_used = (w, h)

    if use_roi:
        # 主点を ROI 原点に合わせる
        newK_use = adjust_K_for_crop(newK, roi)
        img_wh_used = (roi[2], roi[3])

    # （デバッグ用・一度だけ表示したいなら）
    # print("[DEBUG] use_roi=", use_roi, "roi=", roi, "img_wh_used=", img_wh_used)

        # --- ROI/主点補正 決定済み: use_roi, newK_use, img_wh_used ---

    # 1) 文字列で指定があればそれを使う
    detect_roi = parse_roi(detect_roi_str) if detect_roi_str else None

    # 2) GUIで選ぶ指定なら、フレームを1枚読み出して undistort(+crop) 後の絵で選択
    if detect_roi is None and select_detect_roi:
        cap_tmp = cv2.VideoCapture(str(walk_path))
        total_tmp = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx_sel = int(np.clip(select_roi_frame, 0, max(total_tmp-1, 0)))
        cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, idx_sel)
        ok, f0 = cap_tmp.read()
        cap_tmp.release()
        if not ok:
            raise RuntimeError("ROI選択用フレームを取得できませんでした")

        f0_ud = undistort_with_maps(f0, map1, map2)
        if use_roi:
            x0, y0, w_roi, h_roi = roi
            f0_ud = f0_ud[y0:y0+h_roi, x0:x0+w_roi]

        det = select_detect_roi_interactive(f0_ud, title="Select Detect ROI")
        if det is None:
            raise RuntimeError("ROI選択がキャンセルされました")
        detect_roi = det

        if debug_dir:
            (debug_dir / "roi").mkdir(parents=True, exist_ok=True)
            with open(debug_dir / f"{walk_path.stem}_detect_roi.txt", "w", encoding="utf-8") as f:
                x,y,w_,h_ = detect_roi
                f.write(f"{x},{y},{w_},{h_}\n")
 

    # H推定（良フレームだけ保存したい場合は save_best_k を渡す）
    csv_log = (debug_dir / f"{walk_path.stem}_candidates.csv") if (debug_dir and save_csv) else None
    H_m = estimate_H_from_video_hq(
        walk_path, map1, map2, newK_use, aruco_dict, marker_size_m,
        scan_step=scan_step, scan_max=scan_max,
        candidates_top=candidates_top, reproj_thresh_px=reproj_thresh_px,
        debug_dir=debug_dir, csv_log=csv_log, save_best_k=save_best_k,
        roi=roi if use_roi else None,
        detect_roi=detect_roi,                 # ← これ！
        expected_id=expected_id
    )

    # キャンバスとH_px（高解像ワークスペース）
    px_per_m_eff = px_per_m * max(1.0, float(superres))
    H_px, (W_hr, H_hr) = make_canvas_and_H_px(H_m, img_wh_used, px_per_m_eff)

    # ★ 最終出力サイズ（superres>1.0ならダウンサンプルしたサイズ）
    if superres > 1.0:
        W_out = int(round(W_hr / superres))
        H_out = int(round(H_hr / superres))
    else:
        W_out, H_out = int(W_hr), int(H_hr)

    # ★ H.264系は偶数サイズが安全
    if W_out % 2: W_out += 1
    if H_out % 2: H_out += 1

    # ★ Writerを堅牢に開く（コーデック自動フォールバック）
    out_path = Path(out_path)  # 念のためPath化
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, used_path, used_fourcc = open_video_writer_with_fallback(out_path, fps, (W_out, H_out))

    # ★ フレーム処理（warpはHRで→必要ならAREAでダウンサンプル→writerに書く）
    cap = cv2.VideoCapture(str(walk_path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img_ud = undistort_with_maps(frame, map1, map2)
        if use_roi:
            x, y, w_roi, h_roi = roi
            img_ud = img_ud[y:y+h_roi, x:x+w_roi]

        warped_hr = cv2.warpPerspective(
            img_ud, H_px, (W_hr, H_hr),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
        )
        if (W_hr, H_hr) != (W_out, H_out):
            warped = cv2.resize(warped_hr, (W_out, H_out), interpolation=cv2.INTER_AREA)
        else:
            warped = warped_hr

        writer.write(warped)
        i += 1

    writer.release()
    cap.release()

    if not quiet:
        print(f"[Writer] fourcc={used_fourcc}, file={used_path.name}, size={W_out}x{H_out}, frames={i}")
    
    return f"OK[HQ]: {yaml_path.name} -> {Path(used_path).name}  ({i} frames)"

def _file_exists(p: Path) -> bool:
    try:
        return p.exists() and p.is_file()
    except Exception:
        return False

def _build_jobs_from_yaml(C) -> tuple[list[tuple[Path,Path,Path]], dict]:
    """config.yaml の batch_warp セクションから (pairs, defaults) を組み立てる"""
    BW = C.get("batch_warp", {}) or {}
    pairs_cfg = BW.get("pairs", []) or []

    # 共通デフォルト（個別で上書き可）
    defaults = {
        "dict":              BW.get("dict", "4X4_50"),
        "marker_size_cm":    float(BW.get("marker_size_cm", 14.5)),
        "px_per_m":          float(BW.get("px_per_m", 400.0)),
        "alpha":             float(BW.get("alpha", 0.75)),
        "scan_step":         int(BW.get("scan_step", 1)),
        "scan_max":          int(BW.get("scan_max", 800)),
        "candidates_top":    int(BW.get("candidates_top", 10)),
        "reproj_thresh_px":  float(BW.get("reproj_thresh_px", 0.8)),
        "superres":          float(BW.get("superres", 1.0)),
        "debug_dir":         BW.get("debug_dir", ""),
        "save_csv":          bool(BW.get("save_csv", False)),
        "save_best_k":       int(BW.get("save_best_k", 0)),
        "fisheye":           bool(BW.get("fisheye", False)),
        "calib_size":        BW.get("calib_size", ""),
        "crop_roi":          bool(BW.get("crop_roi", False)),
        "select_detect_roi": bool(BW.get("select_detect_roi", False)),
        "select_roi_frame":  int(BW.get("select_roi_frame", 0)),
        "expected_id":       int(BW.get("expected_id", -1)),
        "detect_roi":        BW.get("detect_roi", ""),
        "workers":           int(BW.get("workers", 0)),
    }

    pairs: list[tuple[Path,Path,Path]] = []
    # 個別パラメータは defaults を shallow copy して上書き
    defaults_per_pair: dict[str, dict] = {}

    for cam in pairs_cfg:
        name = cam.get("name", f"pair{len(pairs)+1}")
        y = Path(cam["calib"])
        v = Path(cam["video"])
        o = Path(cam["out"])
        pairs.append((y, v, o))

        d = defaults.copy()
        # 個別上書き（あれば）
        for k in list(d.keys()):
            if k in cam:
                d[k] = cam[k]
        defaults_per_pair[name] = d
        # name をパスにも紐づけておく（キーにする）
        d["_name"] = name

    return pairs, {"defaults": defaults, "per_pair": defaults_per_pair}

def _kwargs_for_pair(name_or_idx, merged_meta, fallback):
    """ペア名があれば per_pair を、無ければ共通 defaults を返す"""
    per = merged_meta["per_pair"]
    if isinstance(name_or_idx, str) and name_or_idx in per:
        return per[name_or_idx]
    return merged_meta["defaults"]

def main():
    ap = argparse.ArgumentParser(description="HQ版: 各カメラを実世界平面へ高精度warp（合成なし）")
    # 追加: YAML
    ap.add_argument("--cfg", default="",
                    help="設定YAML（batch_warp セクションから複数ジョブ実行）")
    # 既存 CLI
    ap.add_argument("--pair", action="append",
                    help="yaml:walk:out の組（複数可）例: cam1.yaml:cam1.mp4:cam1_plane.mp4")
    ap.add_argument("--dict", default="4X4_50", help="ArUco辞書（例: 4X4_50, 5X5_1000 等）")
    ap.add_argument("--marker-size-cm", type=float, default=14.5, help="ArUco一辺の物理サイズ[cm]")
    ap.add_argument("--px-per-m", type=float, default=400.0, help="仮想平面スケール[px/m]")
    ap.add_argument("--alpha", type=float, default=0.75, help="getOptimalNewCameraMatrixのalpha(0~1)")
    ap.add_argument("--scan-step", type=int, default=1, help="検出スキャン間隔（フレーム）")
    ap.add_argument("--scan-max", type=int, default=800, help="スキャン最大数（小さすぎると見逃し）")
    ap.add_argument("--candidates-top", type=int, default=10, help="面積上位の候補フレーム数")
    ap.add_argument("--reproj-thresh-px", type=float, default=0.8, help="再投影誤差の外れ閾値[px]")
    ap.add_argument("--superres", type=float, default=1.0, help="スーパーサンプリング倍率(>=1.0)")
    ap.add_argument("--debug-dir", default="", help="デバッグ出力先(検出画像/CSV/メタyaml)")
    ap.add_argument("--save-csv", action="store_true", help="候補フレームのCSVを保存")
    ap.add_argument("--workers", type=int, default=0, help="並列数（0/1なら逐次、2以上で並列）")
    ap.add_argument("--quiet", action="store_true", help="ログを最小化（OpenCVログも抑制）")
    ap.add_argument("--save-best-k", type=int, default=0, help="良フレームの保存枚数（0=保存しない）")
    ap.add_argument("--fisheye", action="store_true", help="fisheyeモデルで補正する")
    ap.add_argument("--calib-size", default="", help="キャリブ時の画像サイズ (例 1920x1080)。指定時はKをスケール")
    ap.add_argument("--crop-roi", action="store_true", help="undistort後にROIでクロップして使用（端を使わない）")
    ap.add_argument("--select-detect-roi", action="store_true",
                    help="GUIで検出用ROI（四角）をマウス選択してから実行する")
    ap.add_argument("--select-roi-frame", type=int, default=0,
                    help="ROI選択に使うフレーム番号（0=先頭）")
    ap.add_argument("--expected-id", type=int, default=-1,
                    help="期待するArUco ID（指定>=0ならそのIDのみ採用）")
    ap.add_argument("--detect-roi", default="",
                    help="検出用ROIを x,y,w,h で指定（undistort/crop後の座標系）")

    args = ap.parse_args()
    set_opencv_quiet(args.quiet)
    print(f"[INFO] argv --cfg = {args.cfg}")

    yaml_pairs = []
    yaml_meta = None
    if args.cfg:
        try:
            C = load_cfg(args.cfg)
            # ★ ここからデバッグ出力
            print("[INFO] load_cfg OK")
            print(f"[INFO] keys at top-level: {list(C.keys())}")
            BW = C.get("batch_warp", {})
            print(f"[INFO] batch_warp exists: {bool(BW)}")
            if BW:
                pairs_cfg = BW.get("pairs", [])
                print(f"[INFO] pairs in YAML: {len(pairs_cfg)}")
            # ★ ここまでデバッグ出力
            yaml_pairs, yaml_meta = _build_jobs_from_yaml(C)
            print(f"[INFO] yaml_pairs built: {len(yaml_pairs)}")
        except Exception as e:
            print(f"[ERROR] load_cfg failed: {e}")
            yaml_pairs, yaml_meta = [], None

    # 2) CLIの --pair 直指定があればそれも読む
    cli_pairs = parse_pairs(args.pair) if args.pair else []
    if cli_pairs:
        print(f"[INFO] cli_pairs: {len(cli_pairs)}")

    # 3) 実行ペアを統合
    pairs = []
    if yaml_pairs: pairs.extend(yaml_pairs)
    if cli_pairs:  pairs.extend(cli_pairs)

    print(f"[INFO] total scheduled jobs = {len(pairs)}")
    if not pairs:
        print("[WARN] No jobs found. Check --cfg path and YAML structure (batch_warp.pairs).")
        return  # ★ここで明示終了


    # 4) デフォルト引数（CLIベース）
    common_kwargs_cli = dict(
        aruco_dict_name=args.dict, marker_size_cm=args.marker_size_cm,
        px_per_m=args.px_per_m, alpha=args.alpha,
        scan_step=args.scan_step, scan_max=args.scan_max,
        candidates_top=args.candidates_top, reproj_thresh_px=args.reproj_thresh_px,
        superres=args.superres, debug_dir=(Path(args.debug_dir) if args.debug_dir else None),
        save_csv=args.save_csv, quiet=args.quiet, save_best_k=args.save_best_k,
        fisheye=args.fisheye, calib_size_str=args.calib_size, crop_roi=args.crop_roi,
        detect_roi_str=args.detect_roi, select_detect_roi=args.select_detect_roi,
        select_roi_frame=args.select_roi_frame, expected_id=args.expected_id,
    )

    # 5) ワーカー数（YAML優先）
    workers = args.workers
    if yaml_meta:
        workers = int(yaml_meta["defaults"].get("workers", workers))

    # 6) 実行
    def _run_one(idx, y, v, o):
        # 6-1) YAMLペア個別上書きをマージ
        kwargs = common_kwargs_cli.copy()
        if yaml_meta:
            # name（yaml側で付けたやつ）を推定
            # out パス名から逆引き、なければデフォルト
            guessed = os.path.splitext(o.name)[0]
            per = yaml_meta["per_pair"]
            # per は name をキーに持つ。name を out の stem に一致させるのがおすすめ。
            if guessed in per:
                d = per[guessed]
            else:
                # name キーを総当たりで calib/video/out のどれかが一致すればそれを採用
                d = None
                for name, meta in per.items():
                    # 粗い一致でOK（out 末尾だけ見て判定）
                    if str(o).endswith(name) or name in str(o):
                        d = meta; break
                if d is None:
                    d = yaml_meta["defaults"]
            # 上書き反映
            kwargs.update(
                aruco_dict_name      = d.get("dict", kwargs["aruco_dict_name"]),
                marker_size_cm       = float(d.get("marker_size_cm", kwargs["marker_size_cm"])),
                px_per_m             = float(d.get("px_per_m", kwargs["px_per_m"])),
                alpha                = float(d.get("alpha", kwargs["alpha"])),
                scan_step            = int(d.get("scan_step", kwargs["scan_step"])),
                scan_max             = int(d.get("scan_max", kwargs["scan_max"])),
                candidates_top       = int(d.get("candidates_top", kwargs["candidates_top"])),
                reproj_thresh_px     = float(d.get("reproj_thresh_px", kwargs["reproj_thresh_px"])),
                superres             = float(d.get("superres", kwargs["superres"])),
                debug_dir            = (Path(d["debug_dir"]) if d.get("debug_dir") else kwargs["debug_dir"]),
                save_csv             = bool(d.get("save_csv", kwargs["save_csv"])),
                save_best_k          = int(d.get("save_best_k", kwargs["save_best_k"])),
                fisheye              = bool(d.get("fisheye", kwargs["fisheye"])),
                calib_size_str       = d.get("calib_size", kwargs["calib_size_str"]),
                crop_roi             = bool(d.get("crop_roi", kwargs["crop_roi"])),
                select_detect_roi    = bool(d.get("select_detect_roi", kwargs["select_detect_roi"])),
                select_roi_frame     = int(d.get("select_roi_frame", kwargs["select_roi_frame"])),
                expected_id          = int(d.get("expected_id", kwargs["expected_id"])),
                detect_roi_str       = d.get("detect_roi", kwargs["detect_roi_str"]),
            )

        # 6-2) 入力の存在チェック（無ければスキップ）
        if not _file_exists(y):
            print(f"[WARN] skip (no calib yaml): {y}")
            return
        if not _file_exists(v):
            print(f"[WARN] skip (no video): {v}")
            return

        try:
            msg = process_one_pair_hq(
                y, v, o,
                kwargs["aruco_dict_name"], kwargs["marker_size_cm"],
                kwargs["px_per_m"], kwargs["alpha"],
                kwargs["scan_step"], kwargs["scan_max"], kwargs["candidates_top"], kwargs["reproj_thresh_px"],
                kwargs["superres"],
                kwargs["debug_dir"], kwargs["save_csv"],
                quiet=kwargs["quiet"], save_best_k=kwargs["save_best_k"],
                fisheye=kwargs["fisheye"], calib_size_str=kwargs["calib_size_str"], crop_roi=kwargs["crop_roi"],
                detect_roi_str=kwargs["detect_roi_str"], select_detect_roi=kwargs["select_detect_roi"],
                select_roi_frame=kwargs["select_roi_frame"], expected_id=kwargs["expected_id"]
            )
            print(msg)
        except FileNotFoundError as e:
            print(f"[WARN] skip (open fail): {e}")
        except RuntimeError as e:
            print(f"[WARN] skip (runtime): {e}")
        except Exception as e:
            print(f"[ERROR] unexpected at {v}: {e}")

    if workers and workers > 1:
        with futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for i, (y, v, o) in enumerate(pairs):
                futs.append(ex.submit(_run_one, i, y, v, o))
            for f in futures.as_completed(futs):
                pass
    else:
        for i, (y, v, o) in enumerate(pairs):
            _run_one(i, y, v, o)

if __name__ == "__main__":
    main()