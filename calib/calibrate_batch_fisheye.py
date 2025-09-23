#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibrate_batch.py (fisheye対応)
複数カメラの picked フォルダから一気にキャリブレーションして YAML 出力。

機能:
  - チェスボ検出 (SB優先→従来法フォールバック)
  - モデル選択: "pinhole" (RATIONAL_MODEL) / "fisheye"
  - 1回目の誤差分布から高誤差ビューを一定率トリム（RMSが改善した場合のみ採用）
  - new_camera_matrix と ROI を計算し YAML へ出力
    * pinhole: getOptimalNewCameraMatrix(alpha=1.0)
    * fisheye: newK=K.copy(), roi=(0,0,w,h) を保存（fisheyeにはαの概念がないため）
  - 使用画像一覧・各ビューの誤差も YAML に保存

使い方:
  python calibrate_batch.py
"""

import os, glob, yaml
import numpy as np
import cv2

# ========= ここに各カメラごとの設定を並べる =========
# model: "pinhole" or "fisheye" を指定
INPUTS = [
    {
        "name": "cam1",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\picked\*.png",
        "save": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\cam1_1080p.yaml",
        "checker": (7, 6),
        "square_mm": 30.0,
        "min_keep": 20,
        "trim_frac": 0.12,
        "model": "fisheye",   # ★ 従来pinhole
    },
    {
        "name": "cam2",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam2\picked\*.png",
        "save": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam2\cam2_1080p.yaml",
        "checker": (7, 6),
        "square_mm": 30.0,
        "min_keep": 20,
        "trim_frac": 0.12,
        "model": "fisheye",   # ★ 魚眼
    },

    {
        "name": "cam3",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam3\picked\*.png",
        "save": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam3\cam3_1080p.yaml",
        "checker": (7, 6),
        "square_mm": 30.0,
        "min_keep": 20,
        "trim_frac": 0.12,
        "model": "fisheye",   # ★ 魚眼
    },
    # 以降、必要に応じて追加
]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
flags_cla = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
use_sb = hasattr(cv2, "findChessboardCornersSB")

# ---------- チェスボ検出 ----------
def detect_points(paths, checker, square_mm):
    objp = np.zeros((checker[0] * checker[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker[0], 0:checker[1]].T.reshape(-1, 2)
    objp *= (square_mm)  # 単位(mm)のままでも校正には問題なし（出力には square_size_mm として記録）

    objpoints, imgpoints, used_paths = [], [], []
    gray_ref = None

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_ref = gray

        ret, corners = False, None
        if use_sb:
            ret, corners = cv2.findChessboardCornersSB(gray, checker, flags=0)
            if ret:
                corners = corners.astype(np.float32)
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, checker, flags=flags_cla)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)  # (N,1,2) 形状でもOK
            used_paths.append(p)

    assert used_paths, f"チェスボード検出できず: {len(used_paths)}枚"
    h, w = gray_ref.shape[:2]
    return objpoints, imgpoints, used_paths, (w, h)

# ---------- pinhole（RATIONAL）キャリブ ----------
def calibrate_pinhole(objpoints, imgpoints, imsize):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, imsize, None, None, flags=cv2.CALIB_RATIONAL_MODEL
    )
    per_err = []
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        proj, _ = cv2.projectPoints(objpoints[i], rv, tv, K, dist)
        e = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        per_err.append(float(e))
    return rms, K, dist, per_err

def make_new_camera_matrix_pinhole(K, dist, imsize, alpha=1.0):
    w, h = imsize
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=alpha)
    return newK, roi

# ---------- fisheye キャリブ ----------
def calibrate_fisheye(objpoints, imgpoints, imsize):
    # 入力形状を fisheye API に合わせる（各ビュー: Nx1x2）
    imgpoints_fe = [np.ascontiguousarray(ip.reshape(-1,1,2), dtype=np.float64) for ip in imgpoints]
    objpoints_fe = [np.ascontiguousarray(op.reshape(-1,1,3), dtype=np.float64) for op in objpoints]

    K = np.zeros((3,3), dtype=np.float64)
    D = np.zeros((4,1), dtype=np.float64)  # fisheyeは4係数
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_CHECK_COND |
             cv2.fisheye.CALIB_FIX_SKEW)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints_fe, imgpoints_fe, imsize, K, D, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )

    # per-view error を fisheye.projectPoints で計算
    per_err = []
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        proj, _ = cv2.fisheye.projectPoints(objpoints_fe[i], rv, tv, K, D)
        # いずれも Nx1x2 なので整形
        ip = imgpoints_fe[i].reshape(-1,2)
        pj = proj.reshape(-1,2)
        e = cv2.norm(ip, pj, cv2.NORM_L2) / len(pj)
        per_err.append(float(e))

    return rms, K, D, per_err

def make_new_camera_matrix_fisheye(K, imsize):
    # fisheye には pinhole の α 概念がないので、newK=K を保存、ROIは全域
    w, h = imsize
    newK = K.copy()
    roi = (0, 0, w, h)
    return newK, roi

# ---------- メイン処理（共通トリム込み） ----------
def process_one(cfg):
    name = cfg["name"]
    GLOB = cfg["glob"]
    SAVE = cfg["save"]
    CHECKER = tuple(cfg.get("checker", (7, 6)))
    SQUARE_MM = float(cfg.get("square_mm", 25.0))
    MIN_KEEP = int(cfg.get("min_keep", 20))
    TRIM_FRAC = float(cfg.get("trim_frac", 0.12))
    MODEL = cfg.get("model", "pinhole").lower()  # "pinhole" or "fisheye"

    if MODEL not in ("pinhole", "fisheye"):
        raise ValueError(f"[{name}] model は 'pinhole' または 'fisheye' を指定してください: {MODEL}")

    print(f"\n=== [{name}] start (model={MODEL}) ===")
    os.makedirs(os.path.dirname(SAVE), exist_ok=True)

    imgs_all = sorted(glob.glob(GLOB))
    if not imgs_all:
        print(f"[WARN] [{name}] 画像が見つからないのでスキップ")
        return

    objp, imgp, paths, imsize = detect_points(imgs_all, CHECKER, SQUARE_MM)

    if MODEL == "pinhole":
        rms, K, dist, per_err = calibrate_pinhole(objp, imgp, imsize)
    else:
        rms, K, dist, per_err = calibrate_fisheye(objp, imgp, imsize)

    print(f"[pass1] {name}: N={len(paths)} RMS={rms:.3f} mean={np.mean(per_err):.3f} max={np.max(per_err):.3f}")

    keep = list(range(len(paths)))
    changed = True
    while changed and len(keep) > MIN_KEEP:
        changed = False
        idx_sorted = np.argsort(per_err)[::-1]  # 誤差大きい順
        n_drop = max(1, int(len(keep) * TRIM_FRAC))
        drop_idx = sorted(idx_sorted[:n_drop])
        new_keep = [i for i in range(len(keep)) if i not in drop_idx]

        if len(new_keep) >= MIN_KEEP and len(new_keep) < len(keep):
            objp2 = [objp[i] for i in new_keep]
            imgp2 = [imgp[i] for i in new_keep]
            paths2 = [paths[i] for i in new_keep]

            if MODEL == "pinhole":
                rms2, K2, dist2, per_err2 = calibrate_pinhole(objp2, imgp2, imsize)
            else:
                rms2, K2, dist2, per_err2 = calibrate_fisheye(objp2, imgp2, imsize)

            print(f"[trim] {name}: N={len(paths2)} RMS={rms2:.3f} mean={np.mean(per_err2):.3f} max={np.max(per_err2):.3f}")

            if rms2 <= rms:
                rms, K, dist, per_err = rms2, K2, dist2, per_err2
                objp, imgp, paths = objp2, imgp2, paths2
                keep = new_keep
                changed = True

    # newK と ROI
    if MODEL == "pinhole":
        newK, roi = make_new_camera_matrix_pinhole(K, dist, imsize, alpha=1.0)
        model_str = "opencv.calibrateCamera+RATIONAL"
    else:
        newK, roi = make_new_camera_matrix_fisheye(K, imsize)
        model_str = "cv2.fisheye.calibrate"

    data = {
        "image_width": int(imsize[0]),
        "image_height": int(imsize[1]),
        "camera_matrix": K.tolist(),
        "new_camera_matrix": newK.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "checker_inner_corners": [int(CHECKER[0]), int(CHECKER[1])],
        "square_size_mm": float(SQUARE_MM),
        "rms": float(rms),
        "per_view_error_px": [float(e) for e in per_err],
        "used_images": paths,
        "undistort_alpha": 1.0 if MODEL == "pinhole" else None,  # fisheyeは概念なし
        "roi": list(map(int, roi)) if isinstance(roi, tuple) else None,
        "model": model_str,
    }

    with open(SAVE, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[DONE] [{name}] saved: {SAVE}")

def main():
    for cfg in INPUTS:
        try:
            process_one(cfg)
        except Exception as e:
            print(f"[ERROR] {cfg['name']}:", e)

if __name__ == "__main__":
    main()
