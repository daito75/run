#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibrate_batch.py
複数カメラの picked フォルダから一気にキャリブレーションして YAML 出力。

機能:
  - チェスボ検出 (SB優先→従来法フォールバック)
  - 初回キャリブ (RATIONAL_MODEL)
  - 1回目の誤差分布から高誤差ビューを一定率トリム（RMSが改善した場合のみ採用）
  - new_camera_matrix と ROI を計算し YAML へ出力
  - 使用画像一覧・各ビューの誤差も YAML に保存

使い方:
  python calibrate_batch.py
"""

# calibrate_batch.py
# 複数カメラの picked フォルダから一気にキャリブレーションして yaml 出力

import os, glob, yaml
import numpy as np
import cv2

# ========= ここに各カメラごとの設定を並べる =========
INPUTS = [
    {
        "name": "cam1",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\picked\*.png",
        "save": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\cam1_1080p.yaml",
        "checker": (7, 6),
        "square_mm": 30.0,
        "min_keep": 20,
        "trim_frac": 0.12,
    },
    {
        "name": "cam2",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam2\picked\*.png",
        "save": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam2\cam2_1080p.yaml",
        "checker": (7, 6),
        "square_mm": 30.0,
        "min_keep": 20,
        "trim_frac": 0.12,
    },
    {
        "name": "cam3",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam3\picked\*.png",
        "save": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam3\cam3_1080p.yaml",
        "checker": (7, 6),
        "square_mm": 30.0,
        "min_keep": 20,
        "trim_frac": 0.12,
    },
    {
        "name": "cam4",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam4\picked\*.png",
        "save": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam4\cam4_1080p.yaml",
        "checker": (7, 6),
        "square_mm": 30.0,
        "min_keep": 20,
        "trim_frac": 0.12,
    },
    {
        "name": "cam5",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam5\picked\*.png",
        "save": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam5\cam5_1080p.yaml",
        "checker": (7, 6),
        "square_mm": 30.0,
        "min_keep": 20,
        "trim_frac": 0.12,
    },
    # 必要に応じて追加
]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
flags_cla = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
use_sb = hasattr(cv2, "findChessboardCornersSB")


def detect_points(paths, checker, square_mm):
    objp = np.zeros((checker[0] * checker[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker[0], 0:checker[1]].T.reshape(-1, 2)
    objp *= square_mm

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
            imgpoints.append(corners)
            used_paths.append(p)

    assert used_paths, f"チェスボード検出できず: {len(used_paths)}枚"
    h, w = gray_ref.shape[:2]
    return objpoints, imgpoints, used_paths, (w, h)


def calibrate(objpoints, imgpoints, imsize):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, imsize, None, None, flags=cv2.CALIB_RATIONAL_MODEL
    )
    per_err = []
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        proj, _ = cv2.projectPoints(objpoints[i], rv, tv, K, dist)
        e = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        per_err.append(float(e))
    return rms, K, dist, per_err


def make_new_camera_matrix(K, dist, imsize, alpha=1.0):
    w, h = imsize
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=alpha)
    return newK, roi


def process_one(cfg):
    name = cfg["name"]
    GLOB = cfg["glob"]
    SAVE = cfg["save"]
    CHECKER = tuple(cfg.get("checker", (7, 6)))
    SQUARE_MM = float(cfg.get("square_mm", 25.0))
    MIN_KEEP = int(cfg.get("min_keep", 20))
    TRIM_FRAC = float(cfg.get("trim_frac", 0.12))

    print(f"\n=== [{name}] start ===")
    os.makedirs(os.path.dirname(SAVE), exist_ok=True)

    imgs_all = sorted(glob.glob(GLOB))
    if not imgs_all:
        print(f"[WARN] [{name}] 画像が見つからないのでスキップ")
        return

    objp, imgp, paths, imsize = detect_points(imgs_all, CHECKER, SQUARE_MM)
    rms, K, dist, per_err = calibrate(objp, imgp, imsize)
    print(f"[pass1] {name}: N={len(paths)} RMS={rms:.3f} mean={np.mean(per_err):.3f} max={np.max(per_err):.3f}")

    keep = list(range(len(paths)))
    changed = True
    while changed and len(keep) > MIN_KEEP:
        changed = False
        idx_sorted = np.argsort(per_err)[::-1]
        n_drop = max(1, int(len(keep) * TRIM_FRAC))
        drop_idx = sorted(idx_sorted[:n_drop])
        new_keep = [i for i in range(len(keep)) if i not in drop_idx]

        if len(new_keep) >= MIN_KEEP and len(new_keep) < len(keep):
            objp2 = [objp[i] for i in new_keep]
            imgp2 = [imgp[i] for i in new_keep]
            paths2 = [paths[i] for i in new_keep]

            rms2, K2, dist2, per_err2 = calibrate(objp2, imgp2, imsize)
            print(f"[trim] {name}: N={len(paths2)} RMS={rms2:.3f} mean={np.mean(per_err2):.3f} max={np.max(per_err2):.3f}")

            if rms2 <= rms:
                rms, K, dist, per_err = rms2, K2, dist2, per_err2
                objp, imgp, paths = objp2, imgp2, paths2
                keep = new_keep
                changed = True

    newK, roi = make_new_camera_matrix(K, dist, imsize, alpha=1.0)

    data = {
        "image_width": int(imsize[0]),
        "image_height": int(imsize[1]),
        "camera_matrix": K.tolist(),
        "new_camera_matrix": newK.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "checker_inner_corners": [int(CHECKER[0]), int(CHECKER[1])],
        "square_size_mm": float(SQUARE_MM),
        "rms": float(rms),
        "per_view_error_px": per_err,
        "used_images": paths,
        "undistort_alpha": 1.0,
        "roi": list(map(int, roi)) if isinstance(roi, tuple) else None,
        "model": "opencv.calibrateCamera+RATIONAL",
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
