#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_useful_images_batch.py
複数カメラ分のチェスボード静止画を走査し、良否判定・可視化・重複間引きコピー・一覧出力を一括実行する。

主な判定基準:
  - コーナ検出に成功
  - 画像の鋭さ（Laplacian 分散）閾値以上
  - チェスボードの画面被覆率 + 端への到達度（coverage_score）

メモ:
  - OpenCVの findChessboardCornersSB が使える環境ではそれを優先（use_sb_exhaustive=Trueで広域探索）。
  - skimage の SSIM が無ければ、単純な差分判定にフォールバック。
"""

# scan_useful_images_batch.py
# 複数カメラのチェスボ良否判定を一括処理
import cv2, glob, os, shutil
import numpy as np
from skimage.metrics import structural_similarity as ssim


def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def coverage_score(corners, w, h):
    # 検出コーナの外接矩形の面積率 + 端への到達度
    xy = corners.reshape(-1, 2)
    xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
    area = max(0.0, (xmax - xmin) * (ymax - ymin)) / (w * h + 1e-6)
    margin = min(xmin, ymin, w - xmax, h - ymax)
    reach = 1.0 - max(0.0, margin) / (0.5 * min(w, h) + 1e-6)
    return 0.7 * area + 0.3 * reach


# ========= ここに各カメラの設定を並べる =========
INPUTS = [
    {
        "name": "cam1",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\*.png",
        "checker": (7, 6),  # 内側交点
        "save_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\picked",
        "draw_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\debug_draw",
        "max_draw": 50,
        "use_sb_exhaustive": False,
    },
    {
        "name": "cam2",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam2\*.png",
        "checker": (7, 6),
        "save_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam2\picked",
        "draw_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam2\debug_draw",
        "max_draw": 50,
        "use_sb_exhaustive": False,
    },
    {
        "name": "cam3",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam3\*.png",
        "checker": (7, 6),
        "save_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam3\picked",
        "draw_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam3\debug_draw",
        "max_draw": 50,
        "use_sb_exhaustive": False,
    },
    {
        "name": "cam4",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam4\*.png",
        "checker": (7, 6),
        "save_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam4\picked",
        "draw_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam4\debug_draw",
        "max_draw": 50,
        "use_sb_exhaustive": False,
    },
    {
        "name": "cam5",
        "glob": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam5\*.png",
        "checker": (7, 6),
        "save_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam5\picked",
        "draw_dir": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam5\debug_draw",
        "max_draw": 50,
        "use_sb_exhaustive": False,
    },
    # 追加したいだけ増やしてOK
]


def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)


def process_one(cfg):
    name = cfg["name"]
    GLOB = cfg["glob"]
    CHECKER = tuple(cfg.get("checker", (7, 6)))
    SAVE_DIR = cfg.get("save_dir")
    DRAW_DIR = cfg.get("draw_dir")
    MAX_DRAW = int(cfg.get("max_draw", 50))
    USE_SB_EX = bool(cfg.get("use_sb_exhaustive", False))

    print(f"\n=== [{name}] start ===")
    imgs = sorted(glob.glob(GLOB))
    if not imgs:
        print(f"[WARN] [{name}] 画像が見つからないのでスキップ: {GLOB}")
        return {
            "name": name, "total": 0, "good": 0, "bad": 0,
            "picked_dir": SAVE_DIR, "draw_dir": DRAW_DIR,
            "good_list": None, "bad_list": None
        }

    ensure_dir(SAVE_DIR)
    ensure_dir(DRAW_DIR)

    use_sb = hasattr(cv2, "findChessboardCornersSB")
    flags_cla = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    flags_sb = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY) if (USE_SB_EX and use_sb) else 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

    good, bad = [], []
    draw_count = 0

    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            print(f"[{name}] 読めない: {p}")
            bad.append(p)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = False, None

        # 1) SB優先（使える環境のみ）
        if use_sb:
            ret, corners = cv2.findChessboardCornersSB(gray, CHECKER, flags=flags_sb)
            if ret:
                corners = corners.astype(np.float32)

        # 2) 従来法フォールバック
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, CHECKER, flags=flags_cla)
            if ret:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if ret:
            blur = variance_of_laplacian(gray)
            if blur < 80:  # しきい値調整OK
                bad.append(p)
                continue

            score = coverage_score(corners, gray.shape[1], gray.shape[0])
            good.append((p, score, blur))

            # 可視化保存
            if DRAW_DIR and draw_count < MAX_DRAW:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, CHECKER, corners, True)
                outp = os.path.join(DRAW_DIR, os.path.basename(p))
                cv2.imwrite(outp, vis)
                draw_count += 1

            # 良品コピー
            if SAVE_DIR:
                shutil.copy2(p, os.path.join(SAVE_DIR, os.path.basename(p)))
        else:
            bad.append(p)

    # 結果出力
    good.sort(key=lambda x: (-x[1], -x[2]))  # coverage→blur順にソート
    good_paths = [g[0] for g in good]

    print(f"[{name}] 検出成功: {len(good_paths)} / {len(imgs)} 枚")
    if DRAW_DIR:
        print(f"[{name}] 可視化保存: {draw_count} 枚 → {DRAW_DIR}")
    if SAVE_DIR:
        thumb_cache = None; saved = 0
        for pth in good_paths:
            img = cv2.imread(pth)
            th = cv2.resize(img, (320, 180))
            thg = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)
            if thumb_cache is None:
                ok_dup = True
            else:
                score_ssim = ssim(thumb_cache, thg)
                ok_dup = (score_ssim < 0.97)
            if ok_dup:
                shutil.copy2(pth, os.path.join(SAVE_DIR, os.path.basename(pth)))
                thumb_cache = thg
                saved += 1
        print(f"[{name}] 重複間引き後の良品コピー: {saved} 枚 → {SAVE_DIR}")

    # 一覧をテキストで保存
    root = os.path.dirname(SAVE_DIR) if SAVE_DIR else os.path.dirname(os.path.dirname(GLOB))
    ensure_dir(root)
    good_list = os.path.join(root, f"good_{CHECKER[0]}x{CHECKER[1]}_{name}.txt")
    bad_list = os.path.join(root, f"bad_{CHECKER[0]}x{CHECKER[1]}_{name}.txt")
    with open(good_list, "w", encoding="utf-8") as f:
        f.write("\n".join(good))
    with open(bad_list, "w", encoding="utf-8") as f:
        f.write("\n".join(bad))

    # 参考表示（先頭5件）
    print(f"\n[{name}] === GOOD (先頭5件) ===")
    for x in good[:5]:
        print(x)
    print(f"\n[{name}] === BAD (先頭5件) ===")
    for x in bad[:5]:
        print(x)

    return {
        "name": name,
        "total": len(imgs),
        "good": len(good),
        "bad": len(bad),
        "picked_dir": SAVE_DIR,
        "draw_dir": DRAW_DIR,
        "good_list": good_list,
        "bad_list": bad_list,
    }


def main():
    summary = []
    for cfg in INPUTS:
        try:
            info = process_one(cfg)
            summary.append(info)
        except AssertionError as e:
            print("[ERROR]", e)
        except Exception as e:
            print("[ERROR] unexpected:", e)

    # 総括
    print("\n========== SUMMARY ==========")
    for s in summary:
        print(f"{s['name']}: good {s['good']} / total {s['total']} (bad {s['bad']})")
        print(f" good list: {s['good_list']}")
        print(f" bad list: {s['bad_list']}")
        if s['picked_dir']:
            print(f" picked -> {s['picked_dir']}")
        if s['draw_dir']:
            print(f" debug -> {s['draw_dir']}")


if __name__ == "__main__":
    main()
