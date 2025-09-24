# scan_useful_images_batch.py
import cv2, glob, os, shutil, argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim

def parse_cam_ids(s: str):
    ids = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            ids += list(range(int(a), int(b)+1))
        else:
            ids.append(int(part))
    return sorted(set(ids))

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def coverage_score(corners, w, h):
    xy = corners.reshape(-1, 2)
    xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
    area = max(0.0, (xmax - xmin) * (ymax - ymin)) / (w * h + 1e-6)
    margin = min(xmin, ymin, w - xmax, h - ymax)
    reach = 1.0 - max(0.0, margin) / (0.5 * min(w, h) + 1e-6)
    return 0.7 * area + 0.3 * reach

def ensure_dir(p): 
    if p: os.makedirs(p, exist_ok=True)

def process_one(name, GLOB, CHECKER, SAVE_DIR, DRAW_DIR, MAX_DRAW=50, USE_SB_EX=False):
    print(f"\n=== [{name}] start ===")
    imgs = sorted(glob.glob(GLOB))
    if not imgs:
        print(f"[WARN] [{name}] 画像なし: {GLOB}")
        return {"name": name, "total": 0, "good": 0, "bad": 0, "picked_dir": SAVE_DIR, "draw_dir": DRAW_DIR,
                "good_list": None, "bad_list": None}

    ensure_dir(SAVE_DIR); ensure_dir(DRAW_DIR)
    use_sb = hasattr(cv2, "findChessboardCornersSB")
    flags_cla = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    flags_sb = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY) if (USE_SB_EX and use_sb) else 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

    good, bad = [], []; draw_count = 0
    for p in imgs:
        img = cv2.imread(p); 
        if img is None: bad.append(p); continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = False, None
        if use_sb:
            ret, corners = cv2.findChessboardCornersSB(gray, CHECKER, flags=flags_sb)
            if ret: corners = corners.astype(np.float32)
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, CHECKER, flags=flags_cla)
            if ret: cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        if ret:
            blur = variance_of_laplacian(gray)
            if blur < 80: bad.append(p); continue
            score = coverage_score(corners, gray.shape[1], gray.shape[0])
            good.append((p, score, blur))
            if DRAW_DIR and draw_count < MAX_DRAW:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, CHECKER, corners, True)
                cv2.imwrite(os.path.join(DRAW_DIR, os.path.basename(p)), vis)
                draw_count += 1
        else:
            bad.append(p)

    good.sort(key=lambda x: (-x[1], -x[2]))
    good_paths = [g[0] for g in good]

    # 重複間引きコピー
    saved = 0; thumb_cache = None
    for pth in good_paths:
        img = cv2.imread(pth)
        thg = cv2.cvtColor(cv2.resize(img, (320, 180)), cv2.COLOR_BGR2GRAY)
        if thumb_cache is None or ssim(thumb_cache, thg) < 0.97:
            shutil.copy2(pth, os.path.join(SAVE_DIR, os.path.basename(pth)))
            thumb_cache = thg; saved += 1
    print(f"[{name}] 検出成功: {len(good_paths)} / {len(imgs)}  重複間引き後コピー: {saved} -> {SAVE_DIR}")

    # 一覧
    root = os.path.dirname(SAVE_DIR)
    good_list = os.path.join(root, f"good_{CHECKER[0]}x{CHECKER[1]}_{name}.txt")
    bad_list  = os.path.join(root, f"bad_{CHECKER[0]}x{CHECKER[1]}_{name}.txt")
    with open(good_list, "w", encoding="utf-8") as f: f.write("\n".join(g[0] for g in good))
    with open(bad_list, "w", encoding="utf-8") as f: f.write("\n".join(bad))

    return {"name": name, "total": len(imgs), "good": len(good), "bad": len(bad),
            "picked_dir": SAVE_DIR, "draw_dir": DRAW_DIR, "good_list": good_list, "bad_list": bad_list}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaiseki", type=int, required=True)
    ap.add_argument("--cam-ids", type=str, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--checker", default="7x6")
    ap.add_argument("--use-sb-exhaustive", action="store_true")
    args, _ = ap.parse_known_args()

    cols, rows = map(int, args.checker.lower().split("x"))
    CHECKER = (cols, rows)
    k = args.kaiseki
    cam_ids = parse_cam_ids(args.cam_ids)
    root = args.root.rstrip("\\/")

    summary = []
    for i in cam_ids:
        name = f"cam{i}"
        GLOB = rf"{root}\kaiseki{k}\frame\{name}\*.png"
        SAVE = rf"{root}\kaiseki{k}\frame\{name}\picked"
        DRAW = rf"{root}\kaiseki{k}\frame\{name}\debug_draw"
        info = process_one(name, GLOB, CHECKER, SAVE, DRAW, MAX_DRAW=50, USE_SB_EX=args.use_sb_exhaustive)
        summary.append(info)

    print("\n========== SUMMARY ==========")
    for s in summary:
        print(f"{s['name']}: good {s['good']} / total {s['total']} (bad {s['bad']})")
        print(f" picked -> {s['picked_dir']}")
        print(f" debug  -> {s['draw_dir']}")
        print(f" good list: {s['good_list']}")
        print(f" bad  list: {s['bad_list']}")

if __name__ == "__main__":
    main()
