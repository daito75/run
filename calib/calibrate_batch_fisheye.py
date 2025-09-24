import os, glob, yaml, numpy as np, cv2, argparse

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

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
flags_cla = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
use_sb = hasattr(cv2, "findChessboardCornersSB")

def detect_points(paths, checker, square_mm):
    objp = np.zeros((checker[0]*checker[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker[0], 0:checker[1]].T.reshape(-1,2)
    objp *= square_mm
    objpoints, imgpoints, used, gray_ref = [], [], [], None
    for p in paths:
        img = cv2.imread(p); 
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); gray_ref = gray
        ret, corners = False, None
        if use_sb:
            ret, corners = cv2.findChessboardCornersSB(gray, checker, flags=0)
            if ret: corners = corners.astype(np.float32)
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, checker, flags=flags_cla)
            if ret: cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        if ret:
            objpoints.append(objp); imgpoints.append(corners); used.append(p)
    assert used, "チェスボ検出ゼロ"
    h, w = gray_ref.shape[:2]
    return objpoints, imgpoints, used, (w, h)

def calibrate_pinhole(objp, imgp, imsize):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, imsize, None, None, flags=cv2.CALIB_RATIONAL_MODEL)
    per_err=[]
    for i,(rv,tv) in enumerate(zip(rvecs,tvecs)):
        proj,_ = cv2.projectPoints(objp[i], rv, tv, K, dist)
        e = cv2.norm(imgp[i], proj, cv2.NORM_L2) / len(proj); per_err.append(float(e))
    return rms, K, dist, per_err

def calibrate_fisheye(objp, imgp, imsize):
    imgp_fe = [np.ascontiguousarray(ip.reshape(-1,1,2), np.float64) for ip in imgp]
    objp_fe = [np.ascontiguousarray(op.reshape(-1,1,3), np.float64) for op in objp]
    K = np.zeros((3,3), np.float64); D = np.zeros((4,1), np.float64)
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objp_fe, imgp_fe, imsize, K, D, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    per_err=[]
    for i,(rv,tv) in enumerate(zip(rvecs,tvecs)):
        proj,_ = cv2.fisheye.projectPoints(objp_fe[i], rv, tv, K, D)
        ip = imgp_fe[i].reshape(-1,2); pj = proj.reshape(-1,2)
        e = cv2.norm(ip, pj, cv2.NORM_L2) / len(pj); per_err.append(float(e))
    return rms, K, D, per_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaiseki", type=int, required=True)
    ap.add_argument("--cam-ids", type=str, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--checker", default="7x6")
    ap.add_argument("--square-mm", type=float, default=30.0)
    ap.add_argument("--model", choices=["fisheye","pinhole"], default="fisheye")
    ap.add_argument("--min-keep", type=int, default=20)
    ap.add_argument("--trim-frac", type=float, default=0.12)
    args, _ = ap.parse_known_args()

    cols, rows = map(int, args.checker.lower().split("x"))
    CHECKER = (cols, rows)
    k = args.kaiseki
    cam_ids = parse_cam_ids(args.cam_ids)
    root = args.root.rstrip("\\/")

    for i in cam_ids:
        name = f"cam{i}"
        GLOB = rf"{root}\kaiseki{k}\frame\{name}\picked\*.png"
        SAVE = rf"{root}\kaiseki{k}\frame\{name}\{name}_1080p.yaml"
        os.makedirs(os.path.dirname(SAVE), exist_ok=True)

        imgs_all = sorted(glob.glob(GLOB))
        if not imgs_all:
            print(f"[WARN] [{name}] 画像無し: {GLOB}"); continue

        objp, imgp, paths, imsize = detect_points(imgs_all, CHECKER, args.square_mm)

        if args.model == "pinhole":
            rms, K, dist, per_err = calibrate_pinhole(objp, imgp, imsize)
        else:
            rms, K, dist, per_err = calibrate_fisheye(objp, imgp, imsize)

        print(f"[pass1] {name}: N={len(paths)} RMS={rms:.3f} mean={np.mean(per_err):.3f}")

        # ざっくりトリム
        keep_idx = list(range(len(paths)))
        changed = True
        while changed and len(keep_idx) > args.min_keep:
            changed = False
            idx_sorted = np.argsort(per_err)[::-1]
            n_drop = max(1, int(len(keep_idx)*args.trim_frac))
            drop = set(idx_sorted[:n_drop])
            new_keep = [i for i in range(len(keep_idx)) if i not in drop]
            if len(new_keep) >= args.min_keep and len(new_keep) < len(keep_idx):
                objp2 = [objp[i] for i in new_keep]; imgp2 = [imgp[i] for i in new_keep]
                if args.model == "pinhole":
                    rms2, K2, dist2, per2 = calibrate_pinhole(objp2, imgp2, imsize)
                else:
                    rms2, K2, dist2, per2 = calibrate_fisheye(objp2, imgp2, imsize)
                if rms2 <= rms:
                    rms, K, dist, per_err = rms2, K2, dist2, per2
                    objp, imgp, keep_idx = objp2, imgp2, new_keep
                    changed = True

        # newK / roi 記録
        if args.model == "pinhole":
            w, h = imsize
            newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=1.0)
            model_str = "opencv.calibrateCamera+RATIONAL"
        else:
            newK, roi = K.copy(), (0,0,imsize[0], imsize[1])
            model_str = "cv2.fisheye.calibrate"

        data = {
            "image_width": int(imsize[0]),
            "image_height": int(imsize[1]),
            "camera_matrix": K.tolist(),
            "new_camera_matrix": newK.tolist(),
            "dist_coeffs": np.asarray(dist).reshape(-1).tolist(),
            "checker_inner_corners": [int(CHECKER[0]), int(CHECKER[1])],
            "square_size_mm": float(args.square_mm),
            "rms": float(rms),
            "per_view_error_px": [float(e) for e in per_err],
            "used_images": paths,
            "undistort_alpha": 1.0 if args.model == "pinhole" else None,
            "roi": list(map(int, roi)) if isinstance(roi, tuple) else None,
            "model": model_str,
        }
        with open(SAVE, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"[DONE] [{name}] saved: {SAVE}")

if __name__ == "__main__":
    main()

