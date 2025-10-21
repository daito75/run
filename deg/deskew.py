#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aruco_check.py
動画の全フレームで ArUco 検出し、(1) オーバーレイ動画, (2) 角度CSV, (3) 検出率サマリ を出力する。
"""
import cv2, numpy as np, math, csv, sys
from pathlib import Path
import argparse

def get_dict(name: str):
    tbl = {
        "4X4_50": cv2.aruco.DICT_4X4_50, "4X4_100": cv2.aruco.DICT_4X4_100,
        "4X4_250": cv2.aruco.DICT_4X4_250, "4X4_1000": cv2.aruco.DICT_4X4_1000,
        "5X5_50": cv2.aruco.DICT_5X5_50, "5X5_100": cv2.aruco.DICT_5X5_100,
        "5X5_250": cv2.aruco.DICT_5X5_250, "5X5_1000": cv2.aruco.DICT_5X5_1000,
        "6X6_50": cv2.aruco.DICT_6X6_50, "6X6_100": cv2.aruco.DICT_6X6_100,
        "6X6_250": cv2.aruco.DICT_6X6_250, "6X6_1000": cv2.aruco.DICT_6X6_1000,
        "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    return cv2.aruco.getPredefinedDictionary(tbl[name])

def sort_corners_ccw(corners):
    c = corners.reshape(4,2).astype(np.float32)
    center = c.mean(axis=0)
    ang = np.arctan2(c[:,1]-center[1], c[:,0]-center[0])
    return c[np.argsort(ang)]

def angle_from_corners(c):  # 上辺ベクトルの角度（水平基準）
    c = sort_corners_ccw(c)
    v = c[1] - c[0]
    a = math.degrees(math.atan2(v[1], v[0]))
    # 水平化のための回転角は -a だが、ログでは raw と fix 両方出す
    return a, -a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dict", default="4X4_50")
    ap.add_argument("--expected-id", type=int, default=None)
    args = ap.parse_args()

    inp = Path(args.__dict__["in"])
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    overlay_mp4 = outdir / "aruco_detect_overlay.mp4"
    csv_path = outdir / "angles.csv"
    txt_path = outdir / "summary.txt"

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened(): print("[ERR] cannot open:", inp); sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    aruco_dict = get_dict(args.dict)
    # 角の精度を上げる・検出安定化パラメータ
    try:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        use_new = True
    except Exception:
        detector = aruco_dict; use_new = False

    # writer（mp4→ダメならaviへ）
    writer = None; out_path = None
    for fcc, ext in [("mp4v",".mp4"),("avc1",".mp4"),("MJPG",".avi")]:
        p = overlay_mp4.with_suffix(ext)
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fcc), fps, (w,h))
        if vw.isOpened(): writer = vw; out_path = p; break
    if writer is None: print("[ERR] cannot open writer"); sys.exit(1)

    # CSV
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    cw = csv.writer(fcsv)
    cw.writerow(["frame","detected","used_id","raw_angle_deg","fix_angle_deg","cx","cy","area_px2"])

    det_count = 0
    frame_idx = 0
    while True:
        ret, fr = cap.read()
        if not ret: break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

        if use_new:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, detector)

        used = 0; raw = ""; fix = ""; cx = ""; cy = ""; area = ""
        if ids is not None and len(ids)>0:
            ids = ids.flatten()
            # expected-id があれば優先、なければ最大面積
            idx = None
            if args.expected_id is not None:
                hit = np.where(ids==args.expected_id)[0]
                if len(hit)>0: idx = int(hit[0])
            if idx is None:
                areas = [cv2.contourArea(c.reshape(-1,2).astype(np.float32)) for c in corners]
                idx = int(np.argmax(areas))

            c = corners[idx].reshape(4,2).astype(np.float32)
            a_raw, a_fix = angle_from_corners(c)
            m = c.mean(axis=0)
            ar = cv2.contourArea(c.reshape(-1,1,2))
            # 描画
            cv2.aruco.drawDetectedMarkers(fr, corners, ids)
            cv2.putText(fr, f"id={ids[idx]} angle={a_raw:+.2f} fix={a_fix:+.2f}",
                        (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3, cv2.LINE_AA)
            det_count += 1; used = 1
            raw = f"{a_raw:.6f}"; fix = f"{a_fix:.6f}"
            cx = f"{m[0]:.2f}"; cy = f"{m[1]:.2f}"; area = f"{ar:.1f}"
        else:
            cv2.putText(fr, "NO MARKER", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255), 3, cv2.LINE_AA)

        cw.writerow([frame_idx, used, (ids[idx] if (ids is not None and used==1) else ""), raw, fix, cx, cy, area])
        writer.write(fr)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"[INFO] {frame_idx} frames... detected {det_count}")

    cap.release(); writer.release(); fcsv.close()

    rate = (det_count / frame_idx) if frame_idx>0 else 0.0
    txt = (f"frames={frame_idx}\n"
           f"detected_frames={det_count}\n"
           f"detection_rate={rate*100:.2f}%\n"
           f"overlay_video={out_path}\n"
           f"angles_csv={csv_path}\n")
    txt_path.write_text(txt, encoding="utf-8")
    print("[DONE]\n" + txt)

if __name__ == "__main__":
    main()
