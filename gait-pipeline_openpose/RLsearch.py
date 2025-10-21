#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPose BODY_25 の --write_json 出力を使って、動画に L/R ラベルを重畳するツール。
- 左：青 (BGR=(255,0,0))、右：緑 (BGR=(0,200,0))
- 膝(10=R,13=L)・足首(11=R,14=L)の近くに表示
- people が居ない/信頼度が低いフレームはスキップ

mkdir -p /mnt/d/BRLAB/2025/openpose_out/0LR/1/RLsearch

python3 RLsearch.py \
  --json_dir /mnt/d/BRLAB/2025/openpose_out/0LR/1/json \
  --video_in /mnt/d/BRLAB/2025/mizuno/done/deta/kaiseki2/concat/a0/0LR/1.mp4 \
  --out /mnt/d/BRLAB/2025/openpose_out/0LR/1/RLsearch/labeled_lr.mp4 \
  --conf_th 0.2 \
  --draw_knee


"""

import os, glob, json, argparse
import numpy as np
import cv2
from pathlib import Path
import subprocess, sys

class FFmpegWriter:
    def __init__(self, out_path, w, h, fps):
        # libx264 で yuv420p（ほぼどのプレイヤでも再生可）
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps}",
            "-i", "-",  # 標準入力から受け取る
            "-an",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            str(out_path)
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        # frameはH×W×3のBGR uint8
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        if self.proc.stdin:
            self.proc.stdin.flush()
            self.proc.stdin.close()
        self.proc.wait()

def open_video_writer_with_fallback(out_path, W, H, fps):
    # 1) OpenCVで試す（mp4 / avi）
    trials = [("mp4v",".mp4"), ("XVID",".avi"), ("MJPG",".avi")]
    for fourcc, ext in trials:
        p = Path(out_path).with_suffix(ext)
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), fps, (W,H))
        if vw.isOpened():
            return p, vw  # OpenCVのVideoWriterを返す
    # 2) 失敗したら ffmpeg パイプにフォールバック（mp4出力）
    p = Path(out_path).with_suffix(".mp4")
    fw = FFmpegWriter(p, W, H, fps)
    # OpenCV互換のインターフェースに合わせるため、同名メソッドを持つラッパクラスを返す
    return p, fw


# BODY_25 indices
RKNEE, RANK = 10, 11
LKNEE, LANK = 13, 14

def load_json_paths(json_dir):
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not files:
        raise SystemExit(f"[ERR] no json files in: {json_dir}")
    return files

def pick_person(J):
    """最も総信頼度が高い person を1人選ぶ（居なければ None）"""
    ppl = J.get("people", [])
    if not ppl:
        return None
    def score(p):
        k = np.array(p["pose_keypoints_2d"], dtype=float).reshape(-1,3)[:,2]
        return float(np.nansum(k))
    return max(ppl, key=score)

def draw_label(frame, text, xy, color):
    if not np.all(np.isfinite(xy)): return
    x, y = int(xy[0]), int(xy[1])
    cv2.circle(frame, (x, y), 6, color, -1)
    cv2.putText(frame, text, (x+8, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="OpenPose --write_json のフォルダ")
    ap.add_argument("--video_in", required=True, help="元動画 (OpenPose にかけたもの)")
    ap.add_argument("--out", required=True, help="出力動画パス（拡張子は自動で .mp4/.avi を試す）")
    ap.add_argument("--conf_th", type=float, default=0.2, help="ラベル描画に必要な最小信頼度")
    ap.add_argument("--draw_knee", action="store_true", help="足首に加えて膝にも L/R を描く")
    args = ap.parse_args()

    json_files = load_json_paths(args.json_dir)

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise SystemExit(f"[ERR] cannot open video: {args.video_in}")
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nV  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(json_files)

    # 出力準備（mp4→失敗ならavi）
    out_path, writer = open_video_writer_with_fallback(args.out, W, H, fps)

    nF = min(nV, len(json_files))
    i = 0
    while i < nF:
        ok, frame = cap.read()
        if not ok: break

        # 対応するJSONを読む
        with open(json_files[i], "r") as f:
            J = json.load(f)
        person = pick_person(J)
        if person is not None:
            k = np.array(person["pose_keypoints_2d"], dtype=float).reshape(-1,3)
            # 右足（緑）
            if k[RANK,2] >= args.conf_th:
                draw_label(frame, "R", k[RANK,:2], (0,200,0))
            if args.draw_knee and k[RKNEE,2] >= args.conf_th:
                draw_label(frame, "R", k[RKNEE,:2], (0,200,0))
            # 左足（青）
            if k[LANK,2] >= args.conf_th:
                draw_label(frame, "L", k[LANK,:2], (255,0,0))
            if args.draw_knee and k[LKNEE,2] >= args.conf_th:
                draw_label(frame, "L", k[LKNEE,:2], (255,0,0))

        writer.write(frame)

        i += 1
        if i % 200 == 0:
            print(f"[INFO] {i}/{nF} frames")

    cap.release()
    writer.release()
    print(f"[DONE] wrote: {out_path}")

if __name__ == "__main__":
    main()
