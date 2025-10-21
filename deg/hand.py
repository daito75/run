#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deskew_fixed_angle.py
固定角で動画を回転。--angle-deg で度数指定。--expand で切れ防止のキャンバス拡張。
"""
import cv2, math, argparse
from pathlib import Path

def rotated_bounds(w, h, angle_deg):
    a = math.radians(abs(angle_deg))
    new_w = int(w*abs(math.cos(a)) + h*abs(math.sin(a)))
    new_h = int(w*abs(math.sin(a)) + h*abs(math.cos(a)))
    return new_w, new_h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  required=True, help="入力動画")
    ap.add_argument("--out", required=True, help="出力動画")
    ap.add_argument("--angle-deg", type=float, required=True, help="回転角[deg]（反時計回りが正）")
    ap.add_argument("--expand", action="store_true", help="回転で切れないようキャンバス拡張")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.__dict__["in"])
    if not cap.isOpened():
        raise SystemExit(f"cannot open: {args.__dict__['in']}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_size = (w, h)
    if args.expand:
        out_size = rotated_bounds(w, h, args.angle_deg)

    writer = None; out_path = None
    for fourcc, ext in [("mp4v",".mp4"), ("avc1",".mp4"), ("MJPG",".avi")]:
        p = Path(args.__dict__["out"]).with_suffix(ext)
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), fps, out_size)
        if vw.isOpened():
            writer = vw; out_path = p; break
    if writer is None:
        raise SystemExit("cannot open VideoWriter")

    cx, cy = w/2, h/2
    while True:
        ret, frame = cap.read()
        if not ret: break
        M = cv2.getRotationMatrix2D((cx, cy), args.angle_deg, 1.0)
        if args.expand:
            # 中央合わせの平行移動（拡張キャンバスへ配置）
            a = math.radians(args.angle_deg)
            bw = int(h*abs(math.sin(a)) + w*abs(math.cos(a)))
            bh = int(h*abs(math.cos(a)) + w*abs(math.sin(a)))
            tx = (out_size[0] - bw)//2
            ty = (out_size[1] - bh)//2
            M[0,2] += tx
            M[1,2] += ty
            rot = cv2.warpAffine(frame, M, out_size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        else:
            rot = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        writer.write(rot)

    cap.release(); writer.release()
    print("saved:", out_path)

if __name__ == "__main__":
    main()
