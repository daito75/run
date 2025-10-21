#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

棒人間動画にフレームレートを記載する
python3 annotate_frames.py \
  --in_mp4 "/mnt/d/BRLAB/2025/mizuno/done/deta/kaiseki4/openpose/3.0m/60cm/mywalk_pose_overlay.mp4" \
  --out_mp4 "/mnt/d/BRLAB/2025/mizuno/done/deta/kaiseki4/openpose/3.0m/60cm/mywalk_pose_overlay_annot.mp4" \
  --font_scale 1.0 --thickness 2 --x 16 --y 40 --show_time
"""

import cv2, argparse, math, os

def main():
    ap = argparse.ArgumentParser(description="動画にフレーム番号/時間を焼き込む")
    ap.add_argument("--in_mp4", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--font_scale", type=float, default=1.0)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--x", type=int, default=16, help="左上テキストX座標")
    ap.add_argument("--y", type=int, default=40, help="左上テキストY座標（1行目）")
    ap.add_argument("--line_gap", type=int, default=34, help="行間ピクセル")
    ap.add_argument("--start_index", type=int, default=0, help="フレームの起算（0 or 1）")
    ap.add_argument("--show_time", action="store_true", help="秒数も表示する")
    ap.add_argument("--fps_override", type=float, default=None, help="VFR対策などでfpsを指定したい場合")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.in_mp4)
    assert cap.isOpened(), f"[ERR] open fail: {args.in_mp4}"

    # 入力情報
    in_fps = args.fps_override or (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 出力設定（入力に合わせる）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 動かなければ 'avc1' などに変更
    os.makedirs(os.path.dirname(args.out_mp4), exist_ok=True)
    out = cv2.VideoWriter(args.out_mp4, fourcc, in_fps, (w, h))
    assert out.isOpened(), f"[ERR] writer open fail: {args.out_mp4}"

    # 描画スタイル
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = args.font_scale
    th   = args.thickness
    color_fg = (255,255,255)  # 白
    color_bg = (0,0,0)        # 黒（縁取り用）
    x0, y0 = args.x, args.y
    dy = args.line_gap

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # 0/1始まりを選べるように
        disp_idx = i + args.start_index
        text1 = f"frame {disp_idx}/{total_frames - 1 + args.start_index}"

        # 秒表示（fpsを使って計算）
        if args.show_time:
            t = i / in_fps
            text2 = f"t = {t:6.2f} s"
        else:
            text2 = None

        # 文字が見やすいように黒縁→白字の二度描き
        cv2.putText(frame, text1, (x0, y0), font, fs, color_bg, th+2, cv2.LINE_AA)
        cv2.putText(frame, text1, (x0, y0), font, fs, color_fg, th,   cv2.LINE_AA)

        if text2:
            cv2.putText(frame, text2, (x0, y0+dy), font, fs, color_bg, th+2, cv2.LINE_AA)
            cv2.putText(frame, text2, (x0, y0+dy), font, fs, color_fg, th,   cv2.LINE_AA)

        out.write(frame)
        i += 1

    cap.release(); out.release()
    print(f"[DONE] {args.out_mp4}")

if __name__ == "__main__":
    main()
