# crop_concat.py
import cv2, json, os, argparse
import numpy as np
from pathlib import Path

def parse_cam_ids(s: str):
    ids = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            ids.extend(range(int(a), int(b)+1))
        else:
            ids.append(int(part))
    out = []
    for x in ids:
        if x not in out:
            out.append(x)
    return out

def parse_offset_map(s: str):
    # "1:0,3:0.5" -> {"cam1":0.0, "cam3":0.5}
    out = {}
    if not s: return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        k,v = tok.split(":",1)
        out[f"cam{int(k)}"] = float(v)
    return out

def open_writer_with_fallback(path_mp4, fps, size_wh):
    fourccs = [("mp4v",".mp4"), ("avc1",".mp4"), ("H264",".mp4"), ("XVID",".avi"), ("MJPG",".avi")]
    base, ext0 = os.path.splitext(str(path_mp4))
    for four, ext in fourccs:
        outp = base+ext
        w = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*four), float(fps), size_wh)
        if w.isOpened():
            return w, outp, four
    raise RuntimeError("VideoWriterを開けませんでした")

def main():
    ap = argparse.ArgumentParser(description="ROIクロップ→横並び結合（α廃止・入出力柔軟化）")
    ap.add_argument("--kaiseki", type=int, required=True)
    ap.add_argument("--cam-ids", type=str, required=True, help='例 "1-3,5"（順序=並び順）')
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--target-height", type=int, default=480)
    ap.add_argument("--offset-sec", default="", help='例 "1:0,3:0.5"')

    # ★ 新規：出力場所
    ap.add_argument("--out-dir", default="", help="出力フォルダ（未指定なら既定パス）")
        # ★ 新規：入力マップ（受け取るだけ）
    ap.add_argument("--in-map", default="", help="カメラID→動画パスの明示指定（未使用でもOK）")


    args = ap.parse_args()

    cam_ids = parse_cam_ids(args.cam_ids)
    order = [f"cam{cid}" for cid in cam_ids]

    # ===== ROI設定ファイルの場所 =====
    if args.out_dir:
        base_dir = Path(args.out_dir).resolve()
    else:
        base_dir = Path(args.root) / f"kaiseki{args.kaiseki}" / "concat"
    base_dir.mkdir(parents=True, exist_ok=True)

    roi_json = base_dir / "roi_config.json"
    out_video = base_dir / "walk_concat_roi.mp4"

    with open(roi_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    order = [n for n in order if n in cfg]
    assert order, "roi_config.jsonに対象カメラが見つかりません"

    # オフセット
    offset_map = parse_offset_map(args.offset_sec)

    # ===== 入力オープン =====
    caps = {}
    infos = {}
    for name in order:
        path = cfg[name]["path"]
        roi  = cfg[name]["roi"]
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), f"開けない: {path}"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # オフセット適用
        off = float(offset_map.get(name, 0.0))
        if off != 0.0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(off*fps)))
        caps[name] = cap
        infos[name] = {"fps": fps, "roi": roi}

    fps_out = min(infos[n]["fps"] for n in order)

    # ===== 出力サイズ算出 =====
    frames0 = []
    for name in order:
        cap = caps[name]
        ok, frame = cap.read(); assert ok, f"先頭フレーム取得失敗: {name}"
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(cap.get(cv2.CAP_PROP_POS_FRAMES)-1,0))
        r = infos[name]["roi"]; x,y,w,h = r["x"], r["y"], r["w"], r["h"]
        scale = args.target_height / h
        new_w = int(round(w * scale))
        frames0.append((new_w, args.target_height))

    W_out = sum(w for w,h in frames0)
    H_out = args.target_height

    writer, used_path, used_four = open_writer_with_fallback(out_video, fps_out, (W_out, H_out))
    print(f"[writer] {used_four} -> {used_path} size={W_out}x{H_out} fps={fps_out:.3f}")

    # ===== メインループ =====
    frames = 0
    while True:
        row = []
        for name in order:
            cap = caps[name]
            ok, frame = cap.read()
            if not ok:
                row = None
                break
            r = infos[name]["roi"]; x,y,w,h = r["x"], r["y"], r["w"], r["h"]
            crop = frame[y:y+h, x:x+w]
            scale = args.target_height / h
            new_w = int(round(w * scale))
            resized = cv2.resize(crop, (new_w, args.target_height), interpolation=cv2.INTER_LINEAR)
            row.append(resized)
        if row is None:
            break
        strip = cv2.hconcat(row)
        if (strip.shape[1], strip.shape[0]) != (W_out, H_out):
            strip = cv2.resize(strip, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
        writer.write(strip)
        frames += 1
        if frames == 1:
            cv2.imwrite(str(Path(base_dir) / "_debug_first.jpg"), strip)

    for cap in caps.values():
        cap.release()
    writer.release()
    print("saved:", used_path, "frames:", frames, "size:", (W_out, H_out))

if __name__ == "__main__":
    main()
