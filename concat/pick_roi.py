# pick_roi.py
import cv2, json, os, argparse, glob
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

def parse_in_map(s: str):
    """
    "1:D:\\p1.mp4;3:D:\\p2.mp4" -> { "cam1": Path(...), "cam3": Path(...) }
    """
    out = {}
    if not s:
        return out
    for pair in s.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"--in-map 形式エラー: {pair}")
        k, v = pair.split(":", 1)
        cam = f"cam{int(k.strip())}"
        out[cam] = Path(v.strip()).resolve()
    return out

def pick_one(video_path, win_title, scale=0.5):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    assert ok, f"先頭フレームが読み込めません: {video_path}"
    H, W = frame.shape[:2]
    disp = cv2.resize(frame, (int(W*scale), int(H*scale)))
    r = cv2.selectROI(win_title, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win_title)
    x, y, w, h = [int(v/scale) for v in r]
    return {"x": x, "y": y, "w": w, "h": h}

def main():
    ap = argparse.ArgumentParser(description="各カメラのROI選択（α廃止・入出力柔軟化）")
    ap.add_argument("--kaiseki", type=int, required=True)
    ap.add_argument("--cam-ids", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--scale", type=float, default=0.5)

    # ★ 新規：柔軟入出力
    ap.add_argument("--in-map", default="", help='明示入力 "1:D:\\p1.mp4;3:D:\\p2.mp4"')
    ap.add_argument("--out-dir", default="", help="ROI設定ファイルの保存先（未指定なら既定パス）")

    args = ap.parse_args()

    cam_ids = parse_cam_ids(args.cam_ids)
    cam_names = [f"cam{cid}" for cid in cam_ids]

    # ===== 入力動画の解決 =====
    in_map = parse_in_map(args.in_map)  # 優先
    videos = []
    for name in cam_names:
        if name in in_map:
            videos.append({"name": name, "path": in_map[name]})
        else:
            # 既定パス（α廃止）。必要ならここを自分の既定構成に合わせて調整してね。
            # 例: ...\kaiseki{n}\syaeihenkann\normal\camX_on_plane.mp4 を探す
            default_candidates = [
                Path(args.root) / f"kaiseki{args.kaiseki}" / "syaeihenkann" / "normal" / f"{name}_on_plane.mp4",
                Path(args.root) / f"kaiseki{args.kaiseki}" / "syaeihenkann" / f"{name}_on_plane.mp4",
            ]
            hit = None
            for p in default_candidates:
                if p.exists():
                    hit = p; break
            if hit is None:
                raise FileNotFoundError(f"入力動画が見つかりません: {name}（--in-map で明示指定推奨）")
            videos.append({"name": name, "path": hit})

    # ===== ROI設定ファイルの保存先 =====
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = Path(args.root) / f"kaiseki{args.kaiseki}" / "concat"
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_json = out_dir / "roi_config.json"

    # ===== ROI選択 =====
    rois = {}
    for v in videos:
        print(f"ROI選択: {v['name']}  {v['path']}")
        rois[v["name"]] = {
            "path": str(v["path"]),
            "roi": pick_one(v["path"], f"Pick ROI - {v['name']}", scale=args.scale)
        }

    with open(roi_json, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2, ensure_ascii=False)
    print("保存しました:", str(roi_json))

if __name__ == "__main__":
    main()
