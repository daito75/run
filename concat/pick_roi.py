# pick_roi.py
import cv2, json, os, argparse
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

def pick_one(video_path, win_title, scale=0.5):
    cap = cv2.VideoCapture(video_path)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaiseki", type=int, required=True)
    ap.add_argument("--cam-ids", type=str, required=True)
    ap.add_argument("--a", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    cam_ids = parse_cam_ids(args.cam_ids)
    # 入力動画の場所: {root}\kaiseki{n}\syaeihenkann\a{a}\cam{ID}_on_plane.mp4
    a_dir = f"a{args.a}"
    videos = []
    for cid in cam_ids:
        name = f"cam{cid}"
        path = rf"{args.root}\kaiseki{args.kaiseki}\syaeihenkann\{a_dir}\{name}_on_plane.mp4"
        videos.append({"name": name, "path": path})

    roi_json = rf"{args.root}\kaiseki{args.kaiseki}\concat\{a_dir}\roi_config.json"
    os.makedirs(os.path.dirname(roi_json), exist_ok=True)

    rois = {}
    for v in videos:
        print(f"ROI選択: {v['name']}  {v['path']}")
        rois[v["name"]] = {
            "path": v["path"],
            "roi": pick_one(v["path"], f"Pick ROI - {v['name']}", scale=args.scale)
        }

    with open(roi_json, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2, ensure_ascii=False)
    print("保存しました:", roi_json)

if __name__ == "__main__":
    main()
