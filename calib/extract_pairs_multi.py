# extract_pairs_multi.py
import cv2, os, argparse

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaiseki", type=int, required=True)
    ap.add_argument("--cam-ids", type=str, required=True)  # 例 "1-3,5"
    ap.add_argument("--root", required=True)
    ap.add_argument("--interval-sec", type=float, default=1.0)
    args, _ = ap.parse_known_args()

    k = args.kaiseki
    cam_ids = parse_cam_ids(args.cam-ids if hasattr(args, "cam-ids") else args.cam_ids)  # safety
    root = args.root.rstrip("\\/")
    interval_sec = args.interval_sec

    out_root = rf"{root}\kaiseki{k}\frame"
    os.makedirs(out_root, exist_ok=True)
    print(f"[INFO] kaiseki{k}, cam_ids={cam_ids}, out={out_root}, interval={interval_sec}s")

    for i in cam_ids:
        name = f"cam{i}"
        path = rf"{root}\kaiseki{k}\chess\{name}_chess.mp4"
        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), f"open fail: {path}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur_sec = total / fps if fps > 0 else 0.0

        out_dir = rf"{out_root}\{name}"
        os.makedirs(out_dir, exist_ok=True)

        saved = 0; t = 0.0
        while t <= dur_sec + 1e-6:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok: ok, frame = cap.read()
            if ok:
                fname = f"{int(round(t)):06d}.png"
                cv2.imwrite(os.path.join(out_dir, fname), frame); saved += 1
            t += interval_sec

        cap.release()
        print(f"[done] {name}: fps≈{fps:.2f}, dur≈{dur_sec:.2f}s -> saved={saved}")

if __name__ == "__main__":
    main()
