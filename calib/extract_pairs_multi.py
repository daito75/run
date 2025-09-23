# extract_pairs_multi.py
import cv2, os

# ここにキャリブ用チェス動画を並べる（Cam1が基準）
cams = [
    {"name": "cam1", "video": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\chess\cam1_chess.mp4"},
    {"name": "cam2", "video": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\chess\cam2_chess.mp4"},
    {"name": "cam3", "video": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\chess\cam3_chess.mp4"},
    # 追加例:
    # {"name":"cam3", "video": r"...\cam3.mp4"},
]

out_root = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame"
interval_sec = 1.0  # ← ここを 0.5 とかにすれば0.5秒刻み

os.makedirs(out_root, exist_ok=True)

for c in cams:
    name, path = c["name"], c["video"]
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"open fail: {path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur_sec = total / fps if fps > 0 else 0
    os.makedirs(os.path.join(out_root, name), exist_ok=True)

    saved = 0
    t = 0.0
    while t <= dur_sec + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            # set直後に落ちる環境対策でワンモアトライ
            ok, frame = cap.read()
        if ok:
            # ファイル名は “秒” ベースで揃える（000000, 000001, ...）
            fname = f"{int(round(t)):06d}.png"
            cv2.imwrite(os.path.join(out_root, name, fname), frame)
            saved += 1
        t += interval_sec

    cap.release()
    print(f"[done] {name}: fps≈{fps:.2f}, dur≈{dur_sec:.2f}s -> saved={saved}")