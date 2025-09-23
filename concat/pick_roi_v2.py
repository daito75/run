import cv2, json, os

videos = [
    {"name": "camA", "path": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\a0.1\cam1_on_plane.mp4"},
    {"name": "camC", "path": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\a0.1\cam3_on_plane.mp4"},
]
roi_json = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\concat\a0.1\roi_config_quad.json"

def pick_quad(video_path, win_title, frame_idx=0, scale=0.5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 指定フレームに移動（範囲外なら最後のフレームに）
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total - 1))
    ok, frame = cap.read()
    cap.release()
    assert ok, f"フレーム {frame_idx} が読み込めません: {video_path}"
    H, W = frame.shape[:2]

    disp = cv2.resize(frame, (int(W*scale), int(H*scale)))
    clone = disp.copy()
    points = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal points, disp
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(win_title, disp)

    cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, mouse_cb)

    print(f"フレーム {frame_idx}/{total} ({frame_idx/fps:.2f} 秒) で四隅をクリックしてください: {video_path}")
    while True:
        cv2.imshow(win_title, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(points) == 4:  # Enterで確定
            break
        elif key == 27:  # Escでキャンセル
            points = []
            break

    cv2.destroyWindow(win_title)

    quad = [(int(px/scale), int(py/scale)) for (px, py) in points]
    return {"quad": quad, "frame_idx": frame_idx}


os.makedirs(os.path.dirname(roi_json), exist_ok=True)
rois = {}
for v in videos:
    print(f"ROI選択: {v['name']}  {v['path']}")
    rois[v["name"]] = {
        "path": v["path"],
        # frame_idxを自由に指定できる（例: 100フレーム目）
        "roi": pick_quad(v["path"], f"Pick Quad ROI - {v['name']}", frame_idx=100)
    }

with open(roi_json, "w", encoding="utf-8") as f:
    json.dump(rois, f, indent=2, ensure_ascii=False)
print("保存しました:", roi_json)
