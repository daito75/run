import cv2, json, os

videos = [
    # 左右の順番で並べたい順に書く
    {"name": "camA", "path": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\a0.1\cam1_on_plane.mp4"},
    #{"name": "camB", "path": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\a0.1\cam2_on_plane.mp4"},
    {"name": "camC", "path": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\a0.1\cam3_on_plane.mp4"},
    #{"name": "camD", "path": r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\a0.1\cam4_on_plane.mp4"},
]
roi_json = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\concat\a0.1\roi_config.json"

def pick_one(video_path, win_title, scale=0.5):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"先頭フレームが読み込めません: {video_path}"
    H, W = frame.shape[:2]

    # 縮小して表示
    disp = cv2.resize(frame, (int(W*scale), int(H*scale)))
    r = cv2.selectROI(win_title, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win_title)

    # ROI を元スケールに戻す
    x, y, w, h = [int(v/scale) for v in r]
    return {"x": x, "y": y, "w": w, "h": h}


os.makedirs(os.path.dirname(roi_json), exist_ok=True)
rois = {}
for v in videos:
    print(f"ROI選択: {v['name']}  {v['path']}")
    rois[v["name"]] = {
        "path": v["path"],
        "roi": pick_one(v["path"], f"Pick ROI - {v['name']}")
    }

with open(roi_json, "w", encoding="utf-8") as f:
    json.dump(rois, f, indent=2, ensure_ascii=False)
print("保存しました:", roi_json)
