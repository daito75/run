import cv2, json, os
import numpy as np

roi_json = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\concat\a0.1\roi_config.json"
out_video = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\concat\a0.1\walk_concat_roi.mp4"

# オフセット（秒）で微同期させたいとき
offset_sec = {
    "camA": 0.0,
    "camB": 0.0,
}

# 出力高さ（両動画ともこの高さにスケール）
target_height = 480  # 360～720くらいが扱いやすい

with open(roi_json, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# 並べる順番（JSONの順を守りたい場合はここで指定）
order = list(cfg.keys())  # ["camA","camB"]
caps = {}
infos = {}

for name in order:
    path = cfg[name]["path"]
    roi  = cfg[name]["roi"]  # dict: x,y,w,h
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"開けない: {path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    # オフセットを反映
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(offset_sec.get(name,0.0)*fps)))
    caps[name] = cap
    infos[name] = {"fps": fps, "roi": roi}

# 出力fpsは最小に合わせる
fps_out = min(infos[n]["fps"] for n in order)

# 最初のフレームで幅計算
frames0 = []
for name in order:
    cap = caps[name]
    ok, frame = cap.read()
    assert ok, f"先頭フレーム取得失敗: {name}"
    # 一度読み出したのでバックしておく
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(cap.get(cv2.CAP_PROP_POS_FRAMES)-1,0))
    r = infos[name]["roi"]
    x,y,w,h = r["x"], r["y"], r["w"], r["h"]
    crop = frame[y:y+h, x:x+w]
    scale = target_height / h
    new_w = int(round(w * scale))
    frames0.append((new_w, target_height))

# 出力サイズ（横は合計）
W_out = sum(w for w,h in frames0)
H_out = target_height

os.makedirs(os.path.dirname(out_video), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video, fourcc, fps_out, (W_out, H_out))
if not writer.isOpened():
    out_video = out_video[:-4] + ".avi"
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"XVID"), fps_out, (W_out, H_out))
    assert writer.isOpened(), "出力作成に失敗"

frames = 0
while True:
    row = []
    for name in order:
        cap = caps[name]
        ok, frame = cap.read()
        if not ok:
            row = None
            break
        r = infos[name]["roi"]
        x,y,w,h = r["x"], r["y"], r["w"], r["h"]
        crop = frame[y:y+h, x:x+w]
        # 高さ基準でリサイズ（アスペクト保持）
        scale = target_height / h
        new_w = int(round(w * scale))
        resized = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        row.append(resized)
    if row is None:
        break
    # 横に連結
    strip = cv2.hconcat(row)
    # 念のためサイズを最終合わせ
    if (strip.shape[1], strip.shape[0]) != (W_out, H_out):
        strip = cv2.resize(strip, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
    writer.write(strip)
    frames += 1
    if frames == 1:
        cv2.imwrite(os.path.join(os.path.dirname(out_video), "_debug_first.jpg"), strip)

for cap in caps.values():
    cap.release()
writer.release()
print("saved:", out_video, "frames:", frames, "size:", (W_out, H_out))
