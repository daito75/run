# undistort_video_hq.py
import os, cv2, yaml, numpy as np

IN_MP4  = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\walk\cam1_walk.mp4"
OUT_MP4 = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\a75\cam1_on_plane.mp4"
CALIB   = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\cam1_1080p.yaml"

# 設定
ALPHA = 0.7          # 0..1（0で強めにトリミング、1で黒縁許容して広く）
CROP_VALID_ROI = True  # True なら有効画素領域でクロップ
USE_FISHEYE = True    # 魚眼キャリブなら True（yamlがfisheye用ならここもTrue）

with open(CALIB, "r", encoding="utf-8") as f:
    y = yaml.safe_load(f)
K    = np.array(y["camera_matrix"], dtype=np.float64).reshape(3,3)
dist = np.array(y["dist_coeffs"],   dtype=np.float64).ravel()

cap = cv2.VideoCapture(IN_MP4); assert cap.isOpened(), IN_MP4
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

if not USE_FISHEYE:
    # 通常モデル：32Fマップ + Lanczos4 で高品質
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=ALPHA)  # roi=(x,y,w,h)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w,h), cv2.CV_32FC1)
else:
    # 魚眼モデル
    newK = K.copy()
    R = np.eye(3)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, dist, R, newK, (w,h), cv2.CV_32FC1)
    roi = (0,0,w,h)

# 出力サイズを決定（ROIでクロップするならその大きさ）
if CROP_VALID_ROI and roi is not None and all(v>0 for v in roi):
    x,y,w2,h2 = map(int, roi)
    out_size = (w2, h2)
else:
    x,y = 0,0
    out_size = (w,h)

# H.264系は偶数が安全
W,H = out_size
if W % 2: W += 1
if H % 2: H += 1
out_size = (W,H)

fourccs = [("mp4v",".mp4"), ("avc1",".mp4"), ("H264",".mp4"), ("MJPG",".avi")]
os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
writer = None; out_path = OUT_MP4
for fourcc,ext in fourccs:
    p = out_path if out_path.lower().endswith(ext) else os.path.splitext(out_path)[0]+ext
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*fourcc), float(fps), out_size)
    if vw.isOpened():
        writer = vw; out_path = p; break
assert writer is not None, "VideoWriterを開けませんでした"

while True:
    ok, frm = cap.read()
    if not ok: break
    und = cv2.remap(frm, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    if CROP_VALID_ROI and (x or y):
        und = und[y:y+out_size[1], x:x+out_size[0]]  # 一旦予定サイズに近い切り出し
    if (und.shape[1], und.shape[0]) != out_size:
        und = cv2.resize(und, out_size, interpolation=cv2.INTER_AREA)
    writer.write(und)

cap.release(); writer.release()
print("wrote:", out_path, "size:", out_size, "fps:", fps)
