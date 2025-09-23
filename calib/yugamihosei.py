# undistort_video_quick.py
import cv2, yaml, numpy as np, os
IN_MP4  = r"D:\BRLAB\2025\mizuno\done\test\cam2\cam2_walk.mp4"
OUT_MP4 = r"D:\BRLAB\2025\mizuno\done\test\cam2\cam2_walk_hosei.mp4"
CALIB   = r"D:\BRLAB\2025\mizuno\done\calib\cam2\cam2_1080p.yaml"

with open(CALIB, "r", encoding="utf-8") as f:
    y = yaml.safe_load(f)
K    = np.array(y["camera_matrix"], dtype=np.float32)
dist = np.array(y["dist_coeffs"], dtype=np.float32)

cap = cv2.VideoCapture(IN_MP4); assert cap.isOpened()
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w,h), cv2.CV_16SC2)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
out = cv2.VideoWriter(OUT_MP4, fourcc, fps, (w,h))

while True:
    ret, frm = cap.read()
    if not ret: break
    und = cv2.remap(frm, map1, map2, cv2.INTER_LINEAR)
    out.write(und)
cap.release(); out.release()
print("wrote:", OUT_MP4)
