import cv2
import numpy as np

# === 入力と出力の設定 ===
input_path = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki3\syaeihenkann\normal\cam2_on_plane.mp4"     # 入力動画
output_path = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki3\syaeihenkann\normal\cam2_on_kansei.mp4"  # 出力動画
rotate_angle = 0.2                    # 回転角度（°）正: 反時計回り, 負: 時計回り

# === 動画の読み込み ===
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("動画が開けませんでした。パスを確認してください。")

# 動画情報の取得
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# === 出力設定 ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# === 回転処理 ===
center = (w // 2, h // 2)
rot_mat = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rotated = cv2.warpAffine(frame, rot_mat, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    out.write(rotated)

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ 回転完了:", output_path)
