# convert_yaml_to_filestorage.py
# safe_dump形式のYAML（camera_matrix/dist_coeffs）をOpenCV FileStorage形式（K/dist）に変換
import os, glob, yaml
import numpy as np
import cv2

# 変換したいファイルのパターンを指定（例：全カメラ）
GLOBS = [
    r"D:\BRLAB\2025\mizuno\done\calib\cam1\cam1_1080p.yaml",
    r"D:\BRLAB\2025\mizuno\done\calib\cam2\cam2_1080p.yaml",
    r"D:\BRLAB\2025\mizuno\done\calib\cam3\cam3_1080p.yaml",
    r"D:\BRLAB\2025\mizuno\done\calib\cam4\cam4_1080p.yaml",
    r"D:\BRLAB\2025\mizuno\done\calib\cam5\cam5_1080p.yaml",
]

def looks_like_opencv_yaml(path: str) -> bool:
    # 既にFileStorage形式（%YAML:1.0 で K/dist だけが入ってる）か簡易判定
    try:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            K = fs.getNode("K").mat()
            dist = fs.getNode("dist").mat()
            fs.release()
            return (K is not None) and (dist is not None)
    except Exception:
        pass
    return False

def convert_one(path: str):
    # すでにOpenCV形式ならスキップ
    if looks_like_opencv_yaml(path):
        print(f"[SKIP] already OpenCV FS: {path}")
        return

    # safe_dump形式を読む
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # キー名に柔軟対応
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"], dtype=np.float32)
    elif "K" in data:
        K = np.array(data["K"], dtype=np.float32)
    else:
        print(f"[WARN] no camera matrix in {path}")
        return

    if "dist_coeffs" in data:
        dist = np.array(data["dist_coeffs"], dtype=np.float32).reshape(-1,1)
    elif "dist" in data:
        dist = np.array(data["dist"], dtype=np.float32).reshape(-1,1)
    else:
        print(f"[WARN] no dist coeffs in {path}")
        return

    # 出力先（_opencv.yaml を付ける）
    root, ext = os.path.splitext(path)
    out_path = root + "_opencv.yaml"

    # OpenCV FileStorage 形式で書く（K/dist必須、他メタもあれば併記可）
    fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("dist", dist)
    # あるならメタも併記（読み手が無視してもOK）
    for k in ["image_width","image_height","checker_inner_corners","square_size_mm","rms"]:
        if k in data:
            fs.write(k, np.array(data[k]))
    fs.release()

    print(f"[OK] {path} -> {out_path}")

def main():
    files = []
    for pat in GLOBS:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        print("[INFO] no yaml files matched")
        return
    for p in files:
        try:
            convert_one(p)
        except Exception as e:
            print(f"[ERR] convert fail: {p} -> {e}")

if __name__ == "__main__":
    main()
