import cv2, json, os
import numpy as np
from math import sqrt, isfinite

roi_json = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\concat\a0.1\roi_config_quad.json"
out_video = r"D:\BRLAB\2025\mizuno\done\deta\kaiseki2\concat\a0.1\walk_concat_roi.mp4"

# 秒オフセット（微同期）
offset_sec = {
    "camA": 0.0,
    "camB": 0.0,
    "camC": 0.0,
}

# 出力の各ストリップ高さ
target_height = 480  # 360～720くらいが扱いやすい
VERBOSE = True

def log(*a):
    if VERBOSE:
        print(*a)

def _edge_len(p, q):
    return sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

def _new_width_from_rect(w, h):
    if h <= 0 or w <= 0:
        return 0
    return int(round(w * (target_height / float(h))))

def _new_width_from_quad(quad):
    # 期待順: 左上, 右上, 右下, 左下（順不同でも長さ平均で耐える）
    p0, p1, p2, p3 = quad
    top = _edge_len(p0, p1)
    bottom = _edge_len(p3, p2)
    left = _edge_len(p0, p3)
    right = _edge_len(p1, p2)
    avg_w = 0.5 * (top + bottom)
    avg_h = 0.5 * (left + right)
    if avg_h < 1e-6:
        return 0
    aspect = avg_w / avg_h
    return max(2, int(round(target_height * aspect)))

def _warp_quad_to_rect(frame, quad, out_w, out_h):
    src = np.array(quad, dtype=np.float32)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, H, (out_w, out_h), flags=cv2.INTER_LINEAR)

def _open_video_writer_with_fallback(path, fps, size_wh):
    W_out, H_out = size_wh
    if fps <= 0:
        raise ValueError(f"VideoWriter fps<=0: {fps}")
    if W_out <= 0 or H_out <= 0:
        raise ValueError(f"VideoWriter size invalid: {size_wh}")

    trials = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("H264", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]
    base, _ = os.path.splitext(path)
    for fourcc_str, ext in trials:
        out = base + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(out, fourcc, fps, (W_out, H_out))
        log(f"[writer] try {fourcc_str} -> {out}  opened={vw.isOpened()}")
        if vw.isOpened():
            return vw, out
    raise RuntimeError("出力作成に失敗（全コーデックNG）")

# ========== ロード ==========
with open(roi_json, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# 並べる順序（JSONの順）
order = list(cfg.keys())
if len(order) == 0:
    raise RuntimeError("ROI JSONが空です")

caps = {}
infos = {}

# ========== 各カメラを開いて開始位置へ ==========
for name in order:
    if "path" not in cfg[name]:
        raise ValueError(f"{name}: 'path' が JSON にありません")
    path = cfg[name]["path"]
    roi  = cfg[name].get("roi", {})
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"開けない: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not isfinite(fps) or fps <= 0:
        # 古いコーデックやVFRで 0 が返ることがある。暫定で 30fps を採用
        log(f"[warn] {name}: fps({fps}) が不正。fallback=30.0")
        fps = 30.0

    saved_idx = int(roi.get("frame_idx", 0)) if isinstance(roi, dict) else 0
    start_idx = max(0, saved_idx + int(round(offset_sec.get(name, 0.0) * fps)))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        log(f"[warn] {name}: 総フレーム数が取れない/0。とりあえず start_idx=0 に")
        total = 1
        start_idx = 0

    if start_idx >= total:
        log(f"[warn] {name}: start_idx({start_idx}) >= total({total}) → 最終フレームへ調整")
        start_idx = max(0, total - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    caps[name] = cap
    infos[name] = {"fps": fps, "roi": roi, "start_idx": start_idx, "total": total, "path": path}

    log(f"[open] {name}: path={path}")
    log(f"       fps={fps:.3f}, total={total}, start_idx={start_idx}, offset_sec={offset_sec.get(name,0.0)}")
    log(f"       roi={roi}")

# 出力 fps は最小に合わせる
fps_out = min(infos[n]["fps"] for n in order)
log(f"[fps_out] {fps_out}")

# ========== 最初のフレームで各ストリップ幅を見積もり ==========
frames0 = []
first_frames = {}
for name in order:
    cap = caps[name]
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"{name}: 最初の read() に失敗（start_idx={infos[name]['start_idx']}）")
    # 次のループで同じ位置から始めたいので1フレーム戻す
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
    first_frames[name] = frame

    H0, W0 = frame.shape[:2]
    r = infos[name]["roi"]

    if all(k in r for k in ("x","y","w","h")):
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        # クリップ
        x = max(0, min(x, W0-1))
        y = max(0, min(y, H0-1))
        w = max(1, min(w, W0 - x))
        h = max(1, min(h, H0 - y))
        new_w = _new_width_from_rect(w, h)
        log(f"[rect] {name}: (x,y,w,h)=({x},{y},{w},{h}) -> new_w={new_w}")
    elif "quad" in r and isinstance(r["quad"], (list, tuple)) and len(r["quad"]) == 4:
        quad = [(int(px), int(py)) for (px, py) in r["quad"]]
        # 画面内に軽くクリップ（はみ出し気味でも動くように）
        quad = [(max(0, min(px, W0-1)), max(0, min(py, H0-1))) for (px, py) in quad]
        new_w = _new_width_from_quad(quad)
        log(f"[quad] {name}: quad={quad} -> new_w={new_w}")
        # 万一 new_w が小さ過ぎる場合の救済
        if new_w < 2:
            # バウンディング矩形で代替
            xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
            bx, by = min(xs), min(ys)
            bw, bh = max(xs)-bx+1, max(ys)-by+1
            new_w = _new_width_from_rect(bw, bh)
            log(f"[quad->rect fallback] {name}: bbox=(x={bx},y={by},w={bw},h={bh}) -> new_w={new_w}")
            # ROIも便宜的に矩形切り抜きに差し替え
            infos[name]["roi"] = {"x": bx, "y": by, "w": max(1,bw), "h": max(1,bh)}
    else:
        raise ValueError(f"{name}: ROIフォーマット不正: {r}")

    if new_w <= 0:
        raise RuntimeError(f"{name}: 推定 new_w が 0（ROIやフレームに問題）")

    frames0.append((new_w, target_height))

W_out = sum(w for w, _ in frames0)
H_out = target_height
log(f"[out size] {W_out}x{H_out}")

# ========== ライターを開く ==========
writer, out_video_path = _open_video_writer_with_fallback(out_video, fps_out, (W_out, H_out))
log(f"[writer] using -> {out_video_path}")

# ========== 本ループ ==========
frames = 0
while True:
    row = []
    for name in order:
        cap = caps[name]
        ok, frame = cap.read()
        if not ok or frame is None:
            log(f"[eof] {name}: read() -> False at frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
            row = None
            break

        r = infos[name]["roi"]
        H0, W0 = frame.shape[:2]

        if all(k in r for k in ("x","y","w","h")):
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            x = max(0, min(x, W0-1))
            y = max(0, min(y, H0-1))
            w = max(1, min(w, W0 - x))
            h = max(1, min(h, H0 - y))

            scale = target_height / float(h)
            new_w = max(1, int(round(w * scale)))
            crop = frame[y:y+h, x:x+w]
            resized = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        else:
            quad = [(int(px), int(py)) for (px, py) in r["quad"]]
            quad = [(max(0, min(px, W0-1)), max(0, min(py, H0-1))) for (px, py) in quad]
            new_w = _new_width_from_quad(quad)
            if new_w < 2:
                # 代替（バウンディング矩形）
                xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
                bx, by = min(xs), min(ys)
                bw, bh = max(xs)-bx+1, max(ys)-by+1
                new_w = _new_width_from_rect(bw, bh)
                crop = frame[by:by+bh, bx:bx+bw]
                resized = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
            else:
                resized = _warp_quad_to_rect(frame, quad, new_w, target_height)

        row.append(resized)

    if row is None:
        break

    strip = cv2.hconcat(row)
    if (strip.shape[1], strip.shape[0]) != (W_out, H_out):
        strip = cv2.resize(strip, (W_out, H_out), interpolation=cv2.INTER_LINEAR)

    writer.write(strip)
    frames += 1
    if frames == 1:
        dbg = os.path.join(os.path.dirname(out_video_path), "_debug_first.jpg")
        cv2.imwrite(dbg, strip)
        log(f"[debug] wrote {dbg}")

for cap in caps.values():
    cap.release()
writer.release()
print("saved:", out_video_path, "frames:", frames, "size:", (W_out, H_out))
