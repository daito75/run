#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
HQ version: YAML(K,D) + 歩行動画 → (ArUco複数フレーム) → 実世界平面[m]へのHを頑健推定 → pxへスケール → 各動画を高品質warp

- 歪み補正: CV_32FC1マップ + Lanczos4 (alpha可変)
- H推定: 複数フレーム / サブピクセル / solvePnP(+LM) / 姿勢平均(SVD直交化)
- warp: Lanczos4, 必要に応じてスーパーサンプリング→AREAダウンサンプル

python D:\BRLAB\2025\mizuno\done\run\syaeihenkann\batch_warp_plane_hq.py `
  --pair "D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\cam1_1080p.yaml|D:\BRLAB\2025\mizuno\done\deta\kaiseki2\walk\cam1_walk.mp4|D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\a0\cam1_on_plane_sum.mp4" `
  --dict 4X4_50 `
  --fisheye `
  --marker-size-cm 14.5 `
  --px-per-m 200 `
  --alpha 1.0 `
  --scan-step 1 `
  --scan-max 2000 `
  --candidates-top 30 `
  --reproj-thresh-px 2.0 `
  --superres 1.25 `
  --debug-dir D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\ninsiki\a0 `
  --save-best-k 5 `
  --calib-size 1920x1080 `
  --select-detect-roi `
  --select-roi-frame 100 `
  --expected-id 0 `
  --auto-upright `
  --upright-edge 01 `
  --board-layout "D:\BRLAB\2025\mizuno\done\run\syaeihenkann\board-layout.yml" `
  --quiet `
  --crop-roi `

python D:\BRLAB\2025\mizuno\done\run\syaeihenkann\batch_warp_plane_hq.py `
  --pair "D:\BRLAB\2025\mizuno\done\deta\kaiseki2\frame\cam1\cam1_1080p.yaml|D:\BRLAB\2025\mizuno\done\deta\kaiseki2\walk\cam1_walk.mp4|D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\a0\cam1_on_plane_sum.mp4" `
  --dict 4X4_50 `
  --fisheye `
  --marker-size-cm 14.5 `
  --px-per-m 200 `
  --alpha 1.0 `
  --superres 1.25 `
  --debug-dir D:\BRLAB\2025\mizuno\done\deta\kaiseki2\syaeihenkann\ninsiki\a0 `
  --calib-size 1920x1080 `
  --select-detect-roi `
  --select-roi-frame 100 `
  --expected-id 0 `
  --auto-upright `
  --upright-edge 01 `
  --board-layout "D:\BRLAB\2025\mizuno\done\run\syaeihenkann\board-layout.yml" `
  --crop-roi `
  --solve 4pts `
  --warp-impl numpy `
  --progress-interval-sec 30.0

"""

import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import csv
import cv2
import numpy as np
import yaml
import concurrent.futures as futures
import numpy as _np
# ---------- [ADD] 進捗ログ用 ----------
import time

class _ProgressTicker:
    def __init__(self, total: int, interval_sec: float = 2.0):
        self.total = max(int(total), 0)
        self.interval = float(interval_sec)
        self.start = time.time()
        self.last = self.start

    def tick(self, done: int, prefix: str = ""):
        """呼ぶたびに、intervalを超えていたらログを返す。返り値がNoneなら何もしないでOK。"""
        now = time.time()
        if (now - self.last) < self.interval:
            return None
        self.last = now
        elapsed = now - self.start
        done = max(0, int(done))
        pct = (100.0 * done / self.total) if self.total > 0 else 0.0
        fps = (done / elapsed) if elapsed > 0 else 0.0
        eta = ((self.total - done) / fps) if (fps > 0 and self.total > 0) else float("inf")
        if eta == float("inf"):
            eta_txt = "ETA: --"
        else:
            m, s = divmod(int(eta + 0.5), 60)
            h, m = divmod(m, 60)
            eta_txt = f"ETA: {h:02d}:{m:02d}:{s:02d}"
        m, s = divmod(int(elapsed + 0.5), 60)
        h, m = divmod(m, 60)
        el_txt = f"elapsed: {h:02d}:{m:02d}:{s:02d}"
        msg = f"{prefix} {done}/{self.total} ({pct:5.1f}%) | {fps:5.1f} fps | {el_txt} | {eta_txt}"
        return msg



#先頭付近のimportの後あたりに追記
def set_opencv_quiet(quiet: bool):  #警告をなくす定義
    if not quiet:  #入力で--quietと入れてないと何もなし
        return
    try:  #入力で--quietと入れていると以下の処理が行われる
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  #SILENTが完全に黙らせる文
    except Exception:  #上記で適応されない場合
        pass  #古いOpenCVでも無視


def open_video_writer_with_fallback(out_path: Path, fps: float, size_wh: tuple):
    """複数コーデックで順に試す。最後はMJPG(AVI)にフォールバック。"""
    trials = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("H264", ".mp4"),
        ("MJPG", ".avi"),
    ]
    for fourcc, ext in trials:  #各候補を順にチェック
        use_path = out_path
        if ext != out_path.suffix.lower():  #拡張子が違えば差し替え
            use_path = out_path.with_suffix(ext)
        writer = cv2.VideoWriter(  #VideoWriterを生成
            str(use_path),
            cv2.VideoWriter_fourcc(*fourcc),  #fourccコード指定
            float(fps),
            (int(size_wh[0]), int(size_wh[1])),  #出力サイズ (width,height)
            True,
        )
        if writer.isOpened():  #正しく開けたら
            return writer, use_path, fourcc  #Writer, 実際のパス, コーデックを返す
    raise RuntimeError(  #全部ダメなら例外を出す
        "VideoWriterを開けませんでした（利用可能なコーデックが見つからない）"
    )

def homography_from_4pts(src_xy: _np.ndarray, dst_xy: _np.ndarray) -> _np.ndarray:
    """
    src_xy, dst_xy: (4,2) float64
    戻り: H (3x3) で src→dst
    """
    assert src_xy.shape == (4,2) and dst_xy.shape == (4,2)
    A, b = [], []
    for (x, y), (xp, yp) in zip(src_xy, dst_xy):
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp]); b.append(xp)
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp]); b.append(yp)
    A = _np.asarray(A, _np.float64)   # 8x8
    b = _np.asarray(b, _np.float64)   # 8
    h = _np.linalg.solve(A, b)        # [a,b,c,d,e,f,g,h]
    H = _np.array([[h[0], h[1], h[2]],
                   [h[3], h[4], h[5]],
                   [h[6], h[7], 1.0]], dtype=_np.float64)
    return H

def warp_by_homography_numpy(img: _np.ndarray, H: _np.ndarray, out_wh=None, border_value=0):
    """
    img: HxW[xC], uint8/float
    H:   src→dst
    out_wh: (W_out, H_out)
    """
    if img.ndim == 2:
        H0, W0, C = img.shape[0], img.shape[1], 1
        img_ch = img[..., None]
    else:
        H0, W0, C = img.shape
        img_ch = img

    if out_wh is None:
        W, Hh = W0, H0
    else:
        W, Hh = int(out_wh[0]), int(out_wh[1])

    xx, yy = _np.meshgrid(_np.arange(W, dtype=_np.float64),
                          _np.arange(Hh, dtype=_np.float64))
    ones = _np.ones_like(xx)
    dst_h = _np.stack([xx, yy, ones], axis=0).reshape(3, -1)

    Hinv = _np.linalg.inv(H)
    src_h = Hinv @ dst_h
    w = src_h[2, :] + 1e-12
    xs = (src_h[0, :] / w).reshape(Hh, W)
    ys = (src_h[1, :] / w).reshape(Hh, W)

    x0 = _np.floor(xs).astype(_np.int64); y0 = _np.floor(ys).astype(_np.int64)
    x1 = x0 + 1;                          y1 = y0 + 1

    wx = xs - x0; wy = ys - y0
    x0c = _np.clip(x0, 0, W0-1); x1c = _np.clip(x1, 0, W0-1)
    y0c = _np.clip(y0, 0, H0-1); y1c = _np.clip(y1, 0, H0-1)
    inside = (xs >= 0) & (xs <= W0-1) & (ys >= 0) & (ys <= H0-1)

    out = _np.empty((Hh, W, C), dtype=img_ch.dtype)
    w00 = (1-wx)*(1-wy); w10 = wx*(1-wy); w01 = (1-wx)*wy; w11 = wx*wy

    for c in range(C):
        I00 = img_ch[y0c, x0c, c]; I10 = img_ch[y0c, x1c, c]
        I01 = img_ch[y1c, x0c, c]; I11 = img_ch[y1c, x1c, c]
        out[..., c] = (w00*I00 + w10*I10 + w01*I01 + w11*I11)
        bv = _np.array(border_value, dtype=out.dtype) if _np.issubdtype(out.dtype, _np.integer) else float(border_value)
        out[..., c] = _np.where(inside, out[..., c], bv)

    return out[..., 0] if C == 1 else out

#========== 基本ユーティリティ ==========
def load_cam_yaml(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:  #入力したyamlファイルから内部行列Kと歪み係数Dを読み込む
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    K = np.array(y["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(y["dist_coeffs"], dtype=np.float64).ravel()
    return K, D
def build_undistort_maps_hq(K: np.ndarray, D: np.ndarray, size_wh: Tuple[int, int], alpha: float = 0.75):
    """高精度歪み補正用のマップ(newK含む)を作る（CV_32FC1 + alpha>0推奨）"""
    w, h = size_wh
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=alpha)  #newK:補正後の見え方を反映した新しい内部行列K, ROI黒帯無しの有効領域
    #ゆがみ補正後は、切り取りや拡大縮小の影響でみかけの内部行列が変化する
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, newK, (w, h), m1type=cv2.CV_32FC1
    )  #map1,map2補正用の座標マップ（元画像のどこから画素を拾うのか）
    return map1, map2, newK, roi


def build_undistort_maps_fisheye(K, D, size_wh, alpha=0.0): #魚眼レンズの歪み補正用マップを作成
    w, h = size_wh #内部行列をコピー
    #fisheyeはalphaの概念が違う。newKはKを好みに調整
    newK = K.copy()
    R = np.eye(3)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, newK, (w, h), cv2.CV_32FC1 #歪みパラメータを使って補正マップを生成
    )
    roi = (0, 0, w, h)
    return map1, map2, newK, roi #使えるマップと新しい内部行列を返す

def parse_wh(s: str) -> Tuple[int, int]: #文字列を受け取って、整数に変換する関数(画像サイズ"1980×1080")
    w, h = s.lower().split("x") #"×"を区切りに分割
    return (int(w), int(h)) #整数に返す


def parse_roi(s: str) -> Optional[Tuple[int, int, int, int]]: #文字列を受け取って、整数に変換する関数(短形領域"100,200,600,400")
    if not s:
        return None
    x, y, w, h = [int(v) for v in s.replace(" ", "").split(",")] #カンマ区切りを整数化
    return (x, y, w, h)


def scale_intrinsics(K: np.ndarray, calib_wh: Tuple[int, int], video_wh: Tuple[int, int]) -> np.ndarray: #キャリブ画像と動画の解像度の違いを補正.歪み補正や射影変換をかけるとき、K が動画サイズに合ってないと結果が歪んでしまう
    sx = video_wh[0] / calib_wh[0] #幅方向の拡大率
    sy = video_wh[1] / calib_wh[1] #高さ方向の拡大率
    K2 = K.copy()
    K2[0, 0] *= sx   #fx を横方向スケール
    K2[1, 1] *= sy   #fy を縦方向スケール
    K2[0, 2] *= sx   #cx を横方向スケール
    K2[1, 2] *= sy   #cy を縦方向スケール
    return K2


def _fit_to_screen(img, max_w=1600, max_h=1200):
    h, w = img.shape[:2]   #入力画像の高さh, 幅w を取得
    s = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    #→ 横幅方向と高さ方向の「縮小率」を計算し、その最小値をスケールにする
    #さらに 1.0 と比較して、拡大はせず縮小のみに制限

    if s < 1.0:
        #縮小が必要なら、INTER_AREA でリサイズ（縮小に強い補間方法）
        disp = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    else:
        #縮小不要なら元の画像のまま
        disp = img

    return disp, s


class _RoiSelector: #画面上で矩形をドラッグして切り取り範囲（ROI）を選ぶ
    def __init__(self, img_bgr, win="Select ROI"):
        self.img = img_bgr
        self.disp, self.s = _fit_to_screen(img_bgr)
        self.win = win
        self.pt0 = None
        self.pt1 = None
        self.drawing = False
        self.result = None

    def _on_mouse(self, ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt0 = (x, y)
            self.pt1 = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.pt1 = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.pt1 = (x, y)

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)
        while True:
            canvas = self.disp.copy()
            if self.pt0 and self.pt1:
                x0, y0 = self.pt0
                x1, y1 = self.pt1
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(
                    canvas,
                    "Drag to select ROI | Enter=OK, C=clear, Esc=cancel",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(self.win, canvas)
            k = cv2.waitKey(16) & 0xFF
            if k in (13, 10):  #Enter
                if self.pt0 and self.pt1:
                    x0, y0 = self.pt0
                    x1, y1 = self.pt1
                    x, y = min(x0, x1), min(y0, y1)
                    w, h = abs(x1 - x0), abs(y1 - y0)
                    if w >= 4 and h >= 4:
                        #画面座標 → 元画像座標へ逆スケール
                        sx = 1.0 / self.s
                        sy = 1.0 / self.s
                        self.result = (
                            int(round(x * sx)),
                            int(round(y * sy)),
                            int(round(w * sx)),
                            int(round(h * sy)),
                        )
                        break
            elif k in (ord("c"), ord("C")):
                self.pt0 = self.pt1 = None
            elif k == 27:  #Esc
                self.result = None
                break
        cv2.destroyWindow(self.win)
        return self.result


def select_detect_roi_interactive(frame_bgr, title="Select Detect ROI"): #呼び出すためだけのラッパ関数
    """frame_bgr: undistort/crop後の画像（検出に使う見え方そのまま）"""
    sel = _RoiSelector(frame_bgr, win=title)
    return sel.run()


def adjust_K_for_crop(K: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray: #ROIで切り取った後でもカメラ内部行列が破綻しないように補正する関数
    x, y, _, _ = roi
    K2 = K.copy()
    K2[0, 2] -= x #cxをROI原点に合わせるため補正
    K2[1, 2] -= y #cyをROI原点に合わせるため補正
    return K2


def undistort_with_maps(img: np.ndarray, map1, map2) -> np.ndarray: #キャリブレーション結果のmapを使って、元フレームを補正し、精度優先の補間で再構成する
    return cv2.remap(
        img, #入力フレーム(1枚)
        map1, #事前計算した歪み補正マップ
        map2, 
        interpolation=cv2.INTER_LANCZOS4, #補間方法
        borderMode=cv2.BORDER_CONSTANT, #画像の外を参照したらどうするか
    )  #精度優先ならINTER_LANCZOS4, 速度優先ならINTER_LINEAR


def get_aruco_dict(name: str):
    table = {
        "4X4_50": cv2.aruco.DICT_4X4_50,
        "4X4_100": cv2.aruco.DICT_4X4_100,
        "4X4_250": cv2.aruco.DICT_4X4_250,
        "4X4_1000": cv2.aruco.DICT_4X4_1000,
        "5X5_50": cv2.aruco.DICT_5X5_50,
        "5X5_100": cv2.aruco.DICT_5X5_100,
        "5X5_250": cv2.aruco.DICT_5X5_250,
        "5X5_1000": cv2.aruco.DICT_5X5_1000,
        "6X6_50": cv2.aruco.DICT_6X6_50,
        "6X6_250": cv2.aruco.DICT_6X6_250,
        "6X6_1000": cv2.aruco.DICT_6X6_1000,
    }
    key = name.strip().upper()
    if key not in table:
        raise ValueError(f"Unsupported ArUco dict: {name}")
    return cv2.aruco.getPredefinedDictionary(table[key])

def load_board_layout(yaml_path: str) -> Dict[int, Dict[str, float]]:
    """board.yml を読み込み。形式:
    markers:
      0: { cx: 0.00, cy: 0.00, size: 0.145 }
      1: { cx: 0.50, cy: 0.00, size: 0.145 }
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return {int(k): v for k, v in y["markers"].items()}

def board_corners_world(id_: int, layout: Dict[int, Dict[str, float]]) -> Optional[np.ndarray]:
    """ID に対応する 4隅(世界座標, z=0)を返す。順序: 左上→右上→右下→左下"""
    if id_ not in layout:
        return None
    cx, cy, size = float(layout[id_]["cx"]), float(layout[id_]["cy"]), float(layout[id_]["size"])
    s = size * 0.5
    return np.array([
        [cx - s, cy - s, 0.0],
        [cx + s, cy - s, 0.0],
        [cx + s, cy + s, 0.0],
        [cx - s, cy + s, 0.0],
    ], np.float32)

def detect_aruco(gray, aruco_dict):
    """OpenCV 4.7+ / 旧API 両対応"""
    try:
        det = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
    except Exception:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


def marker_area(corners) -> float:
    """4点コーナーの投影面積"""
    pts = corners.reshape(-1, 2).astype(np.float32)
    return float(cv2.contourArea(pts))


def pick_largest_marker(corners, ids):
    areas = [marker_area(c) for c in corners]
    idx = int(np.argmax(areas))
    return corners[idx].reshape(-1, 2), int(ids[idx][0]), areas[idx]


def corner_subpix_refine(gray, corners_list, win=(5, 5), iters=100, eps=1e-4) -> List[np.ndarray]:
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, eps)
    out = []
    for c in corners_list:
        pts = c.reshape(-1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(gray, pts, win, (-1, -1), term)
        out.append(pts.reshape(-1, 2))
    return out

def H_from_pose(newK: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """画像(undistorted,newK) → 平面[m] のH（スケール正規化）"""
    R, _ = cv2.Rodrigues(rvec)
    r1, r2 = R[:, 0], R[:, 1]
    G = newK @ np.column_stack([r1, r2, tvec.reshape(3)])  #3x3
    H = np.linalg.inv(G)
    return H / H[2, 2]


def project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.hstack([pts_xy, np.ones((pts_xy.shape[0], 1), np.float64)])
    out = (H @ pts.T).T
    return out[:, :2] / out[:, 2:3]

def rotation2d_deg(deg: float) -> np.ndarray:
    """平面[m]座標系の2D回転（左から掛ける用）"""
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=np.float64)

def autosize_from_H(H: np.ndarray, src_wh: Tuple[int,int], margin_px: int = 20):
    """Hで射影した四隅のバウンディングでキャンバスを作り直す（左掛け後の平行移動も含む）"""
    w, h = src_wh
    corners = np.array([[0,0], [w,0], [w,h], [0,h]], np.float64)
    dst = project_points(H, corners)  #(4,2)
    xmin, ymin = dst.min(axis=0); xmax, ymax = dst.max(axis=0)
    W = int(np.ceil(xmax - xmin)) + 2*margin_px
    Hh = int(np.ceil(ymax - ymin)) + 2*margin_px
    T = np.array([[1,0,-xmin+margin_px],
                  [0,1,-ymin+margin_px],
                  [0,0,1]], dtype=np.float64)
    return (T @ H), (max(64, W), max(64, Hh))

def compute_upright_angle_from_marker(
    frame_bgr: np.ndarray,
    map1, map2,                    #undistortマップ
    roi_crop: Optional[Tuple[int,int,int,int]],   #undistort後に使うROI（--crop-roi時のみ）
    detect_roi: Optional[Tuple[int,int,int,int]], #検出用ROI（undistort(+crop)後座標）
    aruco_dict,
    expected_id: int,
    H_m: np.ndarray               #いったん推定済みの平面H[m]
) -> Optional[float]:
    """単一フレームでマーカーを検出し、選んだ辺の角度を平面[m]上で測って水平化用角度を返す（度）。
       失敗時は None を返す。"""
    #1) undistort（+crop）
    img_ud = undistort_with_maps(frame_bgr, map1, map2)
    if roi_crop is not None:
        x0,y0,w0,h0 = roi_crop
        img_ud = img_ud[y0:y0+h0, x0:x0+w0]

    #2) 検出ビュー抽出
    det_view = img_ud
    offx = offy = 0
    if detect_roi is not None:
        dx,dy,dw,dh = detect_roi
        det_view = img_ud[dy:dy+dh, dx:dx+dw]
        offx, offy = dx, dy

    gray = cv2.cvtColor(det_view, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_aruco(gray, aruco_dict)
    if ids is None or len(ids) == 0:
        return None
    #expected_id でフィルタ
    if expected_id >= 0:
        mask = (ids.reshape(-1) == expected_id)
        if not np.any(mask):
            return None
        corners = [corners[i] for i,m in enumerate(mask) if m]
        ids = ids[mask]
    if len(corners) == 0:
        return None

    #一番面積が大きいのを採用
    pts_local, _id, _area = pick_largest_marker(corners, ids)
    pts = pts_local + np.array([offx, offy], np.float32)   #元（undistort(+crop)後）座標へ

    #サブピクセルで軽く締める（安定性UP）
    pts_ref = corner_subpix_refine(cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY), [pts])[0]

    #コーナ順序: OpenCVは通常 [TL(0), TR(1), BR(2), BL(3)]（時計回り）
    return pts_ref  #呼び出し側で辺を選んで角度を出す


def make_canvas_and_H_px(H_m: np.ndarray, img_wh: Tuple[int, int], px_per_m: float, margin_m=0.2):
    w, h = img_wh
    img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    plane_corners_m = project_points(H_m, img_corners)

    xmin, ymin = plane_corners_m.min(axis=0) - margin_m
    xmax, ymax = plane_corners_m.max(axis=0) + margin_m

    W = max(64, int(np.ceil((xmax - xmin) * px_per_m)))
    H = max(64, int(np.ceil((ymax - ymin) * px_per_m)))

    S = np.array([
        [px_per_m, 0, -xmin * px_per_m],
        [0, px_per_m, -ymin * px_per_m],
        [0, 0, 1]
    ], np.float64)

    H_px = S @ H_m
    return H_px, (W, H)

#========== HQ H推定（複数フレーム, サブピクセル, PnP, 姿勢平均） ==========
def estimate_H_from_video_hq(
    walk_path: Path,
    map1,
    map2,
    newK,
    aruco_dict,
    marker_size_m: float,
    scan_step=1,
    scan_max=800,
    candidates_top=10,
    reproj_thresh_px=0.8,
    debug_dir: Optional[Path] = None,
    csv_log: Optional[Path] = None,
    save_best_k: int = 0,   #★ 追加：良フレームの保存枚数（0で保存しない）
    roi: Optional[Tuple[int, int, int, int]] = None,
    detect_roi: Optional[Tuple[int, int, int, int]] = None,  #★ 追加
    expected_id: int = -1,   #★ 追加
    board_layout: Optional[Dict[int, Dict[str, float]]] = None,
    progress_interval_sec: float = 2.0,   # [ADD] 進捗ログ間隔
) -> np.ndarray:

    cap = cv2.VideoCapture(str(walk_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open: {walk_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    detections = []
    scanned, idx = 0, 0

    #1) スキャンして候補を集める（画像保存はしない）
    # [ADD] スキャン進捗ticker
    scan_total = min(int(scan_max), int(total)) if scan_max > 0 and total > 0 else int(total)
    scan_progress = _ProgressTicker(total=scan_total if scan_total > 0 else 1,
                                    interval_sec=2.0)  # 間隔は後で上書き可能
    scan_progress.interval = float(progress_interval_sec)  # [ADD]


    while scanned < scan_max and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        img_ud = undistort_with_maps(frame, map1, map2)

        if roi is not None:
            x0, y0, w_roi, h_roi = roi
            img_ud = img_ud[y0:y0+h_roi, x0:x0+w_roi]

        #--- 検出用ROI（さらに絞る）。座標は img_ud 基準 ---
        det_view = img_ud
        offx = offy = 0
        if detect_roi is not None:
            dx, dy, dw, dh = detect_roi
            det_view = img_ud[dy:dy+dh, dx:dx+dw]
            offx, offy = dx, dy

        gray_full = cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY)   #サブピクセルは元座標で
        gray = cv2.cvtColor(det_view, cv2.COLOR_BGR2GRAY)      #検出は det_view

        corners, ids = detect_aruco(gray, aruco_dict)
        if ids is None or len(ids) == 0:
            scanned += 1
            idx += scan_step
            # [ADD] 定期ログ
            msg = scan_progress.tick(done=min(scanned, scan_total), prefix="[scan]")
            if msg:
                print(msg)
            continue

        #期待IDでフィルタ（>=0 のときのみ採用）
        if expected_id >= 0:
            mask = (ids.reshape(-1) == expected_id)
            if not np.any(mask):
                scanned += 1
                idx += scan_step
                continue
            corners = [corners[i] for i, m in enumerate(mask) if m]
            ids = ids[mask]

        pts_local, _id, area = pick_largest_marker(corners, ids)
        #ROI切り出し分を元座標へ戻す
        pts = pts_local + np.array([offx, offy], np.float32)

        #小さすぎる四角は除外
        side_lens = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        if float(np.min(side_lens)) < 40.0:
            scanned += 1
            idx += scan_step
            continue

        detections.append({
            "idx": idx,
            "area": area,
            "pts": pts,
            "gray": gray_full
        })

        scanned += 1
        idx += scan_step

    if len(detections) == 0:
        cap.release()
        raise RuntimeError(f"ArUco not found in {walk_path}")

    #2) 面積上位を選抜
    detections.sort(key=lambda d: d["area"], reverse=True)
    cand = detections[:min(candidates_top, len(detections))]

    #3) PnP → reprojection誤差
    objp = np.array([
        [0, 0, 0],
        [marker_size_m, 0, 0],
        [marker_size_m, marker_size_m, 0],
        [0, marker_size_m, 0]
    ], np.float32)

    refined = []
    for d in cand:
        gray_full = d["gray"]

    #再検出（undistort後の同じ絵で）
        corners, ids = detect_aruco(gray_full, aruco_dict)
        if ids is None or len(ids) == 0:
            continue

        imgp_list, objp_list = [], []
        for i, idv in enumerate(ids.reshape(-1)):
            pts = corners[i].reshape(-1,2).astype(np.float32)

            # 小さすぎるマーカー除外
            side_lens = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
            if float(np.min(side_lens)) < 40.0:
                continue

            # サブピクセルで締める
            pts_ref = corner_subpix_refine(gray_full, [pts], win=(9,9), iters=300, eps=1e-6)[0].astype(np.float32)
            imgp_list.append(pts_ref.reshape(-1,1,2))

            # ★ board_layout を使う（m単位）
            if board_layout is not None:
                wc = board_corners_world(int(idv), board_layout)  # (4,3) or None
                if wc is None:
                    wc = np.array([
                        [0,0,0],
                        [marker_size_m,0,0],
                        [marker_size_m,marker_size_m,0],
                        [0,marker_size_m,0]
                        ], np.float32)
            else:
                wc = np.array([
                    [0,0,0],
                    [marker_size_m,0,0],
                    [marker_size_m,marker_size_m,0],
                    [0,marker_size_m,0]
                ], np.float32)

            objp_list.append(wc.reshape(-1,1,3))

        if len(imgp_list) == 0:
            continue

        imgp = np.concatenate(imgp_list, axis=0)  # Nx1x2
        objp = np.concatenate(objp_list, axis=0)  # Nx1x3


        ok = False; rvec=None; tvec=None; err=1e9

        if len(imgp_list) == 1:
        #1枚のときは IPPE_SQUARE をまず試す（平面正方形に強い）
            try:
                ok1, r1, t1 = cv2.solvePnP(objp, imgp, newK, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok1:
                    reproj, _ = cv2.projectPoints(objp, r1, t1, newK, None)
                    err1 = float(np.sqrt(np.mean(np.sum((reproj.reshape(-1,2)-imgp.reshape(-1,2))**2, axis=1))))
                    ok, rvec, tvec, err = True, r1, t1, err1
            except Exception:
                pass

        #IPPEが無い/誤差大 の場合や、マーカーが複数のときは ITERATIVE(+LM)
        if (not ok) or (err > reproj_thresh_px):
            ok2, r2, t2 = cv2.solvePnP(objp, imgp, newK, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2:
                try:
                    r2, t2 = cv2.solvePnPRefineLM(objp, imgp, newK, None, r2, t2)
                except Exception:
                    pass
                reproj2, _ = cv2.projectPoints(objp, r2, t2, newK, None)
                err2 = float(np.sqrt(np.mean(np.sum((reproj2.reshape(-1,2)-imgp.reshape(-1,2))**2, axis=1))))
                if not ok or err2 < err:
                    ok, rvec, tvec, err = True, r2, t2, err2

            if ok and err <= reproj_thresh_px:
                d2 = dict(d)  #もとの検出辞書をコピー
    #万一 d に gray が無ければ、現在の gray_full を入れておく（保険）
                if "gray" not in d2:
                    d2["gray"] = gray_full

                d2.update({
                    "rvec": rvec.reshape(3),
                    "tvec": tvec.reshape(3),
                    "err": err,
                    "imgp": pts_ref
                })
                refined.append(d2)



    #CSV
    if csv_log:
        import csv
        with open(csv_log, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "area", "reproj_err_px"])
            for d in refined:
                w.writerow([d["idx"], f"{d['area']:.3f}", f"{d['err']:.4f}"])

    if len(refined) == 0:
        cap.release()
        raise RuntimeError(
            f"robust candidates not found (all reproj_err > {reproj_thresh_px}px): {walk_path}"
        )

    #4) 重み付き姿勢平均
    eps = 1e-9
    weights = np.array([d["area"] / (d["err"] ** 2 + eps) for d in refined], dtype=np.float64)
    Wsum = float(np.sum(weights))
    weights = weights / Wsum

    R_acc = np.zeros((3, 3), np.float64)
    t_acc = np.zeros(3, np.float64)

    for d, w in zip(refined, weights):
        R, _ = cv2.Rodrigues(d["rvec"])
        R_acc += w * R
        t_acc += w * d["tvec"]

    U, S, Vt = np.linalg.svd(R_acc)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    t_avg = t_acc

    #H
    r1, r2 = R_avg[:, 0], R_avg[:, 1]
    G = newK @ np.column_stack([r1, r2, t_avg])
    H_m = np.linalg.inv(G)
    H_m = H_m / H_m[2, 2]

    #★ 良フレームだけ保存（オプション）
    if debug_dir and save_best_k > 0:
        best_dir = debug_dir / f"{walk_path.stem}_best"
        best_dir.mkdir(parents=True, exist_ok=True)
        #上位wのフレームidxでソート
        order = np.argsort(weights)[::-1][:save_best_k]
        for rank, i in enumerate(order, start=1):
            d = refined[i]
            vis = cv2.cvtColor(d["gray"], cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, [d["imgp"].astype(int)], True, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"rank={rank} idx={d['idx']} area={d['area']:.1f} err={d['err']:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(best_dir / f"best_{rank:02d}_idx{d['idx']:06d}.png"), vis)

    cap.release()
    return H_m

#========== 1カメラ処理（HQ） ==========
def process_one_pair_hq(
    yaml_path: Path,
    walk_path: Path,
    out_path: Path,
    aruco_dict_name: str,
    marker_size_cm: float,
    px_per_m: float,
    alpha: float,
    scan_step: int,
    scan_max: int,
    candidates_top: int,
    reproj_thresh_px: float,
    superres: float,
    debug_dir: Optional[Path],
    save_csv: bool,
    quiet: bool = False,
    save_best_k: int = 0,
    fisheye: bool = False,
    calib_size_str: str = "",       #★ 追加
    crop_roi: bool = False,         #★ 追加
    detect_roi_str: str = "",       #★ 追加
    select_detect_roi: bool = False,#★ 追加
    select_roi_frame: int = 0,      #★ 追加
    expected_id: int = -1,           #★ 追加
    board_layout: Optional[Dict[int, Dict[str, float]]] = None, #★ 追加
    auto_upright: bool = False,
    upright_edge: str = "01",
    *,
    solve: str = "pnp",            # [ADD]
    warp_impl: str = "opencv",
    progress_interval_sec: float = 2.0,   # [ADD]
) -> str:

    K, D = load_cam_yaml(str(yaml_path))
    cap0 = cv2.VideoCapture(str(walk_path))
    if not cap0.isOpened():
        raise FileNotFoundError(f"cannot open: {walk_path}")

    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    cap0.release()

    #キャリブ時の画像サイズが指定されていれば、Kを動画サイズへスケール
    if calib_size_str:
        calib_wh = parse_wh(calib_size_str)  #例: "1920x1080"
        K = scale_intrinsics(K, calib_wh, (w, h))

    #--- ここまで: K,D 読み込み & 動画の w,h,fps 取得 & Kスケール（必要なら） ---
    if fisheye:
        map1, map2, newK, roi = build_undistort_maps_fisheye(K, D, (w, h))
    else:
        map1, map2, newK, roi = build_undistort_maps_hq(K, D, (w, h), alpha=alpha)

    aruco_dict = get_aruco_dict(aruco_dict_name)
    marker_size_m = marker_size_cm / 100.0

    #---- ここから“必ず” newK_use を決める（ガード付き）----
    if roi is None:
        roi = (0, 0, w, h)  #ROIがNoneの場合にも備える

    use_roi = bool(crop_roi and roi[2] > 0 and roi[3] > 0)

    #まず newK_use を newK で初期化（最後の保険）
    newK_use = newK
    img_wh_used = (w, h)

    if use_roi:
        #主点を ROI 原点に合わせる
        newK_use = adjust_K_for_crop(newK, roi)
        img_wh_used = (roi[2], roi[3])

    #--- ROI/主点補正 決定済み: use_roi, newK_use, img_wh_used ---

    #1) 文字列で指定があればそれを使う
    detect_roi = parse_roi(detect_roi_str) if detect_roi_str else None

    #2) GUIで選ぶ指定なら、フレームを1枚読み出して undistort(+crop) 後の絵で選択
    if detect_roi is None and select_detect_roi:
        cap_tmp = cv2.VideoCapture(str(walk_path))
        total_tmp = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx_sel = int(np.clip(select_roi_frame, 0, max(total_tmp - 1, 0)))
        cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, idx_sel)
        ok, f0 = cap_tmp.read()
        cap_tmp.release()
        if not ok:
            raise RuntimeError("ROI選択用フレームを取得できませんでした")

        f0_ud = undistort_with_maps(f0, map1, map2)
        if use_roi:
            x0, y0, w_roi, h_roi = roi
            f0_ud = f0_ud[y0:y0+h_roi, x0:x0+w_roi]

        det = select_detect_roi_interactive(f0_ud, title="Select Detect ROI")
        if det is None:
            raise RuntimeError("ROI選択がキャンセルされました")
        detect_roi = det

        if debug_dir:
            (debug_dir / "roi").mkdir(parents=True, exist_ok=True)
            with open(debug_dir / f"{walk_path.stem}_detect_roi.txt", "w", encoding="utf-8") as f:
                x, y, w_, h_ = detect_roi
                f.write(f"{x},{y},{w_},{h_}\n")

    #H推定（良フレームだけ保存したい場合は save_best_k を渡す）
    if solve == "4pts":
        # --- [4点法] 代表フレーム1枚で H を直接解く ---
        cap_tmp = cv2.VideoCapture(str(walk_path))
        total_tmp = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx_sel = int(np.clip(select_roi_frame, 0, max(total_tmp - 1, 0)))
        cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, idx_sel)
        ok, f0 = cap_tmp.read()
        cap_tmp.release()
        if not ok:
            raise RuntimeError("4pts用のフレーム取得に失敗")

        img_ud0 = undistort_with_maps(f0, map1, map2)
        if use_roi:
            x0, y0, w_roi, h_roi = roi
            img_ud0 = img_ud0[y0:y0+h_roi, x0:x0+w_roi]

        # 検出ROI適用
        det_view = img_ud0
        offx = offy = 0
        if detect_roi is not None:
            dx, dy, dw, dh = detect_roi
            det_view = img_ud0[dy:dy+dh, dx:dx+dw]
            offx, offy = dx, dy

        gray = cv2.cvtColor(det_view, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_aruco(gray, aruco_dict)
        if ids is None or len(ids) == 0:
            raise RuntimeError("4pts: ArUco未検出")

        # expected_id のみ使う（指定がなければ最大面積）
        if expected_id >= 0:
            mask = (ids.reshape(-1) == expected_id)
            if not np.any(mask):
                raise RuntimeError(f"4pts: expected_id={expected_id} が見つからない")
            corners = [corners[i] for i, m in enumerate(mask) if m]
            ids = ids[mask]

        if len(corners) > 1:
            # 複数検出時は最大面積
            areas = [marker_area(c) for c in corners]
            k = int(np.argmax(areas))
            corners = [corners[k]]
            ids = ids[[k]]

        pts_local = corners[0].reshape(-1, 2).astype(np.float32)
        pts_img = pts_local + np.array([offx, offy], np.float32)

        # サブピクセル
        pts_img = corner_subpix_refine(cv2.cvtColor(img_ud0, cv2.COLOR_BGR2GRAY),
                                       [pts_img], win=(9,9), iters=300, eps=1e-6)[0]

        # 世界側4点（m単位, z=0）
        idv = int(ids.reshape(-1)[0])
        if board_layout is not None:
            W4 = board_corners_world(idv, board_layout)  # (4,3)
            if W4 is None:
                raise RuntimeError(f"4pts: board-layoutにID {idv} が無い")
            dst_m = W4[:, :2].astype(np.float64)  # (4,2) in [m]
        else:
            s = marker_size_m
            dst_m = np.array([[0,0],[s,0],[s,s],[0,s]], np.float64)

        # H_m: 画像座標(undistort(+crop)後) → 平面[m]
        H_m = homography_from_4pts(pts_img.astype(np.float64), dst_m.astype(np.float64))
    else:
        # --- [従来PnPルート] ---
        csv_log = (debug_dir / f"{walk_path.stem}_candidates.csv") if (debug_dir and save_csv) else None
        H_m = estimate_H_from_video_hq(
            walk_path,
            map1, map2, newK_use, aruco_dict, marker_size_m,
            scan_step=scan_step, scan_max=scan_max,
            candidates_top=candidates_top,
            reproj_thresh_px=reproj_thresh_px,
            debug_dir=debug_dir,
            csv_log=csv_log,
            save_best_k=save_best_k,
            roi=roi if use_roi else None,
            detect_roi=detect_roi,
            expected_id=expected_id,
            board_layout=board_layout,
            progress_interval_sec=progress_interval_sec,   # [ADD]

        )


    if auto_upright:
        #ROI選択用と同じフレーム番号を使う（なければ0）
        cap_tmp = cv2.VideoCapture(str(walk_path))
        total_tmp = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx_sel = int(np.clip(select_roi_frame, 0, max(total_tmp - 1, 0)))
        cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, idx_sel)
        ok, f0 = cap_tmp.read()
        cap_tmp.release()

        if ok:
            pts_ref = compute_upright_angle_from_marker(
                f0, map1, map2,
                roi if use_roi else None,
                detect_roi,
                aruco_dict,
                expected_id,
                H_m
            )
            if pts_ref is not None:
                #辺の選択（例: "01" なら TL->TR）
                edge = upright_edge
                idx0, idx1 = int(edge[0]), int(edge[1])
                p0, p1 = pts_ref[idx0], pts_ref[idx1]

                #画像座標 → 平面[m] へ投影
                P = project_points(H_m, np.vstack([p0, p1]).astype(np.float64))
                v = P[1] - P[0]
                ang_deg = -np.degrees(np.arctan2(v[1], v[0]))  #水平化のため逆回転

                #回転（平面[m]側の基準回転を左掛け）
                R2 = rotation2d_deg(ang_deg)
                H_m = R2 @ H_m

                if not quiet:
                    print(f"[upright] edge={edge}, angle={ang_deg:.3f} deg  -> 平面水平化適用")
            else:
                if not quiet:
                    print("[upright] マーカー検出失敗：自動水平化スキップ")
        else:
            if not quiet:
                print("[upright] フレーム取得失敗：自動水平化スキップ")

    #キャンバスとH_px（高解像ワークスペース）
    px_per_m_eff = px_per_m * max(1.0, float(superres))
    H_px, (W_hr, H_hr) = make_canvas_and_H_px(H_m, img_wh_used, px_per_m_eff)

    #★ 最終出力サイズ（superres>1.0ならダウンサンプルしたサイズ）
    if superres > 1.0:
        W_out = int(round(W_hr / superres))
        H_out = int(round(H_hr / superres))
    else:
        W_out, H_out = int(W_hr), int(H_hr)

    #★ H.264系は偶数サイズが安全
    if W_out % 2:
        W_out += 1
    if H_out % 2:
        H_out += 1

    #★ Writerを堅牢に開く（コーデック自動フォールバック）
    out_path = Path(out_path)  #念のためPath化
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, used_path, used_fourcc = open_video_writer_with_fallback(out_path, fps, (W_out, H_out))
    # [ADD] 出力フレーム進捗ticker
    cap = cv2.VideoCapture(str(walk_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress = _ProgressTicker(
        total=total_frames if total_frames > 0 else 1,
        interval_sec=float(progress_interval_sec)
    )


    #★ フレーム処理（warpはHRで→必要ならAREAでダウンサンプル→writerに書く）
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        img_ud = undistort_with_maps(frame, map1, map2)
        if use_roi:
            x, y, w_roi, h_roi = roi
            img_ud = img_ud[y:y+h_roi, x:x+w_roi]

        if warp_impl == "opencv":
            warped_hr = cv2.warpPerspective(
                img_ud, H_px, (W_hr, H_hr),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT
            )
        else:
            # NumPy版（双線形補間）
            warped_hr = warp_by_homography_numpy(
                img_ud, H_px, out_wh=(W_hr, H_hr), border_value=0
            )


        if (W_hr, H_hr) != (W_out, H_out):
            warped = cv2.resize(warped_hr, (W_out, H_out), interpolation=cv2.INTER_AREA)
        else:
            warped = warped_hr

        writer.write(warped)
        i += 1       
        # [ADD] 定期ログ（warp書き出し）
        msg = progress.tick(done=i, prefix="[warp]")
        if msg:
            print(msg)

    writer.release()
    cap.release()

    if not quiet:
        print(f"[Writer] fourcc={used_fourcc}, file={used_path.name}, size={W_out}x{H_out}, frames={i}")

    return f"OK[HQ]: {yaml_path.name} -> {Path(used_path).name} ({i} frames)"

#========== CLI ==========
def parse_pairs(pairs):
    out = []
    for p in pairs:
        s = p.strip().strip('"').strip("'")
        if "|" in s:
            parts = s.split("|")
            if len(parts) != 3:
                raise ValueError(f"--pair は 'yaml:walk:out' or 'yaml|walk|out' を指定: {p}")
            y, v, o = parts
        else:
            try:
                y, v, o = s.rsplit(":", 2)  #Windows D:\ を壊さない
            except ValueError:
                raise ValueError(f"--pair は 'yaml:walk:out' 形式で指定: {p}")
        out.append((Path(y), Path(v), Path(o)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="HQ版: 各カメラを実世界平面へ高精度warp（合成なし）"
    )
    ap.add_argument("--pair", action="append", required=True,
                    help="yaml:walk:out の組（複数可）例: cam1.yaml:cam1.mp4:cam1_plane.mp4")
    ap.add_argument("--dict", default="4X4_50",
                    help="ArUco辞書（例: 4X4_50, 5X5_1000 等）")
    ap.add_argument("--marker-size-cm", type=float, default=14.5,
                    help="ArUco一辺の物理サイズ[cm]")
    ap.add_argument("--px-per-m", type=float, default=400.0,
                    help="仮想平面スケール[px/m]")
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="getOptimalNewCameraMatrixのalpha(0~1)")
    ap.add_argument("--scan-step", type=int, default=1,
                    help="検出スキャン間隔（フレーム）")
    ap.add_argument("--scan-max", type=int, default=800,
                    help="スキャン最大数（小さすぎると見逃し）")
    ap.add_argument("--candidates-top", type=int, default=10,
                    help="面積上位の候補フレーム数")
    ap.add_argument("--reproj-thresh-px", type=float, default=0.8,
                    help="再投影誤差の外れ閾値[px]")
    ap.add_argument("--superres", type=float, default=1.0,
                    help="スーパーサンプリング倍率(>=1.0)")
    ap.add_argument("--debug-dir", default="",
                    help="デバッグ出力先(検出画像/CSV/メタyaml)")
    ap.add_argument("--save-csv", action="store_true",
                    help="候補フレームのCSVを保存")
    ap.add_argument("--workers", type=int, default=0,
                    help="並列数（0/1なら逐次、2以上で並列）")
    ap.add_argument("--quiet", action="store_true",
                    help="ログを最小化（OpenCVログも抑制）")
    ap.add_argument("--save-best-k", type=int, default=0,
                    help="良フレームの保存枚数（0=保存しない）")
    ap.add_argument("--fisheye", action="store_true",
                    help="fisheyeモデルで補正する")
    ap.add_argument("--calib-size", default="",
                    help="キャリブ時の画像サイズ (例 1920x1080)。指定時はKをスケール")
    ap.add_argument("--crop-roi", action="store_true",
                    help="undistort後にROIでクロップして使用（端を使わない）")
    ap.add_argument("--select-detect-roi", action="store_true",
                    help="GUIで検出用ROI（四角）をマウス選択してから実行する")
    ap.add_argument("--select-roi-frame", type=int, default=0,
                    help="ROI選択に使うフレーム番号（0=先頭）")
    ap.add_argument("--expected-id", type=int, default=-1,
                    help="期待するArUco ID（指定>=0ならそのIDのみ採用）")
    ap.add_argument("--detect-roi", default="",
                    help="検出用ROIを数値で指定する場合の x,y,w,h（undistort/crop後の座標系）")
    ap.add_argument("--auto-upright", action="store_true",
                    help="マーカーの指定辺を水平に回す（出力平面の自動水平化）")
    ap.add_argument("--upright-edge", default="01", choices=["01", "12", "23", "30"],
                    help="水平化に使うマーカー辺（TL=0,TR=1,BR=2,BL=3）。例: 01=上辺、12=右辺")
    ap.add_argument("--board-layout", default="",
                    help="共通平面座標系のレイアウトYAML（cx,cy,size[m]）")
    ap.add_argument("--solve", choices=["pnp", "4pts"], default="pnp",
                    help="Hの推定方法: pnp(既定) or 4pts(対応点で直接解く)")
    ap.add_argument("--warp-impl", choices=["opencv", "numpy"], default="opencv",
                    help="warp実装: OpenCV(cv2.warpPerspective) or NumPy(双線形補間)")
    # [ADD] 進捗ログの更新間隔（秒）
    ap.add_argument("--progress-interval-sec", type=float, default=2.0,
                    help="進捗ログの出力間隔[秒]（デフォルト: 2.0）")





    args = ap.parse_args()
    set_opencv_quiet(args.quiet)

    layout = load_board_layout(args.board_layout) if args.board_layout else None

    pairs = parse_pairs(args.pair)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    jobs = []
    if args.workers and args.workers > 1:
        #並列処理
        with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for (y, v, o) in pairs:
                jobs.append(ex.submit(
                    process_one_pair_hq,
                    y, v, o,
                    args.dict, args.marker_size_cm, args.px_per_m, args.alpha,
                    args.scan_step, args.scan_max, args.candidates_top,
                    args.reproj_thresh_px, args.superres, debug_dir,
                    args.save_csv, quiet=args.quiet, save_best_k=args.save_best_k,
                    fisheye=args.fisheye, calib_size_str=args.calib_size,
                    crop_roi=args.crop_roi, detect_roi_str=args.detect_roi,
                    select_detect_roi=args.select_detect_roi,
                    select_roi_frame=args.select_roi_frame,
                    expected_id=args.expected_id,
                    auto_upright=args.auto_upright,
                    upright_edge=args.upright_edge,
                    board_layout=layout,
                    solve=args.solve,                
                    warp_impl=args.warp_impl,
                    progress_interval_sec=args.progress_interval_sec,  # [ADD]
  
                ))
            for j in futures.as_completed(jobs):
                print(j.result())
    else:
        #シングル処理
        for (y, v, o) in pairs:
            msg = process_one_pair_hq(
                y, v, o,
                args.dict, args.marker_size_cm, args.px_per_m, args.alpha,
                args.scan_step, args.scan_max, args.candidates_top,
                args.reproj_thresh_px, args.superres, debug_dir,
                args.save_csv, quiet=args.quiet, save_best_k=args.save_best_k,
                fisheye=args.fisheye, calib_size_str=args.calib_size,
                crop_roi=args.crop_roi, detect_roi_str=args.detect_roi,
                select_detect_roi=args.select_detect_roi,
                expected_id=args.expected_id,
                auto_upright=args.auto_upright,
                upright_edge=args.upright_edge,
                board_layout=layout,
                solve=args.solve,                 
                warp_impl=args.warp_impl,
                progress_interval_sec=args.progress_interval_sec,  # [ADD]  
                
            )
            print(msg)

if __name__ == "__main__":
    main()
