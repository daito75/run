#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#アフィン変換．akazeにより，特徴点どうしをマッチング．その後，合成

import os, sys
import cv2
import numpy as np

# ========= 設定 =========
IMG1     = r"D:\BRLAB\2025\affin\s_test1.jpeg"     # 左側（基準）
IMG2     = r"D:\BRLAB\2025\affin\s_test2.jpeg"     # 右側（ワープして重ねる）
OUT_DIR  = r"D:\BRLAB\2025\affin\out_seam"
DETECTOR = "AKAZE"  # "AKAZE" or "ORB" or "SIFT"
RATIO    = 0.8      # Lowe比率テスト
RANSAC_REPROJ = 3.0
FEATHER_PX = 25     # シームの両側のフェザー幅（ピクセル）

# どちらか選ぶ
USE_MANUAL_SEAM = False   # True=手動（クリックで折れ線）；False=自動（動的計画法）

os.makedirs(OUT_DIR, exist_ok=True)

# ========= ユーティリティ =========
def must_read(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img

def get_detector(name):
    name = name.upper()
    if name == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if name == "ORB":
        return cv2.ORB_create(5000), cv2.NORM_HAMMING
    if name == "SIFT":
        return cv2.SIFT_create(), cv2.NORM_L2
    raise ValueError("unknown detector")

def compute_homography(img1, img2):
    det, norm = get_detector(DETECTOR)
    g1, g2 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (img1, img2)]
    kp1, des1 = det.detectAndCompute(g1, None)
    kp2, des2 = det.detectAndCompute(g2, None)
    if des1 is None or des2 is None:
        raise RuntimeError("特徴が見つかりません")

    bf = cv2.BFMatcher(norm, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m,n in knn if m.distance < RATIO*n.distance]
    if len(good) < 4:
        raise RuntimeError(f"マッチ不足: {len(good)}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, RANSAC_REPROJ)  # IMG2→IMG1 へ
    if H is None:
        raise RuntimeError("H推定失敗")
    return H

def make_union_canvas(img1, img2, H21):
    """img2をH21でimg1座標へ。両者が全部入るキャンバスを作り、平行移動で正領域へ。"""
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    # img2の四隅をimg1座標へ
    c2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    c2w = cv2.perspectiveTransform(c2, H21).reshape(-1,2)
    c1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]])

    all_pts = np.vstack([c1, c2w])
    xmin, ymin = np.floor(all_pts.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(all_pts.max(axis=0)).astype(int)

    tx, ty = -min(0, xmin), -min(0, ymin)  # 平行移動量
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
    W = int(xmax + tx); H = int(ymax + ty)

    canvas = np.zeros((H, W, 3), np.uint8)
    # img1配置
    canvas[ty:ty+h1, tx:tx+w1] = img1
    # img2ワープ
    img2w = cv2.warpPerspective(img2, T @ H21, (W, H))
    return canvas, img2w, (tx,ty)

def overlap_masks(imgL, imgR):
    """どっちが有効画素かのマスク（>0）"""
    maskL = (imgL.sum(axis=2) > 0).astype(np.uint8)
    maskR = (imgR.sum(axis=2) > 0).astype(np.uint8)
    overlap = (maskL & maskR).astype(np.uint8)
    return maskL, maskR, overlap

def build_manual_seam(width, height):
    """クリックで上→下の折れ線を取り、行ごとにシームx(y)を補間して返す。"""
    pts = []
    viz = np.zeros((height, width, 3), np.uint8)
    disp = "Click top→bottom for seam. ENTER=finish, r=reset, ESC=cancel"
    cv2.namedWindow("seam")
    cv2.imshow("seam", viz)

    def on_mouse(e,x,y,flags,userdata):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))

    cv2.setMouseCallback("seam", on_mouse)
    while True:
        img = viz.copy()
        for i,p in enumerate(pts):
            cv2.circle(img, p, 3, (0,255,255), -1, cv2.LINE_AA)
            if i>0:
                cv2.line(img, pts[i-1], pts[i], (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(img, disp, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
        cv2.imshow("seam", img)
        k = cv2.waitKey(30) & 0xFF
        if k == 13:  # Enter
            break
        if k in (27,):  # ESC
            cv2.destroyWindow("seam")
            return None
        if k in (ord('r'), ord('R')):
            pts.clear()
    cv2.destroyWindow("seam")
    if len(pts) < 2:
        return None
    # yでソート → 連続区間の直線補間
    pts = np.array(sorted(pts, key=lambda t: t[1]), dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    seam_x = np.interp(y, pts[:,1], pts[:,0], left=pts[0,0], right=pts[-1,0])
    return seam_x  # 長さ=H

def dp_vertical_seam(cost):
    """上→下へ最小コスト縦経路を動的計画法で取得。cost:H×W"""
    H, W = cost.shape
    dp = cost.copy().astype(np.float32)
    back = np.zeros((H,W), np.int16)  # -1,0,+1

    for i in range(1, H):
        for j in range(W):
            j0 = max(0, j-1); j1 = j; j2 = min(W-1, j+1)
            candidates = [dp[i-1, j0], dp[i-1, j1], dp[i-1, j2]]
            k = int(np.argmin(candidates))
            dp[i, j] += candidates[k]
            back[i, j] = (-1, 0, +1)[k] + (j0!=j-1) * 0  # 実質 -1/0/+1

    # 下端で最小の列から逆走
    j = int(np.argmin(dp[-1]))
    seam = np.zeros(H, np.int32)
    seam[-1] = j
    for i in range(H-2, -1, -1):
        j = seam[i+1] + back[i+1, j]
        j = max(0, min(W-1, j))
        seam[i] = j
    return seam  # 長さ=H、各行のx

def alpha_from_seam(seam_x, width, feather=25, prefer_right=True):
    """行ごとにシームxから左右を分け、フェザーで0..1のアルファを作る（1=右画像）。"""
    H = len(seam_x)
    X = np.arange(width, dtype=np.float32)[None, :]  # (1,W)
    S = seam_x[:, None].astype(np.float32)           # (H,1)
    dist = X - S  # >0 なら右側
    # フェザー：中心0、±featherで 0/1 へスムーズ
    a = 0.5*(1.0 + np.clip(dist/float(max(1,feather)), -1.0, 1.0))
    # prefer_right=True なら右(>シーム)を右画像に
    alpha = a if prefer_right else 1.0 - a
    return alpha.astype(np.float32)  # (H,W)

def blend_with_alpha(left, right, alpha01):
    if alpha01.ndim == 2:
        alpha01 = alpha01[..., None]
    L = left.astype(np.float32)
    R = right.astype(np.float32)
    out = L*(1.0-alpha01) + R*alpha01
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    img1 = must_read(IMG1)
    img2 = must_read(IMG2)

    # H: IMG2 -> IMG1
    H21 = compute_homography(img1, img2)

    # 横に広いキャンバスを作って両方配置
    canvasL, img2w, (tx,ty) = make_union_canvas(img1, img2, H21)

    # 左画像（キャンバス上）
    Hc, Wc = canvasL.shape[:2]
    # 右画像（同サイズ）をキャンバス上にもってくる
    canvasR = img2w

    # オーバーラップ領域抽出
    maskL, maskR, overlap = overlap_masks(canvasL, canvasR)
    if overlap.sum() == 0:
        # 重なりがないなら単純に重ねるだけ
        out = np.maximum(canvasL, canvasR)
        cv2.imwrite(os.path.join(OUT_DIR, "no_overlap_stitched.jpg"), out)
        print("[INFO] overlapなし。no_overlap_stitched.jpg を出力")
        return

    # 手動/自動で seam_x を作る
    if USE_MANUAL_SEAM:
        # オーバーラップの“見やすい表示”を作る
        vis = (0.5*canvasL + 0.5*canvasR).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, "preview_overlap.jpg"), vis)
        print("[INFO] 画面に 'preview_overlap.jpg' が表示されます。上→下へ数点クリック、Enterで決定。")
        seam_x = build_manual_seam(Wc, Hc)
        if seam_x is None:
            print("[WARN] 手動シームが未確定のため終了")
            return
    else:
        # 自動：オーバーラップ内の差分コストから縦シーム
        # コストは輝度差・彩度差など組み合わせてもOK。ここではL1/L2の混合。
        diff = cv2.absdiff(canvasL, canvasR).astype(np.float32)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # オーバーラップ外は大コストにする
        big = gray.max() + 1.0
        cost = np.where(overlap==1, gray, big).astype(np.float32)
        seam_x = dp_vertical_seam(cost)

        # 可視化
        seam_vis = (0.5*canvasL + 0.5*canvasR).astype(np.uint8)
        for y,x in enumerate(seam_x):
            cv2.circle(seam_vis, (int(x), int(y)), 1, (0,0,255), -1)
        cv2.imwrite(os.path.join(OUT_DIR, "auto_seam_preview.jpg"), seam_vis)

    # seam_x からアルファを作成（右画像優先でブレンド）
    alpha = alpha_from_seam(seam_x, Wc, FEATHER_PX, prefer_right=True)
    # オーバーラップ外は hard 決定：Lのみ or Rのみ
    alpha_full = alpha.copy()
    alpha_full[maskL==1 & (maskR==0)] = 0.0
    alpha_full[(maskL==0) & (maskR==1)] = 1.0
    # （※この2行は冗長なので実質 alpha で十分。厳密にやるなら条件分岐合成でもOK）

    blended = blend_with_alpha(canvasL, canvasR, alpha)
    cv2.imwrite(os.path.join(OUT_DIR, "stitched_seam_blend.jpg"), blended)

    # デバッグ用のαを出力
    cv2.imwrite(os.path.join(OUT_DIR, "alpha_visual.png"), (alpha*255).astype(np.uint8))

    print("[DONE] 出力：")
    print(" - preview_overlap.jpg（手動時のプレビュー）/ auto_seam_preview.jpg（自動時）")
    print(" - alpha_visual.png（アルファ可視化）")
    print(" - stitched_seam_blend.jpg（最終合成）")

if __name__ == "__main__":
    main()
