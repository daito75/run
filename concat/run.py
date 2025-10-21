# run_concat.py
"""
cam1 と cam3 を並べる。
python run.py --kaiseki 4 --cam-ids "1,2,3" `
  --root D:\BRLAB\2025\mizuno\done\deta `
  --scale 0.5 `
  --offset-sec "1:0,2:0,3:0" `
  --in-map "1:D:\BRLAB\2025\mizuno\done\deta\kaiseki4\syaeihenkann\3.0m\cam1_on_plane.mp4;2:D:\BRLAB\2025\mizuno\done\deta\kaiseki4\syaeihenkann\3.0m\cam2_on_plane.mp4;3:D:\BRLAB\2025\mizuno\done\deta\kaiseki4\syaeihenkann\3.0m\cam3_on_plane.mp4" `
  --out-dir "D:\BRLAB\2025\mizuno\done\deta\kaiseki4\concat\3.0m"
"""

import sys, subprocess, argparse, glob
from pathlib import Path
from typing import Dict, List, Optional

# ===== カメラID関連ユーティリティ =====
def parse_cam_ids(s: str) -> List[int]:
    ids = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    # 順序維持 + 重複排除
    out = []
    for x in ids:
        if x not in out:
            out.append(x)
    return out

def parse_in_map(s: str) -> Dict[int, Path]:
    """
    "1:D:\\p1.mp4;3:D:\\p2.mp4" → {1: Path(...), 3: Path(...)}
    """
    if not s:
        return {}
    result = {}
    for pair in s.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"--in-map 形式エラー: {pair}")
        cam_str, path_str = pair.split(":", 1)
        result[int(cam_str.strip())] = Path(path_str.strip())
    return result

def auto_find_inputs(cam_ids: List[int], search_dirs: List[Path], name_pattern: str) -> Dict[int, Path]:
    """
    指定ディレクトリ群から cam{id} 名パターンで動画を再帰探索
    """
    out = {}
    if not search_dirs or not name_pattern:
        return out

    for cid in cam_ids:
        filename = name_pattern.replace("{id}", str(cid)).replace("{cam}", f"cam{cid}")
        found = None
        for d in search_dirs:
            patterns = [str(d / "**" / filename), str(d / "**" / filename.lower())]
            for pat in patterns:
                hits = glob.glob(pat, recursive=True)
                if hits:
                    found = Path(hits[0]).resolve()
                    break
            if found:
                break
        if found:
            out[cid] = found
    return out

def build_pass_through_list(s: str) -> List[str]:
    """--pass-through の中身を空白で分解"""
    if not s:
        return []
    return [tok for tok in s.strip().split() if tok.strip()]

# ===== メイン =====
def main():
    ap = argparse.ArgumentParser(description="ROI選択→クロップ結合を一括実行（入出力柔軟版, α削除）")
    ap.add_argument("--kaiseki", type=int, required=True, help="解析番号 (例: 2)")
    ap.add_argument("--cam-ids", type=str, help='使うカメラID列。例: "1-3,5"')
    ap.add_argument("--cams", type=int, help="1..N（後方互換）")
    ap.add_argument("--root", default=r"D:\BRLAB\2025\mizuno\done\deta", help="ルートフォルダ（子スクリプトへ渡す）")
    ap.add_argument("--target-height", type=int, default=480, help="結合出力の高さ(px)")
    ap.add_argument("--scale", type=float, default=0.5, help="ROI選択時の表示縮尺")
    ap.add_argument("--offset-sec", default="", help='同期オフセット "camId:sec,..." 例 "1:0,3:0.5"')

    # 入力系
    ap.add_argument("--in-map", default="", help='カメラID→動画パスの明示指定。例 "1:D:\\p1.mp4;3:D:\\p2.mp4"')
    ap.add_argument("--search-dirs", default="", help='探索ディレクトリ(;区切り) 例 "D:\\walks;E:\\dump"')
    ap.add_argument("--name-pattern", default="cam{id}_walk.mp4", help='探索ファイル名パターン')

    # 出力系
    ap.add_argument("--out-dir", default="", help="最終出力フォルダ（子スクリプトにも渡す）")

    # 拡張オプションパススルー
    ap.add_argument("--pass-through", default="", help="pick_roi.py / crop_concat.py にそのまま渡すオプション列")

    args = ap.parse_args()

    # cam 列の決定
    if args.cam_ids:
        cam_ids = parse_cam_ids(args.cam_ids)
        cam_ids_str = args.cam_ids
    elif args.cams:
        cam_ids = list(range(1, args.cams + 1))
        cam_ids_str = ",".join(map(str, cam_ids))
    else:
        raise SystemExit("ERROR: --cam-ids か --cams を指定してね")

    base = Path(__file__).parent.resolve()
    print(f"[INFO] base={base}")
    print(f"[INFO] kaiseki={args.kaiseki}, cams={cam_ids}")

    # 入力解決
    in_map = parse_in_map(args.in_map)
    if not in_map:
        search_dirs = [Path(p).resolve() for p in args.search_dirs.split(";") if p.strip()]
        if search_dirs:
            auto_map = auto_find_inputs(cam_ids, search_dirs, args.name_pattern)
            in_map.update(auto_map)

    missing = [cid for cid in cam_ids if cid not in in_map]
    if missing:
        print(f"[WARN] 未解決のcam: {missing} → 子スクリプト側に任せる")

    # ===== 引数セットを分離（pick と cropで分ける） =====
    # pick_roi.py 用（--offset-sec は渡さない）
    common_pick = [
        f"--kaiseki={args.kaiseki}",
        f"--cam-ids={cam_ids_str}",
        f"--root={args.root}",
    ]
    if args.out_dir:
        common_pick.append(f"--out-dir={args.out_dir}")
    if in_map:
        imap_str = ";".join([f"{cid}:{p}" for cid, p in in_map.items()])
        common_pick.append(f"--in-map={imap_str}")

    # crop_concat.py 用（こっちは offset-sec を渡す）
    common_crop = [
        f"--kaiseki={args.kaiseki}",
        f"--cam-ids={cam_ids_str}",
        f"--root={args.root}",
    ]
    if args.out_dir:
        common_crop.append(f"--out-dir={args.out_dir}")
    if in_map:
        common_crop.append(f"--in-map={imap_str}")
    if args.offset_sec:
        common_crop.append(f"--offset-sec={args.offset_sec}")

    passthru = build_pass_through_list(args.pass_through)


    # ===== 実行フロー =====
    print("\n=== RUN: pick_roi.py ===")
    subprocess.run(
        [sys.executable, str(base / "pick_roi.py"), *common_pick, f"--scale={args.scale}", *passthru],
        check=True, cwd=str(base)
    )   
    cmd = [sys.executable, str(base / "crop_concat.py"), *common_crop, f"--target-height={args.target_height}", *passthru]

    subprocess.run(cmd, check=True, cwd=str(base))

    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    main()
