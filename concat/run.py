# run_concat.py
"""
cam1 と cam3 を並べる。解析=2、アルファ=0.1 → a0.1
python run.py --kaiseki 3 --cam-ids "1,3" --a 1.0 `
  --root D:\BRLAB\2025\mizuno\done\deta `
  --scale 0.5 `
  --offset-sec "1:0,3:0"
  
"""



import sys, subprocess
from pathlib import Path
import argparse

def parse_cam_ids(s: str):
    ids = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            ids.extend(range(int(a), int(b)+1))
        else:
            ids.append(int(part))
    # 順序維持のため set しない（重複だけ落とす）
    out = []
    for x in ids:
        if x not in out:
            out.append(x)
    return out

def main():
    ap = argparse.ArgumentParser(description="ROI選択→クロップ結合を一括実行")
    ap.add_argument("--kaiseki", type=int, required=True, help="解析番号 (例: 2)")
    ap.add_argument("--cam-ids", type=str, help='使うカメラID列。例: "1-3,5"')
    ap.add_argument("--cams", type=int, help="1..N（後方互換）")
    ap.add_argument("--a", type=str, required=True, help='アルファ文字列（例: "0.1" -> a0.1 フォルダ）')
    ap.add_argument("--root", default=r"D:\BRLAB\2025\mizuno\done\deta", help="ルートフォルダ")
    ap.add_argument("--target-height", type=int, default=480, help="結合出力の高さ(px)")
    ap.add_argument("--scale", type=float, default=0.5, help="ROI選択時の表示縮尺")
    ap.add_argument("--offset-sec", default="", help='同期オフセット "camId:sec,..." 例 "1:0,3:0.5"')
    args = ap.parse_args()

    # cam 列の決定
    if args.cam_ids:
        cam_ids = parse_cam_ids(args.cam_ids)
        cam_ids_str = args.cam_ids
    elif args.cams:
        cam_ids = list(range(1, args.cams+1))
        cam_ids_str = ",".join(map(str, cam_ids))
    else:
        raise SystemExit("ERROR: --cam-ids か --cams を指定してね")

    base = Path(__file__).parent.resolve()
    print(f"[INFO] base={base}")
    print(f"[INFO] kaiseki={args.kaiseki}, cams={cam_ids}, a={args.a}")

    common = [
        f"--kaiseki={args.kaiseki}",
        f"--cam-ids={cam_ids_str}",
        f"--a={args.a}",
        f"--root={args.root}",
    ]

    # 1) ROI選択
    print("\n=== RUN: pick_roi.py ===")
    subprocess.run(
        [sys.executable, str(base/"pick_roi.py"),
         *common,
         f"--scale={args.scale}"],
        check=True, cwd=str(base)
    )

    # 2) クロップ結合
    print("\n=== RUN: crop_concat.py ===")
    cmd = [sys.executable, str(base/"crop_concat.py"),
           *common,
           f"--target-height={args.target_height}"]
    if args.offset_sec:
        cmd.append(f"--offset-sec={args.offset_sec}")
    subprocess.run(cmd, check=True, cwd=str(base))

    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    main()
