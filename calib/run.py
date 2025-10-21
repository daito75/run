# run.py
#実行:python run.py --kaiseki 2 --cam-ids "1-3,5,7" --root D:\BRLAB\2025\mizuno\done\deta
#python run.py --kaiseki 2 --cams 3 --root D:\BRLAB\2025\mizuno\done\deta

import sys, subprocess
from pathlib import Path
import argparse, re

def parse_cam_ids(cam_ids: str):
    ids = []
    for part in cam_ids.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            ids += list(range(int(a), int(b)+1))
        else:
            ids.append(int(part))
    # 重複/ソート整形
    return sorted(set(ids))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaiseki", type=int, required=True)
    ap.add_argument("--cams", type=int, help="cam1..camN を使う（後方互換）")
    ap.add_argument("--cam-ids", type=str, help='例: "1-3,5,7"')
    ap.add_argument("--root", default=r"D:\BRLAB\2025\mizuno\done\deta")
    ap.add_argument("--interval-sec", type=float, default=1.0)
    ap.add_argument("--checker", default="7x6")
    ap.add_argument("--square-mm", type=float, default=30.0)
    ap.add_argument("--model", choices=["fisheye","pinhole"], default="fisheye")
    args = ap.parse_args()

    # cam ID 決定
    if args.cam_ids:
        cam_ids_str = args.cam_ids
        cam_ids = parse_cam_ids(cam_ids_str)
    elif args.cams:
        cam_ids = list(range(1, args.cams+1))
        cam_ids_str = ",".join(map(str, cam_ids))
    else:
        raise SystemExit("ERROR: --cam-ids または --cams のどちらかを指定してね")

    BASE = Path(__file__).parent.resolve()
    print(f"[INFO] base={BASE}")
    print(f"[INFO] kaiseki={args.kaiseki}, cam_ids={cam_ids}")

    scripts = [
        "extract_pairs_multi.py",
        "scan_useful_images_batch.py",
        "calibrate_batch_fisheye.py",
    ]
    common = [
        f"--kaiseki={args.kaiseki}",
        f"--cam-ids={cam_ids_str}",
        f"--root={args.root}",
        f"--interval-sec={args.interval_sec}",
        f"--checker={args.checker}",
        f"--square-mm={args.square_mm}",
        f"--model={args.model}",
    ]
    for name in scripts:
        script = (BASE / name).resolve()
        if not script.exists():
            print(f"[ERROR] Not found: {script}")
            sys.exit(1)
        print(f"\n=== RUN: {script.name} ===")
        subprocess.run([sys.executable, str(script), *common], check=True, cwd=str(BASE))

    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n*** FAILED: {e} ***")
        sys.exit(e.returncode)
