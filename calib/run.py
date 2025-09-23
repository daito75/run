import sys, subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()  # ← run.py がある cam1 フォルダ

SCRIPTS = [
    "extract_pairs_multi.py",
    "scan_useful_images_batch.py",  # 名前にミスがないか要確認
    "calibrate_batch_fisheye.py",
]

def main():
    print(f"[INFO] runner base: {BASE}")
    for name in SCRIPTS:
        script = (BASE / name).resolve()
        if not script.exists():
            print(f"[ERROR] Not found: {script}")
            sys.exit(1)
        print(f"\n=== RUN: {script} ===")
        # cwd=BASE に固定：各スクリプトの相対パスが cam 基準で解決される
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(BASE))
    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n*** FAILED: {e} ***")
        sys.exit(e.returncode)
