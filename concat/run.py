
import sys, subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()  # ← run.py がある cam1 フォルダ
SCRIPTS = [
    "pick_roi_v2.py",
    "crop_concat_v2.py"
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
