#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cd /mnt/d/BRLAB/2025/mizuno/gait-pipeline_openpose
#実行python3 run_openpose_pipeline.py
#OpenPoseの実行 → JSONとオーバーレイ動画の生成 → 歩行解析スクリプトの実行

import subprocess, sys, os, shutil
from pathlib import Path
import yaml

def run(cmd: list, cwd=None):
    print(">>", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"Command failed with code {r.returncode}")

def main():
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        raise SystemExit("config.yaml が見つかりません。")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    in_video = Path(cfg["in_video"])
    out_dir  = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- OpenPose 設定 ----
    op = cfg["openpose"]
    op_bin = Path(op["bin"])
    model_folder = Path(op["model_folder"])
    render_pose = str(op.get("render_pose", 2))
    alpha_pose = str(op.get("alpha_pose", 0.6))
    alpha_heatmap = str(op.get("alpha_heatmap", 0.0))
    number_people_max = str(op.get("number_people_max", 1))
    num_gpu = str(op.get("num_gpu", 1))
    num_gpu_start = str(op.get("num_gpu_start", 0))
    display = str(op.get("display", 0))
    model_pose = op.get("model_pose", "BODY_25")
    net_resolution = op.get("net_resolution", "-1x368")
    write_overlay_name = op.get("write_overlay_name", "pose_overlay.mp4")
    write_json_subdir = op.get("write_json_subdir", "json")

    overlay_path = out_dir / write_overlay_name
    json_dir = out_dir / write_json_subdir

    # ---- 解析設定 ----
    an = cfg["analysis"]
    conf_th = str(an.get("conf_th", 0.3))
    speed_th = str(an.get("speed_th", 3.0))
    min_hold = str(an.get("min_hold", 4))
    scale_m_per_px = an.get("scale_m_per_px", None)
    overlay_out_subdir = an.get("overlay_out_subdir", "gaitevents")
    overlay_out_dir = out_dir / overlay_out_subdir
    overlay_out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 実行ポリシー ----
    pipe = cfg.get("pipeline", {})
    skip_if_json = bool(pipe.get("skip_openpose_if_json_exists", True))
    force_openpose = bool(pipe.get("force_openpose", False))

    # ---- OpenPose 実行要否判定 ----
    need_openpose = True
    if skip_if_json and json_dir.exists() and any(json_dir.glob("*.json")):
        need_openpose = False
    if force_openpose:
        need_openpose = True

    if need_openpose:
        # 既存JSONを消す（安全のため上位 out_dir は消さない）
        if json_dir.exists():
            shutil.rmtree(json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)

        logging_level = str(op.get("logging_level", 255))
        # OpenPose 実行コマンド
        cmd = [
            str(op_bin),
            "--video", str(in_video),
            "--display", display,
            "--render_pose", render_pose,
            "--alpha_pose", alpha_pose,
            "--alpha_heatmap", alpha_heatmap,
            "--num_gpu", num_gpu, "--num_gpu_start", num_gpu_start,
            "--net_resolution", net_resolution, "--scale_number", "1",
            "--model_pose", model_pose,
            "--model_folder", str(model_folder),
            "--write_video", str(overlay_path),
            "--write_json", str(json_dir),
            "--number_people_max", number_people_max,
            "--logging_level", logging_level,
        ]
        # 注意: --disable_blending は付けない（付けると黒背景に骨格だけになるあるある）
        run(cmd)
        print(f"[OK] OpenPose 完了: {overlay_path}")
        print(f"[OK] JSON: {json_dir}")
    else:
        print(f"[SKIP] JSON既存のため OpenPose をスキップ: {json_dir}")

    # ---- 歩行解析（前回のスクリプトを同フォルダに保存して呼ぶ）----
    py = sys.executable
    gait_script = Path(__file__).with_name("gait_from_openpose_json.py")
    if not gait_script.exists():
        raise SystemExit("gait_from_openpose_json.py がありません（前回のスクリプトを保存してください）。")

    args = [
        py, str(gait_script),
        "--json_dir", str(json_dir),
        "--video_in", str(in_video),
        "--video_overlay", str(overlay_path),
        "--out_dir", str(overlay_out_dir),
        "--conf_th", conf_th,
        "--speed_th", speed_th,
        "--min_hold", min_hold,
    ]
    if scale_m_per_px is not None:
        args += ["--scale_m_per_px", str(scale_m_per_px)]

    run(args)
    print(f"[DONE] 解析出力: {overlay_out_dir}")
    print(f"  - overlay_with_events.mp4")
    print(f"  - metrics_summary.csv")
    print(f"  - events.csv")

if __name__ == "__main__":
    main()
