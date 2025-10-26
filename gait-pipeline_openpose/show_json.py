import json

# BODY_25 の対応表
BODY25_JP = {
    0: "鼻",
    1: "首",
    2: "右肩",
    3: "右肘",
    4: "右手首",
    5: "左肩",
    6: "左肘",
    7: "左手首",
    8: "腰（中央）",
    9: "右腰",
    10: "右膝",
    11: "右足首",
    12: "左腰",
    13: "左膝",
    14: "左足首",
    15: "右目",
    16: "左目",
    17: "右耳",
    18: "左耳",
    19: "左足親指",
    20: "左足小指",
    21: "左かかと",
    22: "右足親指",
    23: "右足小指",
    24: "右かかと"
}

# --- JSONファイルのパスを指定 ---
json_path = r"C:\brlab\2025\mizuno\openpose\3.0m\60cm\json\60cm_000000000250_keypoints.json"

# --- JSONを読み込み ---
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

people = data.get("people", [])
if not people:
    print("No people detected.")
else:
    keypoints = people[0]["pose_keypoints_2d"]
    print(f"\n[INFO] Detected {len(keypoints)//3} keypoints (BODY_25)\n")

    # --- 3要素ずつ取り出して整形出力 ---
    for i in range(25):
        x = keypoints[3*i]
        y = keypoints[3*i + 1]
        c = keypoints[3*i + 2]
        name = BODY25_JP.get(i, f"kp{i}")
        print(f"{i:2d}: {name:12s}  x={x:8.2f},  y={y:8.2f},  conf={c:.3f}")
