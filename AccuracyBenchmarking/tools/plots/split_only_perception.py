
import json
import random
from collections import defaultdict

file_path = "data/data/drivebench-test.json"
out_file_path = "data/data/drivebench-test-perception-100-scenes.json"

seed = 42
max_scenes = 100

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

perception = [
    item for item in data
    if item.get("question_type") == "perception"
    and item.get("scene_token") is not None
    and item.get("frame_token") is not None
]

# Group questions by scene, then frame.
by_scene = defaultdict(lambda: defaultdict(list))

for item in perception:
    by_scene[item["scene_token"]][item["frame_token"]].append(item)

rng = random.Random(seed)
scenes = list(by_scene.keys())
rng.shuffle(scenes)
scenes = scenes[:max_scenes]

out = []

for scene_token in scenes:
    # Select one frame from this scene.
    frame_token = rng.choice(list(by_scene[scene_token].keys()))

    # Select one perception question from that frame.
    question = rng.choice(by_scene[scene_token][frame_token])
    out.append(question)

with open(out_file_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Available perception questions: {len(perception)}")
print(f"Available unique scenes: {len(by_scene)}")
print(f"Selected unique scenes: {len({x['scene_token'] for x in out})}")
print(f"Selected unique frames: {len({x['frame_token'] for x in out})}")
print(f"Selected questions: {len(out)}")
print(f"Saved to: {out_file_path}")
