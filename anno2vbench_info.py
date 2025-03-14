# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, json
from tqdm import tqdm

anno_path = "annotations/annotations.json"
info_path = "vbench_info.json"
video_root = "uve_bench_videos"

aspects2bvenchdim = {
    "tv_alignment": "overall_consistency",
    "motion_naturalness": "motion_smoothness",
    "aesthetic_quality": "aesthetic_quality",
    "technical_quality": "imaging_quality",
    "dynamic_degree": "dynamic_degree",
    "appearance_consistency": "subject_consistency",
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path, indent=4):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


annos = load_json(anno_path)
full_info = []

for anno_id, anno in tqdm(annos.items()):
    for subaspect in anno['subaspects']:
        if subaspect not in aspects2bvenchdim:
            continue
        for vid in ['video_a', 'video_b']:
            idx = f"{anno_id}-{anno[vid].split('/')[0]}"
            video_path = os.path.join(video_root, anno['dataset'], anno[vid])
            prompt = anno['prompt'] if anno['prompt'] else ""
            full_info.append({
                "idx": idx,
                "prompt_en": prompt,
                "dimension": [aspects2bvenchdim[subaspect]],
                "video_list": [video_path]
            })
    
save_json(full_info, info_path)
