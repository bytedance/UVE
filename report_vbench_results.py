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

import json
from tqdm import tqdm
from cat_eval_results import eval_model_rating

aspects = ["dynamic_degree",
           "aesthetic_quality", "technical_quality",
           "appearance_consistency", "motion_naturalness",
           "tv_alignment"]

aspects2bvenchdim = {
    "tv_alignment": "overall_consistency",
    "motion_naturalness": "motion_smoothness",
    "aesthetic_quality": "aesthetic_quality",
    "technical_quality": "imaging_quality",
    "dynamic_degree": "dynamic_degree",
    "appearance_consistency": "subject_consistency",
}
vbenchdim2aspect = {
    "overall_consistency": "tv_alignment",
    "motion_smoothness": "motion_naturalness",
    "aesthetic_quality": "aesthetic_quality",
    "imaging_quality": "technical_quality",
    "dynamic_degree": "dynamic_degree",
    "subject_consistency": "appearance_consistency",
}

def load_json(json_file):
    with open(json_file, 'r') as f:
        datas = json.load(f)
    return datas

def normalize_vbenchscore(score, aspect):
    if aspect in ['appearance_consistency', 'motion_naturalness']:
        return score
    elif aspect=='aesthetic_quality':
        return score*1.25
    elif aspect=='tv_alignment':
        return score * 2.5
    elif aspect=='technical_quality':
        return score / 100
    elif aspect=='dynamic_degree':
        return score
    
vbench_results = load_json("results/vbench/results.json")
annos = load_json("annotations/annotations.json")

results = {}
num_correct, num_total = 0, 0
for aspect in aspects:
    results[aspect] = {"correct": 0, "total": 0}

for anno_id, anno in tqdm(annos.items()):
    for subaspect in anno['subaspects']:
        if subaspect not in aspects2bvenchdim:
            continue
        vlm_rating = {}
        curr_results = vbench_results[aspects2bvenchdim[subaspect]][1]
        for r in curr_results:
            if subaspect=='tv_alignment':
                if anno_id not in r['idx']:
                    continue
                if anno['video_a'].split('/')[-1]==r['video_path'].split('/')[-1]:
                    vlm_rating['video_a'] = normalize_vbenchscore(r['video_results'], subaspect)
                elif anno['video_b'].split('/')[-1]==r['video_path'].split('/')[-1]:
                    vlm_rating['video_b'] = normalize_vbenchscore(r['video_results'], subaspect)
            else:
                if anno['dataset'] not in r['video_path']:
                    continue
                if anno['video_a'].split('/')[-1]==r['video_path'].split('/')[-1]:
                    vlm_rating['video_a'] = normalize_vbenchscore(r['video_results'], subaspect)
                elif anno['video_b'].split('/')[-1]==r['video_path'].split('/')[-1]:
                    vlm_rating['video_b'] = normalize_vbenchscore(r['video_results'], subaspect)
        rating, choice = eval_model_rating(anno['preference'], vlm_rating, video_eval_mode='single_soft_yn', mllm_eval_mode='score')
        results[subaspect]['correct'] += rating
        results[subaspect]['total'] += 1
        num_correct += rating
        num_total += 1
for aspect, result in results.items():
    if result['total'] > 0:
        score = 100*result['correct'] / result['total']
        print(f"{aspect}: {score:.2f}% ({result['correct']}/{result['total']})")
print(f"VBench Metrics: {100*num_correct/num_total:.2f} ({num_correct:.2f}/{num_total})")
