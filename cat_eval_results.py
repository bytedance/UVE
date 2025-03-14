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

import os, json, argparse, re
import numpy as np

good_threh, bad_threh = 0.8, 0.4
aspects = ["dynamic_degree", "subject_motion_degree", "camera_motion_degree", "light_change", 
           "static_visual_quality", "aesthetic_quality", "technical_quality", "structural_correctness",
           "temporal_visual_quality", "appearance_consistency", "flickering", "motion_naturalness",
           "tv_alignment", "tv_alignment_appearance", "tv_alignment_motion"]

def load_json(json_file):
    with open(json_file, 'r') as f:
        datas = json.load(f)
    return datas

def save_json(datas, json_file):
    with open(json_file, 'w') as f:
        datas = json.dump(datas, f, indent=4)

def exp_decrease(x, left, right, speed=10, init_value=1, final_value=0, reversed=False):
    """
        expenential decay from left to right
        reversed: if True, decay from right to left
    """
    if reversed:
        return np.where(x > right, 
                init_value,  # if x > right, return init_value
                np.where(x <= right, 
                            np.exp(-speed * (right - x)) * init_value,  # if left <= x <= right, use exponential function
                            final_value))  # if x < left, return final_value
    else:
        return np.where(x < left, 
                init_value,  # if x < left, return init_value
                np.where(x <= right, 
                            np.exp(-speed * (x - left)) * init_value,  # if left <= x <= right, use exponential function
                            final_value))  # if x > right, return final_value

def extract_hard_rating(vlm_rating):
    for key in ["video_a", "video_b"]:
        rating = re.findall(r'\d+', vlm_rating[key])
        vlm_rating[key] = int(rating[0]) if rating else 0
    return vlm_rating

def match_pairwise(human_rating, vlm_rating):
    """
        human_rating: A is better, B is better, same good, same bad
        vlm_rating: the first video, the second video, same good, same bad
    """
    if "first video" in vlm_rating.lower():
        vlm_rating = "A is better"
    elif "second video" in vlm_rating.lower():
        vlm_rating = "B is better"
    elif "same good" in vlm_rating.lower() or "same high" in vlm_rating.lower() or "both good" in vlm_rating.lower() or "both high" in vlm_rating.lower():
        vlm_rating = "same good"
    elif "same bad" in vlm_rating.lower() or "same low" in vlm_rating.lower() or "both bad" in vlm_rating.lower() or "both low" in vlm_rating.lower():
        vlm_rating = "same bad"
    else:
        return False, ""
    return human_rating == vlm_rating, vlm_rating

def eval_model_rating(human_rating, vlm_rating, video_eval_mode, mllm_eval_mode='score'):
    """
        video_eval_mode: how videos are evaluated
        mllm_eval_mode: how mllms are evaluated
    """
    assert video_eval_mode in ['pairwise', 'pairwise_no_vid_index', 'single_soft_yn', 'single_soft_good_bad', 'single_soft_adaptive',
                               'single_hard', 'single_soft_reg-dim', 'single_soft_reg-avg']
    assert mllm_eval_mode in ['acc', 'score']
    if video_eval_mode == "single_hard":
        vlm_rating = extract_hard_rating(vlm_rating)
    
    if isinstance(vlm_rating, dict) and isinstance(vlm_rating['video_a'], list):
        vlm_rating['video_a'] = vlm_rating['video_a'][0]
        vlm_rating['video_b'] = vlm_rating['video_b'][0]

    if mllm_eval_mode=='acc' and video_eval_mode.startswith('pairwise'):    # video pair preference
        return match_pairwise(human_rating, vlm_rating)
    elif mllm_eval_mode=='acc' and video_eval_mode.startswith('single'):    # adapt single video rating to pairwise preference
        if abs(vlm_rating['video_a']-vlm_rating['video_b'])>=0.05 or (vlm_rating['video_a']<good_threh and vlm_rating['video_a']>bad_threh) or (vlm_rating['video_b']<good_threh and vlm_rating['video_b']>bad_threh):
            vlm_rating = "A is better" if vlm_rating['video_a']>vlm_rating['video_b'] else "B is better"
        elif vlm_rating['video_a']>=good_threh and vlm_rating['video_b']>=good_threh:
            vlm_rating = "same good"
        elif vlm_rating['video_a']<=bad_threh and vlm_rating['video_b']<=bad_threh:
            vlm_rating = "same bad"
        else:
            print(vlm_rating)
        return human_rating==vlm_rating, vlm_rating
    elif mllm_eval_mode=='score':   # single video rating
        if video_eval_mode.startswith('single_soft_reg'):   # normalize videoscore rating to [0,1]
            for key in ["video_a", "video_b"]:
                vlm_rating[key] = (vlm_rating[key]-1) / (4-1)
        if human_rating == "A is better":
            return vlm_rating["video_a"] > vlm_rating["video_b"], None
        elif human_rating == "B is better":
            return vlm_rating["video_a"] < vlm_rating["video_b"], None
        elif human_rating == "same good":
            return exp_decrease(vlm_rating["video_a"], 0, good_threh, reversed=True) * exp_decrease(vlm_rating["video_b"], 0, good_threh, reversed=True), None
        elif human_rating == "same bad":
            return exp_decrease(vlm_rating["video_a"], bad_threh, 1) * exp_decrease(vlm_rating["video_b"], bad_threh, 1), None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--result_path', default=None)   
    parser.add_argument('--result_prefix', default=None) 
    parser.add_argument('--eval_mode', default=None)   
    parser.add_argument('--mllm_eval_mode', default=None)   
    parser.add_argument('--overwrite_merge_result', action='store_true')   
    args = parser.parse_args()

    merged_file = os.path.join(args.result_path, '_'.join(args.result_prefix.split('_')[:-1])+'.json')
    if os.path.exists(merged_file) and not args.overwrite_merge_result:
        merged_results = load_json(merged_file)
    else:
        merged_results = {}
        result_files = [f"{args.result_path}/{rf}" for rf in os.listdir(args.result_path) if rf.startswith(args.result_prefix)]
        for rf in result_files:
            results = load_json(rf)
            merged_results.update(results)
            # os.remove(rf)
        save_json(merged_results, merged_file)
    
    # Load Human Annotations
    annos = load_json("annotations/annotations.json")
    annos_balanced = load_json("annotations/annotations_balanced.json")

    # Calculate the results
    num_correct, num_total = 0, 0
    results = {}
    for aspect in aspects:
        results[aspect] = {"correct": 0, "total": 0}

    for r in merged_results.values():
        if args.eval_mode.startswith('pairwise') and (r['idx'] not in annos_balanced or r['subaspect'] not in annos_balanced[r['idx']]['subaspects']):
            continue
        if (r['idx'] not in annos) or (r['subaspect'] not in annos[r['idx']]['subaspects']):
            continue
        subaspect = r['subaspect']
        if subaspect not in aspects:
            continue
        if args.mllm_eval_mode is None:
            mllm_eval_mode = 'score' if args.eval_mode.startswith('single') else 'acc'
        else:
            mllm_eval_mode = args.mllm_eval_mode
        rating, choice = eval_model_rating(r['human_rating'], r['vlm_rating'], video_eval_mode=args.eval_mode, mllm_eval_mode=mllm_eval_mode)   # return rating and the choice of vlm
        results[subaspect]['correct'] += rating
        results[subaspect]['total'] += 1
        num_correct += rating
        num_total += 1
    for aspect, result in results.items():
        if result['total'] > 0:
            score = 100*result['correct'] / result['total']
            print(f"{aspect}: {score:.2f}% ({result['correct']}/{result['total']})")
    print(f"Averate Score: {100*num_correct/num_total:.2f}% ({num_correct:.2f}/{num_total})")
    results['avg'] = {"correct": num_correct, "total": num_total}
