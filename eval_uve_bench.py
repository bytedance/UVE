# Copyright (2025) [UVE] Yuanxin Liu 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

import argparse, os, json, math, csv
from tqdm import tqdm
from uve.constants import *

from uve import UVE

def load_json(json_file):
    with open(json_file, 'r') as f:
        datas = json.load(f)
    return datas

def save_json(datas, json_file):
    with open(json_file, 'w') as f:
        datas = json.dump(datas, f, indent=4)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    print(len(lst), type(n), type(k))
    chunks = split_list(lst, n)
    return chunks[k]

def inference_single(args, evaluator):
    results = load_json(args.infer_result_file) if os.path.isfile(args.infer_result_file) else {}
    annos = load_json(os.path.join(args.anno_path, "annotations.json"))
    annos = dict(get_chunk(list(annos.items()), args.num_chunks, args.chunk_idx))

    for idx, anno in tqdm(annos.items()):
        video_prompt = anno['prompt']
        for subaspect in anno['subaspects']:
            eval_idx = f"{idx}-{subaspect}"
            if eval_idx in results:
                continue
            if 'videoscore' in args.model_name and (subaspect not in aspect2videoscoredim):
                continue
            curr_result={
                "idx": idx,
                "text_prompt": video_prompt,
                "video_a": anno['video_a'],
                "video_b": anno['video_b'],
                "aspect": anno['aspect'],
                "subaspect": subaspect,
                "attribution": anno['attribution'] if 'attribution' in anno else None,
                "human_rating": anno['preference'],
                "difficulty": anno['difficulty'] if 'difficulty' in anno else None,
                "vlm_rating": {}
            }
            for vid in ['video_a', 'video_b']:
                vfile = os.path.join(args.video_path, anno['dataset'], anno[vid])
                ans_score = evaluator.evaluate(vfile, video_prompt, subaspect, args.eval_mode, args.prompt_type)
                curr_result["vlm_rating"][vid] = ans_score
            results[eval_idx] = curr_result

        save_json(results, args.infer_result_file)

def inference_pairwise(args, evaluator):
    results = load_json(args.infer_result_file) if os.path.isfile(args.infer_result_file) else {}
    annos = load_json(os.path.join(args.anno_path, "annotations_balanced.json"))
    annos = dict(get_chunk(list(annos.items()), args.num_chunks, args.chunk_idx))

    for idx, anno in tqdm(annos.items()):
        video_prompt = anno['prompt']
        for subaspect in anno['subaspects']:
            eval_idx = f"{idx}-{subaspect}"
            if eval_idx in results:
                continue
            curr_result={
                "idx": idx,
                "text_prompt": video_prompt,
                "video_a": anno['video_a'],
                "video_b": anno['video_b'],
                "aspect": anno['aspect'],
                "subaspect": subaspect,
                "attribution": anno['attribution'] if 'attribution' in anno else None,
                "human_rating": anno['preference'],
                "difficulty": anno['difficulty'] if 'difficulty' in anno else None,
            }
            vfiles = [os.path.join(args.video_path, anno['dataset'], anno[vid]) for vid in ['video_a', 'video_b']]
            ans_score = evaluator.evaluate(vfiles, video_prompt, subaspect, args.eval_mode, args.prompt_type)
            curr_result["vlm_rating"] = ans_score
            results[eval_idx] = curr_result

        save_json(results, args.infer_result_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--model_name', default='qwen2-vl-7b')
    parser.add_argument('--local_model_path', default=None, help="local path to load mllms, if cannot download from huggingface, set this to local path")
    parser.add_argument('--video_path', default="uve_bench_videos") 
    parser.add_argument('--anno_path', default="annotations", help='path to human preference annotations')
    parser.add_argument('--result_path', default='results')
    parser.add_argument('--eval_mode', default='single_hard', choices=['single_hard', 'single_soft_yn', 'single_soft_good_bad', 'single_soft_adaptive', 'single_soft_custom', 
                                                                          'single_soft_reg-avg', 'single_soft_reg-dim', 'pairwise', 'pairwise_no_vid_index'])
    parser.add_argument('--prompt_type', default='full', choices=['full', '_wo_answer_prompt', '_wo_task_instruction', '_wo_aspect_description', '_simple_prompt'])
    parser.add_argument('--max_num_frames', default=16, type=int)
    parser.add_argument('--base_url', default=None)
    parser.add_argument('--ak', default=None, help="Your OpenAI API key")
    parser.add_argument('--num_chunks', default=1, type=int)
    parser.add_argument('--chunk_idx', default=0, type=int)
    args = parser.parse_args()

    evaluator = UVE(args.model_name, args.max_num_frames, args.local_model_path, args.base_url, args.ak)

    args.infer_result_file = f"{args.result_path}/{args.eval_mode}{args.prompt_type if args.prompt_type!='full' else ''}_frame{args.max_num_frames}_{args.num_chunks}_{args.chunk_idx}.json"
    os.makedirs(os.path.dirname(args.infer_result_file), exist_ok=True)

    if args.eval_mode.startswith('single'):
        inference_single(args, evaluator)
    elif args.eval_mode.startswith('pairwise'):
        inference_pairwise(args, evaluator)