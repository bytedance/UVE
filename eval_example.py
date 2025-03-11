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

import argparse
from uve import UVE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--model_name', default="qwen2-vl-7b")     
    parser.add_argument('--local_model_path', default=None, help="local path to load mllms, if cannot download from huggingface, set this to local path")
    parser.add_argument('--video_path1', default=None, required=True)
    parser.add_argument('--video_path2', default=None)
    parser.add_argument('--base_url', default=None)
    parser.add_argument('--pos_tokens', default="yes", help="positive scoring tokens for customized evaluation")
    parser.add_argument('--neg_tokens', default="no", help="negative scoring tokens for customized evaluation")
    parser.add_argument('--custom_prompt', default=None)
    parser.add_argument('--ak', default=None, help="Your OpenAI API key")
    parser.add_argument('--video_prompt', default=None, help='required when evluating tv_alignment')     
    parser.add_argument('--aspect', default="dynamic_degree,appearance_consistency") 
    parser.add_argument('--eval_mode', default='single_soft_yn', choices=['single_hard', 'single_soft_yn', 'single_soft_good_bad', 'single_soft_adaptive', 'single_soft_custom', 
                                                                          'single_soft_reg-avg', 'single_soft_reg-dim', 'pairwise', 'pairwise_no_vid_index'])
    parser.add_argument('--prompt_type', default='full', choices=['full', '_wo_answer_prompt', '_wo_task_instruction', '_wo_aspect_description', '_simple_prompt'])
    parser.add_argument('--max_num_frames', default=16, type=int)
    args = parser.parse_args()

    evaluator = UVE(args.model_name, args.max_num_frames, args.local_model_path, args.base_url, args.ak)
    for aspect in args.aspect.split(','):
        if args.video_path2 is not None:
            video_path = [args.video_path1, args.video_path2]
        else:
            video_path = args.video_path1
        score = evaluator.evaluate(video_path, args.video_prompt, aspect, args.eval_mode, args.prompt_type, args.pos_tokens, args.neg_tokens, args.custom_prompt)
        print(f"{aspect}: {score}")