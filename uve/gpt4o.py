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

from string import Template
import torch.nn.functional as F
import os, copy, json, time
from .utils import *
from .constants import *

def eval_gpt4o_single(video_path, video_prompt, aspect, eval_mode, prompt_type,
                      model, processor, tokenizer, max_num_frames, 
                      pos_tokens, neg_tokens, custom_prompt,
                      model_name="gpt-4o-2024-08-06", sys_prompt="You are an expert in video understanding.", maxtry=10):
    template = Template(mode2prompt[eval_mode][aspect])
    prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template

    if video_path.endswith('.jpg'):
        frame_files = [video_path]
        encoded_frms = []
        for ff in frame_files:
            encoded_frms.append(encode_image_gpt(ff))
    elif video_path.endswith('.mp4'):
        encoded_frms = encode_video_gpt(video_path, os.path.dirname(video_path)+f"_frames{max_num_frames}/{os.path.basename(video_path).replace('.mp4', '')}", max_num_frames)

    while True:
        try:
            llm_response = get_response_gpt(model, encoded_frms, prompt, sys_prompt, model_name, return_prob=True if 'soft' in eval_mode else False)
            time.sleep(4)
            return llm_response
        except:
            if maxtry<=0:
                llm_response = None
                return llm_response
            maxtry -= 1
            print(f"Not success! {maxtry} retries remaining...")
            time.sleep(10)

def eval_gpt4o_pairwise(video_paths, video_prompt, aspect, eval_mode, prompt_type,
                      model, processor, tokenizer, max_num_frames, 
                      pos_tokens, neg_tokens, custom_prompt,
                      model_name="gpt-4o-2024-08-06", sys_prompt="You are an expert in video understanding.", maxtry=10):
    template = Template(mode2prompt[eval_mode][aspect])
    prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template

    encoded_vids = []
    for video_path in video_paths:
        encoded_frms = encode_video_gpt(video_path, 
                                     os.path.dirname(video_path)+f"_frames{max_num_frames}/{os.path.basename(video_path).replace('.mp4', '')}", 
                                     max_num_frames)
        encoded_vids.append(encoded_frms)

    while True:
        try:
            llm_response = get_response_gpt_video_pair(model, encoded_vids, prompt, sys_prompt, model_name)
            time.sleep(4)
            return llm_response
        except:
            if maxtry<=0:
                llm_response = None
                return llm_response
            maxtry -= 1
            print(f"Not success! {maxtry} retries remaining...")
            time.sleep(10)
