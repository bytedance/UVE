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
from .utils import *
from .constants import *
import os, copy, json, time
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token

def eval_llava_ov_single(video_path, video_prompt, aspect, eval_mode, prompt_type,
                        model, processor, tokenizer, max_num_frames,
                        pos_tokens, neg_tokens, custom_prompt):
    if eval_mode=='single_soft_custom':
        prompt = custom_prompt
    else:
        template = Template(mode2prompt[eval_mode][aspect])
        prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template

    if max_num_frames>0:
        video_frames = load_video_llava_onevision(video_path, max_num_frames)
        # print(video_frames.shape) # (max_num_frames, 1024, 576, 3)
        image_tensors = []
        frames = processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)
        image_sizes = [frame.size for frame in video_frames]
        question = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    else:
        image_tensors, image_sizes = None, None
        question = prompt

    # Prepare conversation input
    conv_template = "qwen_1_5"
    from llava.conversation import conv_templates
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    if eval_mode.startswith('single_soft_yn'):
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=1,
            modalities=["video"],
            return_dict_in_generate=True, 
            output_logits=True
        )
        logits = cont.logits
        probs = [F.softmax(logits_)[0] for logits_ in logits]
        yes_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()=='yes']
        no_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()=='no']
        yes_prob = sum([probs[0][id] for id in yes_ids])
        no_prob = sum([probs[0][id] for id in no_ids])
        yes_prob, no_prob = yes_prob / (yes_prob + no_prob), no_prob / (yes_prob + no_prob)
        return float(yes_prob.cpu())
    elif eval_mode=='single_hard':
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4,
            modalities=["video"]
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    
def eval_llava_ov_pairwise(video_paths, video_prompt, aspect, eval_mode, prompt_type,
                           model, processor, tokenizer, max_num_frames):
    template = Template(mode2prompt[eval_mode][aspect])
    prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template

    image_tensors, image_sizes, question = [], [], ""
    for video_index, video_path in zip(['The first video:\n', 'The second video:\n'], video_paths):
        video_frames = load_video_llava_onevision(video_path, max_num_frames)
        image_sizes += [frame.size for frame in video_frames]
        # print(video_frames.shape) # (max_num_frames, 1024, 576, 3)
        frames = processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)

        question += f"{video_index}{DEFAULT_IMAGE_TOKEN}\n"
    question += prompt

    conv_template = "qwen_1_5"
    from llava.conversation import conv_templates
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=5,
        modalities=["video", "video"]
    )
    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return response
