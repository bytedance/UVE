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

def eval_internvl_2_5_single(video_path, video_prompt, aspect, eval_mode, prompt_type,
                             model, processor, tokenizer, max_num_frames,
                             pos_tokens, neg_tokens, custom_prompt,
                             IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>'):
    if eval_mode=='single_soft_custom':
        prompt = custom_prompt
    else:
        template = Template(mode2prompt[eval_mode][aspect])
        prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template
        if prompt_type=='full':
            prompt = prompt
        elif prompt_type=='_wo_answer_prompt':
            prompt = prompt.replace('\nPlease directly answer yes or no:', '')
        elif prompt_type=='_wo_task_instruction':
            prompt = '\n\n'.join(prompt.split('\n\n')[1:])
        elif prompt_type=='_wo_aspect_description':
            prompt = '\n\n'.join([prompt.split('\n\n')[i] for i in [0, -1]])
        elif prompt_type=='_simple_prompt':
            template = Template(mode2prompt[eval_mode+prompt_type][aspect])
            prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template

    if max_num_frames>0:
        pixel_values, num_patches_list = load_video_internvl(video_path, num_segments=max_num_frames, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    else:
        pixel_values, num_patches_list = None, []
        video_prefix = ""
    question = video_prefix + prompt

    if eval_mode.startswith('single_soft'):
        generation_config = dict(max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_logits=True)

        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id

        template = get_conv_template(model.template)
        template.system_message = model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = []
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(model.device)
        attention_mask = model_inputs['attention_mask'].to(model.device)
        generation_config['eos_token_id'] = eos_token_id
        
        output = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )

        logits = output.logits
        probs = [F.softmax(logits_)[0] for logits_ in logits]
        if eval_mode.startswith('single_soft_yn'):
            yes_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()=='yes']
            no_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()=='no']
            yes_prob = sum([probs[0][id] for id in yes_ids])
            no_prob = sum([probs[0][id] for id in no_ids])
            yes_prob, no_prob = yes_prob / (yes_prob + no_prob), no_prob / (yes_prob + no_prob)
            return float(yes_prob.cpu())
        elif eval_mode == 'single_soft_good_bad':
            good_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()=='good']
            bad_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()=='bad']
            good_prob = sum([probs[0][id] for id in good_ids])
            bad_prob = sum([probs[0][id] for id in bad_ids])
            good_prob, bad_prob = good_prob / (good_prob + bad_prob), bad_prob / (good_prob + bad_prob)
            return float(good_prob.cpu())
        elif eval_mode=='single_soft_adaptive':
            good_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()==asepct2scoretokens[aspect][0]]
            bad_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()==asepct2scoretokens[aspect][1]]
            good_prob = sum([probs[0][id] for id in good_ids])
            bad_prob = sum([probs[0][id] for id in bad_ids])
            good_prob, bad_prob = good_prob / (good_prob + bad_prob), bad_prob / (good_prob + bad_prob)
            return float(good_prob.cpu())
        elif eval_mode == 'single_soft_custom':
            good_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower() in pos_tokens]
            bad_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower() in neg_tokens]
            good_prob = sum([probs[0][id] for id in good_ids])
            bad_prob = sum([probs[0][id] for id in bad_ids])
            good_prob, bad_prob = good_prob / (good_prob + bad_prob), bad_prob / (good_prob + bad_prob)
            return float(good_prob.cpu())
    elif eval_mode=='single_hard':
        generation_config = dict(max_new_tokens=4, do_sample=False)
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                            num_patches_list=num_patches_list, history=None, return_history=True)
        return response
    
def eval_internvl_2_5_pairwise(video_paths, video_prompt, aspect, eval_mode, prompt_type,
                             model, processor, tokenizer, max_num_frames, pos_tokens, neg_tokens, custom_prompt):
    if prompt_type=='full':
        template = Template(mode2prompt[eval_mode][aspect])
    else:
        template = Template(mode2prompt[eval_mode+prompt_type][aspect])
    prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template

    if eval_mode=='single_soft_custom':
        prompt = custom_prompt

    if max_num_frames>0:
        video_prefix = ""
        pixel_values, num_patches_list = [], []
        if eval_mode=="pairwise":
            vid_indices = ['The first video:\n', 'The second video:\n']
        elif eval_mode=="pairwise_no_vid_index":
            vid_indices = ['', '']
        for video_index, video_path in zip(vid_indices, video_paths):
            curr_pixel_values, curr_num_patches_list = load_video_internvl(video_path, num_segments=max_num_frames, max_num=1)
            curr_pixel_values = curr_pixel_values.to(torch.bfloat16).cuda()
            pixel_values.append(curr_pixel_values)
            num_patches_list += curr_num_patches_list
            video_prefix += video_index + ''.join([f'Frame{i+1}: <image>\n' for i in range(len(curr_num_patches_list))]) + '\n'
        pixel_values = torch.cat(pixel_values, dim=0)
    else:
        pixel_values, num_patches_list = None, []
        video_prefix = ""
    question = video_prefix + prompt
    
    generation_config = dict(max_new_tokens=5, do_sample=False)
    response, _ = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
    return response
