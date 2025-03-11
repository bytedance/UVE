from string import Template
import torch.nn.functional as F
from .utils import *
from .constants import *
import os, copy, json, time
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token

def eval_llava_video_single(video_path, video_prompt, aspect, eval_mode, prompt_type,
                            model, processor, tokenizer, max_num_frames):
    template = Template(mode2prompt[eval_mode][aspect])
    prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template

    video,frame_time,video_time = load_video_llava_video(video_path, max_num_frames, 1, force_sample=True)
    video = processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n{prompt}"

    from llava.conversation import conv_templates
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    if eval_mode.startswith('single_soft_yn'):
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=1,
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
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    
def eval_llava_video_pairwise(video_paths, video_prompt, aspect, eval_mode, prompt_type,
                            model, processor, tokenizer, max_num_frames):
    template = Template(mode2prompt[eval_mode][aspect])
    prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template

    videos, question = [], ""
    for video_index, video_path in zip(['The first video:\n', 'The second video:\n'], video_paths):
        video, frame_time, video_time = load_video_llava_video(video_path, max_num_frames, 1, force_sample=True)
        video = processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        videos.append(video)
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video)} frames are uniformly sampled from it. These frames are located at {frame_time}."
        question += f"{video_index}{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n"
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
            images=videos,
            modalities= ["video", "video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=5,
        )

    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return response