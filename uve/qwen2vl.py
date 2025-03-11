from string import Template
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from moviepy.editor import VideoFileClip
from .constants import *
import os
from .utils import *

def eval_qwen2vl_single(video_path, video_prompt, aspect, eval_mode, prompt_type,
                        model, processor, tokenizer, max_num_frames,
                        pos_tokens, neg_tokens, custom_prompt):
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

    if max_num_frames>=2:
        clip = VideoFileClip(video_path)
        max_num_frames = min(max_num_frames, clip.duration*clip.fps)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 720 * 480,
                        "nframes": max_num_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    elif max_num_frames==1:
        image_path = os.path.join(os.path.dirname(video_path), 'center_frames', video_path.replace('.mp4', '.jpg'))
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        sample_middle_frame(video_path, image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    elif max_num_frames==0:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]        

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    if eval_mode.startswith('single_soft_yn'):
        logits = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True).logits
        probs = [F.softmax(logits_)[0] for logits_ in logits]
        yes_ids = [id for v, id in processor.tokenizer.get_vocab().items() if v.lower()=='yes']
        no_ids = [id for v, id in processor.tokenizer.get_vocab().items() if v.lower()=='no']
        yes_prob = sum([probs[0][id] for id in yes_ids])
        no_prob = sum([probs[0][id] for id in no_ids])
        yes_prob, no_prob = yes_prob / (yes_prob + no_prob), no_prob / (yes_prob + no_prob)
        return float(yes_prob.cpu())
    elif eval_mode == 'single_soft_good_bad':
        logits = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True).logits
        probs = [F.softmax(logits_)[0] for logits_ in logits]
        good_ids = [id for v, id in processor.tokenizer.get_vocab().items() if v.lower()=='good']
        bad_ids = [id for v, id in processor.tokenizer.get_vocab().items() if v.lower()=='bad']
        good_prob = sum([probs[0][id] for id in good_ids])
        bad_prob = sum([probs[0][id] for id in bad_ids])
        good_prob, bad_prob = good_prob / (good_prob + bad_prob), bad_prob / (good_prob + bad_prob)
        return float(good_prob.cpu())
    elif eval_mode == 'single_soft_adaptive':
        logits = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True).logits
        probs = [F.softmax(logits_)[0] for logits_ in logits]
        good_ids = [id for v, id in processor.tokenizer.get_vocab().items() if v.lower()==asepct2scoretokens[aspect][0]]
        bad_ids = [id for v, id in processor.tokenizer.get_vocab().items() if v.lower()==asepct2scoretokens[aspect][1]]
        good_prob = sum([probs[0][id] for id in good_ids])
        bad_prob = sum([probs[0][id] for id in bad_ids])
        good_prob, bad_prob = good_prob / (good_prob + bad_prob), bad_prob / (good_prob + bad_prob)
        return float(good_prob.cpu())
    elif eval_mode == 'single_soft_custom':
        logits = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True).logits
        probs = [F.softmax(logits_)[0] for logits_ in logits]
        good_ids = [id for v, id in processor.tokenizer.get_vocab().items() if v.lower() in pos_tokens]
        bad_ids = [id for v, id in processor.tokenizer.get_vocab().items() if v.lower() in neg_tokens]
        good_prob = sum([probs[0][id] for id in good_ids])
        bad_prob = sum([probs[0][id] for id in bad_ids])
        good_prob, bad_prob = good_prob / (good_prob + bad_prob), bad_prob / (good_prob + bad_prob)
        return float(good_prob.cpu())
    elif eval_mode == 'single_hard':
        generated_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


def eval_qwen2vl_pairwise(video_pair_paths, video_prompt, aspect, eval_mode, prompt_type,
                          model, processor, tokenizer, max_num_frames, pos_tokens, neg_tokens, custom_prompt):
    if prompt_type=='full':
        template = Template(mode2prompt[eval_mode][aspect])
    else:
        template = Template(mode2prompt[eval_mode+prompt_type][aspect])
    prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template
    if eval_mode=='single_soft_custom':
        assert custom_prompt is not None
        prompt = custom_prompt

    clips = [VideoFileClip(path) for path in video_pair_paths]
    for clip in clips:
        max_num_frames = min(max_num_frames, clip.duration*clip.fps)
    fps = [max_num_frames / clip.duration for clip in clips]

    if max_num_frames>=2:
        if eval_mode=='pairwise':
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "The first video:\n"},
                        {
                            "type": "video",
                            "video": video_pair_paths[0],
                            "max_pixels": 720 * 480,
                            "fps": fps[0],
                        },
                        {"type": "text", "text": "\n\nThe second video:\n"},
                        {
                            "type": "video",
                            "video": video_pair_paths[1],
                            "max_pixels": 720 * 480,
                            "fps": fps[1],
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        elif eval_mode=='pairwise_no_vid_index':
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_pair_paths[0],
                            "max_pixels": 720 * 480,
                            "fps": fps[0],
                        },
                        {
                            "type": "video",
                            "video": video_pair_paths[1],
                            "max_pixels": 720 * 480,
                            "fps": fps[1],
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
    elif max_num_frames==1:
        image_paths = []
        for video_path in video_pair_paths:
            image_path = os.path.join(os.path.dirname(video_path), "center_frames", os.path.basename(video_path).replace(".mp4", ".png"))
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            sample_middle_frame(video_path, image_path)
            image_paths.append(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The first video:\n"},
                    {
                        "type": "image",
                        "image": image_paths[0],
                    },
                    {"type": "text", "text": "\n\nThe second video:\n"},
                    {
                        "type": "image",
                        "image": image_paths[1],
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    elif max_num_frames==0:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text