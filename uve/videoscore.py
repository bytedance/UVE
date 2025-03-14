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

import av
import numpy as np
from typing import List
from PIL import Image
import torch
from .utils import *
from .constants import *

ROUND_DIGIT=3
REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge
for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score, 
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8
For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""

def eval_videoscore_single(video_path, video_prompt, aspect, eval_mode, prompt_type,
                            model, processor, tokenizer, max_num_frames,
                            pos_tokens, neg_tokens, custom_prompt):
    assert custom_prompt is None, "custom_prompt is not supported for VideoScore."
    assert aspect in aspect2videoscoredim, f"VideoScore only supports {list(aspect2videoscoredim.keys())}, but the aspect is {aspect}"
    assert eval_mode in ["single_soft_reg-avg", "single_soft_reg-dim"], f"VideoScore only supports {['single_soft_reg-avg', 'single_soft_reg-dim']}, but the eval_mode is {eval_mode}"

    # sample uniformly 8 frames from the video
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames > max_num_frames:
        indices = np.arange(0, total_frames, total_frames / max_num_frames).astype(int)
    else:
        indices = np.arange(total_frames)

    frames = [Image.fromarray(x) for x in read_video_pyav(container, indices)]
    eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)
    num_image_token = eval_prompt.count("<image>")
    if num_image_token < len(frames):
        eval_prompt += "<image> " * (len(frames) - num_image_token)
    flatten_images = []
    for x in [frames]:
        if isinstance(x, list):
            flatten_images.extend(x)
        else:
            flatten_images.append(x)

    flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
    inputs = processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    """
    model output on visual quality, temporal consistency, dynamic degree,
    text-to-video alignment, factual consistency, respectively
    VideoScore: 
    [2.297, 2.469, 2.906, 2.766, 2.516]

    VideoScore-v1.1:
    [2.328, 2.484, 2.562, 1.969, 2.594]
    """
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    num_aspects = logits.shape[-1]
    aspect_scores = {}
    for i in range(num_aspects):
        dim = list(videoscoredim2aspect.keys())[i]
        aspect_scores[dim] = round(logits[0, i].item(), ROUND_DIGIT)
    if eval_mode == "single_soft_reg-avg":
        return np.mean(list(aspect_scores.values()))
    elif eval_mode == "single_soft_reg-dim":
        return aspect_scores[aspect2videoscoredim[aspect]]
