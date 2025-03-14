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

import importlib, torch, os, dataclasses, math
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip
from decord import VideoReader, cpu
from typing import Dict, List, Tuple, Union
from enum import IntEnum, auto
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel
from mantis.models.idefics2 import Idefics2ForSequenceClassification
from llava.model.builder import load_pretrained_model
import cv2, base64
from openai import OpenAI
from .constants import *


def load_model(model_name, local_model_path, base_url, ak):
    if model_name == "gpt4o":
        assert (ak is not None) and (base_url is not None), "Please set your OpenAI API key and base_url for using GPT-4o."
        client = OpenAI(
            api_key=ak,
            base_url=base_url,
        )
        return client, None, None
    
    if local_model_path and os.path.exists(f"{local_model_path}/{model2name[model_name]}"):
        model_path = f"{local_model_path}/{model2name[model_name]}"
    else:
        model_path = model2name[model_name]
    use_flash_attn = True if importlib.util.find_spec('flash_attn') else False
    print(f"Loading MLLM from {model_path}... \nflash_attn: {use_flash_attn}")
    if "Qwen2-VL" in model_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash_attn else 'sdpa',
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor, None   # model, processor, tokenizer
    elif "InternVL2_5" in model_path:
        if '78B' in model_path:
            print("Mapping model to multiple GPUs....")
            device_map = split_model('InternVL2_5-78B')
            model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=use_flash_attn, trust_remote_code=True, device_map=device_map).eval()
        else:
            model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, load_in_8bit=False, use_flash_attn=use_flash_attn, trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        return model, None, tokenizer
    elif "llava-onevision" in model_path:
        device_map = "auto"
        llava_model_args = {
            "multimodal": True,
        }
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, "llava_qwen", device_map=device_map, attn_implementation="sdpa", **llava_model_args)
        model.eval()
        return model, image_processor, tokenizer
    elif "LLaVA-Video" in model_path:
        device_map = "auto"
        use_flash_attn = True if importlib.util.find_spec('flash_attn') else False
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, "llava_qwen", torch_dtype="bfloat16", 
                                                                            device_map=device_map, attn_implementation='flash_attention_2' if use_flash_attn else 'sdpa')  # Add any other thing you want to pass in llava_model_args
        model.eval()
        return model, image_processor, tokenizer
    elif "MiniCPM-V" in model_path:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model, processor, tokenizer
    elif "VideoScore" in model_path:
        processor = AutoProcessor.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        model = Idefics2ForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, processor, None

# Qwen2-VL
def sample_middle_frame(video_path, image_path):
    if os.path.exists(image_path):
        return
    video = VideoFileClip(video_path)

    middle_frame_time = video.duration / 2
    middle_frame = video.get_frame(middle_frame_time)

    image = Image.fromarray(np.uint8(middle_frame))

    image.save(image_path)

# InterVL-2.5
class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()

@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = '{system_message}'
    # The system message
    system_message: str = ''
    # The names of two roles
    roles: Tuple[str] = ('USER', 'ASSISTANT')
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = '\n'
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ': '  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = '' if system_prompt == '' else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ': '
                        + message.replace('\r\n', '\n').replace('\n\n', '\n')
                    )
                    ret += '\n\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = '[INST] '
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + ' '
                    else:
                        ret += tag + ' ' + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == 'chatglm2' else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ''

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f'[Round {i//2 + round_add_n}]{self.sep}'

                if message:
                    ret += f'{role}：{message}{self.sep}'
                else:
                    ret += f'{role}：'
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = '' if system_prompt == '' else system_prompt + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep + '\n'
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ''
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + ' ' + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ':' + message + seps[i % 2] + '\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ':\n' + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += '\n\n'
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + '<s>' + message + '</s>'
                else:
                    ret += role + ': ' + '<s>'
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ':\n' + message + self.sep
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ''
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f'Invalid style: {self.sep_style}')

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{'role': 'system', 'content': self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({'role': 'user', 'content': msg})
            else:
                if msg is not None:
                    ret.append({'role': 'assistant', 'content': msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            'template_name': self.name,
            'system_message': self.system_message,
            'roles': self.roles,
            'messages': self.messages,
            'offset': self.offset,
        }

conv_templates: Dict[str, Conversation] = {}
def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()

def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f'{template.name} has been registered.'

    conv_templates[template.name] = template

register_conv_template(
    Conversation(
        name='internvl2_5',
        system_template='<|im_start|>system\n{system_message}',
        system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>\n',
    )
)

def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_video_internvl(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# LLaVA-OneVision
def load_video_llava_onevision(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

# LLaVA-Video
def load_video_llava_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

# MiniCPM-V
def encode_video_minicpmv(video_path, max_num_frames):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    # sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    # frame_idx = [i for i in range(0, len(vr), sample_fps)]
    frame_idx = [i for i in range(0, len(vr))]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    # print(f'num frames: {len(frames)}')
    return frames

# GPT-4o
def split_video_into_frames(video_path, video_frame_path, n_parts=16):
    video = cv2.VideoCapture(video_path)
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_part = total_frames // n_parts
    
    for i in range(n_parts):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frames_per_part)
        ret, frame = video.read()
        if not ret:
            print("Unable to read video frame。")
            break
        
        frame_number = i * frames_per_part
        cv2.imwrite(f"{video_frame_path}/frame_{frame_number}.jpg", frame)
    
    video.release()

def encode_video_gpt(video_path, video_frame_path, num_frames):
    os.makedirs(video_frame_path, exist_ok=True)
    frame_files = [f"{video_frame_path}/{ff}" for ff in os.listdir(video_frame_path) if ff.endswith('.jpg')]
    frame_files = sorted(frame_files, key=lambda x: int(x.replace('.jpg', '').split('_')[-1]))
    encoded_frms = []
    if len(frame_files)==0:
        split_video_into_frames(video_path, video_frame_path, num_frames)
    frame_files = [f"{video_frame_path}/{ff}" for ff in os.listdir(video_frame_path) if ff.endswith('.jpg')]
    frame_files = sorted(frame_files, key=lambda x: int(x.replace('.jpg', '').split('_')[-1]))
    for ff in frame_files:
        encoded_frms.append(encode_image_gpt(ff))
    return encoded_frms

def encode_image_gpt(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def get_response_gpt(client, encoded_frms, prompt, sys_prompt, model, max_tokens=1, return_prob=False):
    content = []
    if len(encoded_frms)>1:
        for fid, encoded_frm in enumerate(encoded_frms):
            # content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frm}", "detail": "low"}})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frm}"}})
    else:
        # content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frms[0]}", "detail": "low"}})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frms[0]}"}})
    content.append({"type": "text", "text": prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=max_tokens,
        logprobs=True if return_prob else None,
        top_logprobs=10 if return_prob else None,
        temperature=0,
        extra_headers={"X-TT-LOGID": "${your_logid}"},
    )
    output = completion.choices[0].message.content
    if return_prob:
        logprobs = {logprob.token: logprob.logprob for logprob in completion.choices[0].logprobs.content[0].top_logprobs}
        probs = {token: math.exp(logprobs[token]) for token in logprobs}
        yes_probs = [prob for token, prob in probs.items() if token.lower().strip().startswith('yes')]
        no_probs = [prob for token, prob in probs.items() if token.lower().strip().startswith('no')]
        if len(yes_probs)==0 and len(no_probs)==0:
            return None
        yes_prob, no_prob = sum(yes_probs), sum(no_probs)
        yes_prob, no_prob = yes_prob / (yes_prob + no_prob), no_prob / (yes_prob + no_prob)
        return yes_prob
    else:
        return 1 if output.lower().startswith('yes') else 0
    
def get_response_gpt_video_pair(client, encoded_vids, prompt, sys_prompt, model, max_tokens=5):
    content = []
    for vid_prefix, encoded_frms in zip(["The first video:\n", "The second video:\n"], encoded_vids):
        content.append({"type": "text", "text": vid_prefix})
        for encoded_frm in encoded_frms:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frm}"}})
    content.append({"type": "text", "text": prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=max_tokens,
        logprobs=None,
        top_logprobs=None,
        temperature=0,
        extra_headers={"X-TT-LOGID": "${your_logid}"},
    )
    output = completion.choices[0].message.content
    return output

# VideoScore
def read_video_pyav(
    container, 
    indices
):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])
