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

from .utils import *
import importlib

class UVE(object):
    def __init__(self, model_name, max_num_frames, local_model_path=None,
                 base_url=None, api_version=None, ak=None):
        """
            local_model_path: local path to load mllms
        """
        self.model_name = model_name
        self.model, self.processor, self.tokenizer = load_model(model_name, local_model_path, base_url, ak)
        self.max_num_frames = max_num_frames
        self.local_model_path = local_model_path
        print("-"*100)
        print(f"Configuration: \n model={model_name}\n max_num_frames={max_num_frames}")
        print("-"*100)
    
    def check_parameters(self, video_path, model_name, eval_mode, aspect, video_prompt, prompt_type):
        """
            Check whether the parameter combinations are valid
        """
        if "single" in eval_mode:
            assert isinstance(video_path, str)
        if "pair" in eval_mode:
            assert isinstance(video_path, list) and len(video_path)==2
        if aspect.startswith("tv_alignment"):
            assert video_prompt is not None, "Video prompt is required for TV alignment evaluation."
        if not any([model_name.startswith(x) for x in ['qwen2-vl', 'internvl-2.5', 'llava-video', 'llava-onevision', 'gpt4o']]):
            assert eval_mode.startswith('single'), f"Only support pariwise evaluation for {['qwen2-vl', 'internvl-2.5', 'llava-video', 'llava-onevision', 'gpt4o']}, but the current model is {model_name}"
        if not any([model_name.startswith(x) for x in ['qwen2-vl', 'internvl-2.5']]):
            assert prompt_type.startswith('full'), f"Only support none full prompt for {['qwen2-vl', 'internvl-2.5']}, but the current model is {model_name}"
        if not model_name.startswith('videoscore'):
            assert eval_mode not in ['single_soft_reg-avg', 'single_soft_reg-dim'], f"Only support {eval_mode} for videoscore, but the current model is {model_name}"

    def evaluate(self, video_path, video_prompt=None, aspect='dynamic_degree', eval_mode='single_soft_yn', prompt_type='full', pos_tokens=None, neg_tokens=None, custom_prompt=None):

        self.check_parameters(video_path, self.model_name, eval_mode, aspect, video_prompt, prompt_type)

        if eval_mode=='single_soft_custom':
            assert pos_tokens is not None and neg_tokens is not None and custom_prompt is not None
            pos_tokens = pos_tokens.split(',') if isinstance(pos_tokens, str) else pos_tokens
            neg_tokens = neg_tokens.split(',') if isinstance(neg_tokens, str) else neg_tokens

        module_name = model2module[self.model_name]
        try:
            model_module = importlib.import_module(f"uve.{module_name}")
            evaluate_func = getattr(model_module, f"eval_{module_name}_single") if "single" in eval_mode else getattr(model_module, f"eval_{module_name}_pairwise")
        except Exception as e:
            raise NotImplementedError(f'UnImplemented model {self.model_name}!, {e}')
        
        return evaluate_func(video_path, video_prompt, aspect, eval_mode, prompt_type, self.model, self.processor, self.tokenizer, self.max_num_frames, pos_tokens, neg_tokens, custom_prompt)