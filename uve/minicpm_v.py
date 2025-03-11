from string import Template
import torch.nn.functional as F
import os, copy, json, time
from .utils import *
from .constants import *

def eval_minicpm_v_single(video_path, video_prompt, aspect, eval_mode, prompt_type,
                        model, processor, tokenizer, max_num_frames,
                        system_prompt='', sampling=False, stream=False, max_slice_nums=2, use_image_id=False,
                        max_inp_length=8192, max_new_tokens=1):
    template = Template(mode2prompt[eval_mode][aspect])
    prompt = template.substitute(source=video_prompt) if aspect.startswith('tv_alignment') else template.template

    frames = encode_video_minicpmv(video_path, max_num_frames)
    msgs = [
        {'role': 'user', 'content': frames + [prompt]}, 
    ]

    if eval_mode.startswith('single_soft_yn'):
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = None
        
        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        assert model.config.query_num == processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert model.config.patch_size == processor.image_processor.patch_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert model.config.use_image_id == processor.image_processor.use_image_id, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert model.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert model.config.slice_mode == processor.image_processor.slice_mode, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        prompts_lists = []
        input_images_lists = []
        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = copy.deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert sampling or not stream, "if use stream mode, make sure sampling=True"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["user", "assistant"]
                if i == 0:
                    assert role == "user", "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                msg["content"] = "\n".join(cur_msgs)

            if system_prompt:
                sys_msg = {'role': 'system', 'content': system_prompt}
                copy_msgs = [sys_msg] + copy_msgs        

            prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
            input_images_lists.append(images)

        inputs = processor(
            prompts_lists, 
            input_images_lists, 
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            return_tensors="pt", 
            max_length=max_inp_length
        ).to(model.device)

        generation_config = {
            "num_beams": 3,
            "repetition_penalty": 1.2,
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_logits": True
        }

        inputs.pop("image_sizes")
        with torch.inference_mode():
            input_ids, pixel_values, image_bound, attention_mask, tgt_sizes = inputs["input_ids"], inputs["pixel_values"], inputs["image_bound"], inputs["attention_mask"], inputs["tgt_sizes"]
            assert input_ids is not None
            assert len(input_ids) == len(pixel_values)


            model_inputs = {
                "input_ids": input_ids,
                "image_bound": image_bound,
                "pixel_values": pixel_values,
                "tgt_sizes": tgt_sizes
            }
            model_inputs["inputs_embeds"], _ = model.get_vllm_embedding(model_inputs)

            terminators = [tokenizer.convert_tokens_to_ids(i) for i in model.terminators]
            output = model.llm.generate(
                inputs_embeds=model_inputs["inputs_embeds"],
                pad_token_id=0,
                eos_token_id=terminators,
                attention_mask=attention_mask,
                **generation_config
            )
            logits = output.logits
            probs = [F.softmax(logits_)[0] for logits_ in logits]
            yes_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()=='yes']
            no_ids = [id for v, id in tokenizer.get_vocab().items() if v.lower()=='no']
            yes_prob = sum([probs[0][id] for id in yes_ids])
            no_prob = sum([probs[0][id] for id in no_ids])
            yes_prob, no_prob = yes_prob / (yes_prob + no_prob), no_prob / (yes_prob + no_prob)
            return float(yes_prob.cpu())
    elif eval_mode=='single_hard':
        # Set decode params for video
        params={}
        params["use_image_id"] = use_image_id
        params["max_slice_nums"] = max_slice_nums # use 1 if cuda OOM and video resolution >  448*448
        params["sampling"] = sampling
        params["stream"] = stream
        params["max_new_tokens"] = max_new_tokens
        params["system_prompt"] = system_prompt
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        return answer