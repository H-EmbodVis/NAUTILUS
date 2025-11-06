import json
import argparse
import transformers
from transformers import AutoProcessor
import torch
import sys
from pathlib import Path
import re
from typing import List

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# from qwenvl.Nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import Qwen2_5_VL_Nautilus_ForConditionalGeneration
from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import Qwen2_5_VL_Nautilus_ForConditionalGeneration
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model

from qwen_vl_utils import process_vision_info
from qwenvl.train.utils import (find_all_linear_modules, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3)
from types import MethodType

image_token_id = 151655

def scale_bboxes_in_text(text: str, w_scale: float, h_scale: float) -> str:
    def scale_bbox(match):
        x1 = int(match.group(1))
        y1 = int(match.group(2))
        x2 = int(match.group(3))
        y2 = int(match.group(4))
        new_x1 = round(x1 * w_scale)
        new_y1 = round(y1 * h_scale)
        new_x2 = round(x2 * w_scale)
        new_y2 = round(y2 * h_scale)
        return f"[{new_x1}, {new_y1}, {new_x2}, {new_y2}]"

    pattern = r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]"
    scaled_text = re.sub(pattern, scale_bbox, text)
    return scaled_text

def get_grid_thw(processor, image_file):
    from PIL import Image
    image = Image.open(image_file).convert("RGB")
    width, height = image.size
    visual_processed = processor.preprocess(image, return_tensors="pt")
    image_tensor = visual_processed["pixel_values"]
    if isinstance(image_tensor, List):
        image_tensor = image_tensor[0]
    grid_thw = visual_processed["image_grid_thw"][0]
    return grid_thw, width, height

def double_image_tokens(inputs: dict, image_token_id: int) -> torch.Tensor:
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    input_ids = input_ids.squeeze(0)  # flatten to 1D
    attention_mask = attention_mask.squeeze(0)
    new_ids, new_mask = [], []

    for token,mask in zip(input_ids, attention_mask):
        new_ids.append(token.item())
        new_mask.append(mask.item())
        if token.item() == image_token_id:
            new_ids.append(token.item())  
            new_mask.append(mask.item())

    return torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0),torch.tensor(new_mask, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0)

argparse = argparse.ArgumentParser()
argparse.add_argument('--checkpoint', type=str, help='Path to the checkpoint directory.')
argparse.add_argument('--image', type=str, help='Path to the image')
argparse.add_argument('--prompt', type=str, help='Prompt', default="Describe the image")
args = argparse.parse_args()


# Load Model
checkpoint = args.checkpoint
model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
            checkpoint,
            cache_dir=None,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )

# Set Min/Max Pixel Size
min_pixels = 1 * 28 * 28
max_pixels = 1338 * 28 * 28

# Use Base Model's Processor 
# Ensure that the checkpoint directory includes a preprocessor_config.json file,
# consistent with the one from the Qwen2.5-VL variant. This configuration file is already provided in our checkpoint.
processor = AutoProcessor.from_pretrained(checkpoint, min_pixels=min_pixels, max_pixels=max_pixels)
image_processor = processor.image_processor

image_path = args.image
prompt = args.prompt 

# Scale the bounding box for region captioning or other region-based questions that include pixel coordinates in the prompt.
grid_thw, ori_w, ori_h = get_grid_thw(image_processor, image_path)
input_height = grid_thw[1].item() * 14
input_width = grid_thw[2].item() * 14
scale_h, scale_w = input_height / ori_h, input_width / ori_w
prompt = scale_bboxes_in_text(prompt, scale_w, scale_h)

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

# Preprocess inputs      
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

inputs['input_ids'], inputs['attention_mask']= double_image_tokens(inputs , image_token_id)
inputs = inputs.to(model.device)

# infer
generated_ids = model.generate(**inputs, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
res_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

# Output
print(res_text)