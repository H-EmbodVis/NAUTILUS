import argparse
import torch
import os
import sys
from PIL import Image
import math
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.full_model:
        model_name = "llava_full_model"
    return load_pretrained_model(model_path, args.model_base, model_name,args = args)

def single_batch_infer(args, model_args):
    # Model
    tokenizer, model, image_processor, contexget_model_name_from_patht_len = model_args

    # model = model.cuda().eval()

    questions = args.prompt
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    image_file = args.image
    
    qs = questions.replace("<image>",'').strip()
    cur_prompt = qs
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava_lora_checkpoint")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--full-model", action="store_true", help="load model with a merged checkpoint")
    parser.add_argument("--dinov2-weight", type=str, default="dino_vitl.pth")
    parser.add_argument('--image', type=str, help='Path to the image')
    parser.add_argument('--prompt', type=str, help='Prompt', default="Describe the image")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    

    # Load model
    model_args = get_model(args)
    
    # Inference 
    response = single_batch_infer(args, model_args)
    print(response)