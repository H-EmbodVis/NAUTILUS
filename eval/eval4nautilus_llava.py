import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
from pathlib import Path

current_file = Path(__file__)
parent_dir = current_file.parent.parent
model_dir = parent_dir / "LLaVA"
sys.path.append(str(model_dir))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


from Evaluation_pack.classification import evaluate_classification_metrics 
from Evaluation_pack.count import evaluate_count_metrics
from Evaluation_pack.detection import evaluate_detection_metrics
from Evaluation_pack.grounding import evaluate_ground_metrics
from Evaluation_pack.text import evaluate_text_metrics

from utils import sortbyid

Metric_method_dict = {
    "Counting":evaluate_count_metrics,
    "Detection":evaluate_detection_metrics,
    "Fishnet_Classification":evaluate_classification_metrics,
    "Grounding":evaluate_ground_metrics,
    "Image_caption":evaluate_text_metrics,
    "VQA":evaluate_text_metrics,
    "Region_caption":evaluate_text_metrics,
    "Region_Classification":evaluate_classification_metrics,
    "Zero_shot_Grounding":evaluate_ground_metrics
}

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

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
    return load_pretrained_model(model_path, args.model_base, model_name,args = args)

def eval_with_predictions(prediction_json, questions):
    ans_list = json.load(open(prediction_json, "r"))
    try:
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    except:
        print(f"Question:{questions}")
        exit()
    return sortbyid(ans_list), sortbyid(questions)
def eval_with_infer(args, questions,model_args,case_num=1e6):
    # Model
    tokenizer, model, image_processor, contexget_model_name_from_patht_len = model_args
    model_path = os.path.expanduser(args.model_path)
    model_name = (model_path)

    # model = model.cuda().eval()

    try:
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    except:
        print(f"Question:{questions}")
        exit()

    ans_list = []
    idx  = 1
    count = 0
    for line in tqdm(questions):
        count += 1
        if count > case_num:
            break
        idx = line["id"]
        image_file = line["image"]
        
        qs = line["conversations"][0]['value'].replace("<image>",'').strip()
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

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
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
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_list.append({"id": idx,
                        "prompt": cur_prompt,
                        "gt":line["conversations"][1]['value'],
                        "text": outputs,
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {}})

    return sortbyid(ans_list), sortbyid(questions), ans_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoint")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./nautdata_images")
    parser.add_argument("--answer-folder", type=str, help='Where to save prediction', default="./Answer")
    parser.add_argument('--metric-folder', type=str, help='Path to the Metric folder.', default="./Metric")
    parser.add_argument("--dinov2-weight", type=str, default="./dino_vitl.pth")
    parser.add_argument("--benchmark-json", type=str, default="./test.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--grounding-only', action='store_true', help='Only eval on grounding part')
    parser.add_argument('--use-predictions', action='store_true', help='Use predictions to evaluate')
    parser.add_argument('--prediction-json', type=str, help='Prediction json file', default=None)
    args = parser.parse_args()
    
    ### Setting
    benchmark_json = args.benchmark_json
    Metric_dict = {}
    output_name = os.path.basename(args.model_path)
    model_type = "LLaVA"
    metric_folder = args.metric_folder
    answer_folder = args.answer_folder
    os.makedirs(metric_folder, exist_ok=True)

    # Load questions
    questions = json.load(open(benchmark_json,"r"))
    if args.grounding_only:
        questions = sortbyid(questions)["Grounding"]

    if not args.use_predictions:
        # Eval model type

        model_args = get_model(args)

        # Inference and Eval
        Answer_dict, GT_dict, prediction_list = eval_with_infer(args, questions, model_args)

        # Answer saving
        os.makedirs(answer_folder, exist_ok=True)
        save_root = os.path.join(answer_folder,f"{output_name}.json")
        json.dump(prediction_list, open(save_root,"w"), indent=2)
    else:
        assert args.prediction_json is not None, "Please specify prediction json file"

        Answer_dict, GT_dict = eval_with_predictions(args.prediction_json, questions)

    # Metric calculation
    kwargs= {"model_type" : model_type, "image_folder" : args.image_folder, "vqa_acc" : False, "java" : True}
    for task_name in Answer_dict.keys():
        eval_method = Metric_method_dict[task_name]
        Metric_dict[task_name] = eval_method(Answer_dict[task_name],GT_dict[task_name],kwargs)
        print(Metric_dict[task_name])

    
    ### Result's saving and show

    for idx,(ben_name , metric) in enumerate(Metric_dict.items()):
        print('-' * 10 + f' Benchmark {idx+1} ' + '-' * 10)
        print(f"{ben_name}'s results:")
        for k,v in metric.items():
                print(f"{k}:{v}")

    output_root = f"{metric_folder}/{output_name}.json"
    json.dump(Metric_dict,open(output_root,"w"),indent=2)
    
