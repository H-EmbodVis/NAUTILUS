import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
from pathlib import Path

current_file = Path(__file__)
parent_dir = current_file.parent.parent
model_dir = parent_dir / "qwen-vl-finetune"
sys.path.append(str(model_dir))

from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import Qwen2_5_VL_Nautilus_ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from Evaluation_pack.classification import evaluate_classification_metrics 
from Evaluation_pack.count import evaluate_count_metrics
from Evaluation_pack.detection import evaluate_detection_metrics
from Evaluation_pack.grounding import evaluate_ground_metrics
from Evaluation_pack.text import evaluate_text_metrics

from utils import get_grid_thw, sortbyid, scale_bboxes_in_text, double_image_tokens, image_token_id

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

def eval_with_predictions(prediction_json, questions):
    ans_list = json.load(open(prediction_json, "r"))
    return sortbyid(ans_list), sortbyid(questions)

def eval_with_infer(questions, model, processor, image_processor):
    # Model
    prediction_list = []
    for data in tqdm(questions):
        image_path = os.path.join(base_data_folder,data['image'])
        prompt = data['conversations'][0]['value'].replace("<image>","").strip()

        # For Region_caption and Region_Classification prompt bbox should be resized according to grid_thw
        if data['id'][1] in ["2","7"]:
            grid_thw, ori_w, ori_h = get_grid_thw(image_processor, image_path)
            i_height = grid_thw[1].item()*14
            i_width = grid_thw[2].item()*14
            scale_h, scale_w = i_height / ori_h, i_width / ori_w
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

        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14

        # for Nautilus model.
        inputs['input_ids'], inputs['attention_mask']= double_image_tokens(inputs,image_token_id)

        inputs = inputs.to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        res_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        prediction_list.append({
            "id":data['id'],
            "gt":data['conversations'][1]['value'],
            "text":res_text,
            "prompt":prompt,
            "input_height":input_height.item(),
            "input_width":input_width.item()
        })
    

    return sortbyid(prediction_list), sortbyid(questions), prediction_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to the input model', required=True)
    parser.add_argument("--answer-folder", type=str, help='Where to save prediction', default="./Answer")
    parser.add_argument('--metric-folder', type=str, help='Path to the Metric folder.', default="./Metric")
    parser.add_argument('--data-path', type=str, help='Path to the image.', default="./nautdata_images")
    parser.add_argument('--benchmark-json', type=str, help='Path to the benchmark.',default="./Nautilus-instruct-test.json")
    parser.add_argument('--num-workers', type=int, help='num_workers',default=2)
    parser.add_argument('--grounding-only', action='store_true', help='Only eval on grounding part')
    parser.add_argument('--max-pixel', type=int, help='max_pixel of qwen2_5_vl',default=1338)
    parser.add_argument('--use-predictions', action='store_true', help='Use predictions to evaluate')
    parser.add_argument('--prediction-json', type=str, help='Prediction json file', default=None)
    args = parser.parse_args()
    
    ### Setting
    checkpoint = args.checkpoint
    output_name = os.path.basename(checkpoint)
    model_type = "Qwen"
    benchmark_json = args.benchmark_json
    base_data_folder = args.data_path
    metric_folder = args.metric_folder
    Metric_dict = {}
    metric_folder = args.metric_folder
    answer_folder = args.answer_folder
    # output_folder = checkpoint.split("/")[-2]
    # metric_folder = os.path.join(metric_folder, output_folder)
    # answer_folder = os.path.join(answer_folder, output_folder)
    os.makedirs(metric_folder, exist_ok=True)

    if not benchmark_json.endswith(".json"):
        raise ValueError(f"Not a json file: {benchmark_json}")

    questions = json.load(open(benchmark_json,"r"))
    if args.grounding_only:
        questions = sortbyid(questions)["Grounding"]

    if not args.use_predictions:
        ## Model
        model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
                    checkpoint,
                    cache_dir=None,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map=None,
                )
                
        model.to(torch.device("cuda"))
        model.eval()
        #  1338 * 28 * 28
        min_pixels = 1*28*28
        max_pixels = args.max_pixel * 28 * 28
        print(f"max_pixels:{args.max_pixel} * 28 * 28")
        processor = AutoProcessor.from_pretrained(checkpoint, min_pixels=min_pixels, max_pixels=max_pixels)
        image_processor = processor.image_processor

        # Inference
        Answer_dict, GT_dict, prediction_list= eval_with_infer(questions, model, processor, image_processor)

        # Answer's saving
        os.makedirs(answer_folder, exist_ok=True)
        save_root = os.path.join(answer_folder, f"{output_name}.json")
        json.dump(prediction_list, open(save_root,"w"), indent=2)
    
    else:
        assert args.prediction_json is not None, "Please specify prediction json file"

        Answer_dict, GT_dict = eval_with_predictions(args.prediction_json, questions)

    # Metric calculation
    kwargs= {"model_type" : model_type, "image_folder" : args.data_path, "vqa_acc" : False, "java" : False}
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