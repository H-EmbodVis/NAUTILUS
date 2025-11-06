import json
import os
import numpy as np
import re
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

def clear_special_tokens(in_str):
    return in_str.replace("</ref>","").replace("<ref>","").replace("</box>","").replace("<box>","")

def parse_bboxes(bbox_str):
    try:
        bbox_dict = {}
        lines = bbox_str.strip().split('\n')
        for line in lines:
            try:
                line = clear_special_tokens(line)
                if "[[" in line:
                    category, bbox_coords_str = line.split("[[")
                    bbox_coords_str = "[" + bbox_coords_str[:-1]
                else:
                    category, bbox_coords_str = line.split(':')
            except ValueError:
                continue
            category = re.sub(r'\d+', '', category).lower()
            bbox_coords_str = bbox_coords_str.strip().strip('[]')
            bbox_coords = [float(coord) for coord in bbox_coords_str.split(',') if coord.strip()]

            # For Internvl format bbox
            if max(bbox_coords) > 1:
                    bbox_coords = [float(coord/1000) for coord in bbox_coords]

            if len(bbox_coords) != 4:
                continue
                raise ValueError(f"Bounding box should contain 4 values, but got {bbox_coords}")
            if category in bbox_dict:
                bbox_dict[category].append(bbox_coords)
            else:
                bbox_dict[category] = [bbox_coords]
        return bbox_dict
    except Exception as e:
        print(f"Error parsing bounding box string: {bbox_str}. Error: {e}")
        return None

# qwen format bbox
def parse_bboxes_qwen(bbox_str):
    try:
        bbox_dict = {'default':[]}
        bbox_json_str_list = json.loads(f"[{bbox_str}]")
        for bbox_json in bbox_json_str_list:
            try:
                category, bbox_coords_int = bbox_json["label"],bbox_json["bbox_2d"]
            except ValueError:
                print(bbox_str)
                continue
            bbox_coords = [float(coord) for coord in bbox_coords_int]
            if len(bbox_coords) != 4:
                print(bbox_str)
                raise ValueError(f"Bounding box should contain 4 values, but got {bbox_coords}")
            if category in bbox_dict:
                bbox_dict[category].append(bbox_coords)
            else:
                bbox_dict[category] = [bbox_coords]
        return bbox_dict
    except Exception as e:
        print(f"Error parsing bounding box string: {bbox_str}. Error: {e}")
        return None

def convert_to_coco_format(gt_map, pred_map, category_name_to_id, image_id_to_path, base_path, model_type ="LLaVA"):

    gt, preds = [], []
    gt_id, pred_id = 1, 1
    image_ids = set()

    for image_id, boxes in gt_map.items():
        image_ids.add(image_id)
        image_path = image_id_to_path.get(str(image_id))
        if image_path:
            full_image_path = os.path.join(base_path, image_path)
            try:
                image = Image.open(full_image_path)
                img_width, img_height = image.size
            except Exception as e:
                print(f"Error opening image {full_image_path}: {e}")
                continue
        else:
            print(f"Image path for image_id {image_id} not found.")
            continue
        
        for category_name, box_list in boxes.items():
            category_name = category_name.lower()
            category_id = category_name_to_id[category_name]
            for bbox in box_list:
                if model_type == "LLaVA":
                    x_min, y_min, x_max, y_max = [
                        coord * img_width if i % 2 == 0 else coord * img_height 
                        for i, coord in enumerate(bbox)
                    ]
                elif model_type == "Qwen":
                    x_min, y_min, x_max, y_max = bbox
                width, height = x_max - x_min, y_max - y_min
                area = width * height
                gt.append({
                    "id": gt_id, "image_id": image_id, "category_id": category_id,
                    "bbox": [x_min, y_min, width, height], "area": area, "iscrowd": 0
                })
                gt_id += 1

    for image_id, pred in pred_map.items():
        if not pred or (image_id not in image_ids):
            continue

        image_path = image_id_to_path.get(str(image_id))
        if image_path:
            full_image_path = os.path.join(base_path, image_path)
            try:
                image = Image.open(full_image_path)
                img_width, img_height = image.size
            except Exception as e:
                print(f"Error opening image {full_image_path}: {e}")
                continue
        else:
            print(f"Image path for image_id {image_id} not found.")
            continue

        if model_type == "Qwen":
            boxes = pred["bbox"]
            input_width, input_height = pred['input_width'], pred['input_height']
        elif model_type == "LLaVA":
            boxes = pred
        else:
            raise ValueError(f"Model type {model_type} not supported.")
        for category_name, box_list in boxes.items():
            category_name = category_name.lower()
            category_id = category_name_to_id.get(category_name)
            if category_id is None:
                continue
            for bbox in box_list:
                if model_type == "LLaVA":
                    x_min, y_min, x_max, y_max = [
                        coord * img_width if i % 2 == 0 else coord * img_height 
                        for i, coord in enumerate(bbox)
                    ]
                else:
                    x_min, y_min, x_max, y_max = [
                        coord * img_width / input_width if i % 2 == 0 else coord * img_height / input_height
                        for i, coord in enumerate(bbox)
                    ]

                width, height = x_max - x_min, y_max - y_min
                area = width * height
                preds.append({
                    "id": pred_id, "image_id": image_id, "category_id": category_id,
                    "bbox": [x_min, y_min, width, height], "area": area, "score": 0.9
                })
                pred_id += 1

    return gt, preds

def get_evaluation_metrics(coco_eval):

    metrics = {}

    # Average Precision (AP)
    metrics['AP_50_95'] = float(coco_eval.stats[0])  # AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
    metrics['AP_50'] = float(coco_eval.stats[1])     # AP @[ IoU=0.50 | area=all | maxDets=100 ]
    metrics['AP_75'] = float(coco_eval.stats[2])     # AP @[ IoU=0.75 | area=all | maxDets=100 ]
    metrics['AP_small'] = float(coco_eval.stats[3])  # AP @[ IoU=0.50:0.95 | area=small | maxDets=100 ]
    metrics['AP_medium'] = float(coco_eval.stats[4]) # AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
    metrics['AP_large'] = float(coco_eval.stats[5])  # AP @[ IoU=0.50:0.95 | area=large | maxDets=100 ]
    
    # Average Recall (AR) 
    metrics['AR_50_95_1'] = float(coco_eval.stats[6])  # AR @[ IoU=0.50:0.95 | area=all | maxDets=1 ]
    metrics['AR_50_95_10'] = float(coco_eval.stats[7]) # AR @[ IoU=0.50:0.95 | area=all | maxDets=10 ]
    metrics['AR_50_95_100'] = float(coco_eval.stats[8]) # AR @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
    metrics['AR_small_100'] = float(coco_eval.stats[9])  # AR @[ IoU=0.50:0.95 | area=small | maxDets=100 ]
    metrics['AR_medium_100'] = float(coco_eval.stats[10]) # AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
    metrics['AR_large_100'] = float(coco_eval.stats[11])  # AR @[ IoU=0.50:0.95 | area=large | maxDets=100 ]
    
    return metrics

def evaluate_detection_metrics(predictions, ground_truth, kwargs):
    image_id_to_path = {item["id"][1:]: item["image"] for item in ground_truth}

    gt_map, pred_map = {}, {}
    model_type = kwargs["model_type"]
    for item in ground_truth:
        image_id = int(item["id"][1:])
        for conv in item["conversations"]:
            if conv["from"] == "gpt":
                if model_type == "LLaVA":
                    parsed_bboxes = parse_bboxes(conv["value"])
                else:
                    parsed_bboxes = parse_bboxes_qwen(conv["value"])
                if parsed_bboxes:
                    gt_map[image_id] = parsed_bboxes

    for item in predictions:
        image_id = int(item["id"][1:])
        if model_type == "LLaVA":
            parsed_bboxes = parse_bboxes(item["text"])
        else:
            parsed_bboxes = parse_bboxes_qwen(item["text"])
        if model_type == "Qwen" and parsed_bboxes:
            pred_map[image_id] = {"bbox":parsed_bboxes,"input_width":item['input_width'],"input_height":item['input_height']}
        elif parsed_bboxes:
            pred_map[image_id] = parsed_bboxes
    
    pred_map = {k:v for k,v in pred_map.items() if k in gt_map.keys()}

    all_categories = set()
    for box in gt_map.values():
        for category_name in box.keys():
            all_categories.add(category_name.lower())

    categories = [{'id': idx + 1, 'name': name} for idx, name in enumerate(sorted(all_categories))]
    category_name_to_id = {cat['name']: cat['id'] for cat in categories}

    # COCO format
    gt, pred = convert_to_coco_format(gt_map, pred_map, category_name_to_id, image_id_to_path, kwargs["image_folder"], model_type)

    gt_json = {
        'images': [{"id": img_id} for img_id in set(gt_map.keys())],
        'annotations': gt,
        'categories': categories
    }
    
    coco_gt = COCO()
    coco_gt.dataset = gt_json
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(pred)

    # Calculate COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return get_evaluation_metrics(coco_eval)
