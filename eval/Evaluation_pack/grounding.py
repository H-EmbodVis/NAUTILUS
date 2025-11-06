import json
import os
import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def clear_special_tokens(in_str):
    return in_str.replace("</ref>","").replace("<ref>","").replace("</box>","").replace("<box>","")

def parse_bboxes(bbox_str):
    try:
        bbox_dict = {} 
        if "],[" in bbox_str or "], [" in bbox_str or len(bbox_str.split("\n")) > 1:
            print(f"Invalid bounding box format. {bbox_str}'")
            return None
        elif '[[' in bbox_str:
            bbox_str = clear_special_tokens(bbox_str)
            bbox_str = "[" + bbox_str.split("[[")[-1]
            bbox_str = bbox_str[0:-1]
        lines = bbox_str.strip().split('\n')
        bbox_dict['default'] = []
        if not any(':' in line for line in lines):
            for line in lines:
                bbox_coords = [float(coord) for coord in line.strip('[]').split(',') if coord.strip()]
                # for Internvl format bbox
                if max(bbox_coords) > 1: 
                    bbox_coords = [float(coord/1000) for coord in bbox_coords]
                # if len(bbox_coords) != 4:
                #     raise ValueError(f"Bounding box should contain 4 values, but got {bbox_coords}")
                bbox_dict['default'].append(bbox_coords)
        else:
            for line in lines:
                try:
                    _, bbox_coords_str = line.split(':')
                except ValueError:
                    continue
                bbox_coords_str = bbox_coords_str.strip.strip()('[]')
                bbox_coords = [float(coord) for coord in bbox_coords_str.split(',') if coord.strip()]
                if len(bbox_coords) != 4:
                    continue
                    print(f"Bounding box should contain 4 values, but got {bbox_coords}")
                else:
                    bbox_dict['default'].append(bbox_coords)
        return bbox_dict
    except Exception as e:
        print(f"Error parsing bounding box string: {bbox_str}. Error: {e}")
        return None

def parse_bboxes_qwen(bbox_str):
    try:
        bbox_dict = {'default':[]}
        bbox_json = json.loads(bbox_str)
        if isinstance(bbox_json,list):
            bbox_json=bbox_json[0]
        bbox_2d = bbox_json['bbox_2d']
        if "Fish" in bbox_str or "],[" in bbox_str or "], [" in bbox_str:
            return None
        bbox_coords = [float(coord) for coord in bbox_2d]
        bbox_dict['default'].append(bbox_coords)
        return bbox_dict
    except Exception as e:
        print(f"Error parsing bounding box string: {bbox_str}. Error: {e}")
        bbox_dict['default'].append([0.0, 0.0, 0.0, 0.0])
        return bbox_dict

def convert_to_coco_format(gt_map, pred_map, category_name_to_id, image_id_to_path, base_path, model_type ="LLaVA"):
    gt_annotations, pred_annotations = [], []
    gt_id, pred_id = 1, 1
    image_ids = set()
    gt_boxes_dict = {}
    pred_boxes_dict = {}
    images_info = []

    for image_id, boxes in gt_map.items():
        image_ids.add(image_id)
        image_path = image_id_to_path.get(image_id)
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

        images_info.append({
            "id": image_id,
            "file_name": image_path,
            "width": img_width,
            "height": img_height
        })

        gt_boxes = []

        for category_name, box_list in boxes.items():
            category_name = category_name.lower()
            if category_name not in category_name_to_id:
                print(f"Category '{category_name}' not found in category_name_to_id.")
                continue
            category_id = category_name_to_id[category_name]
            for bbox in box_list:
                if model_type == "LLaVA":
                    x_min, y_min, x_max, y_max = [
                        coord * img_width if i % 2 == 0 else coord * img_height 
                        for i, coord in enumerate(bbox)
                    ]
                elif model_type == "Qwen":
                    x_min, y_min, x_max, y_max = bbox
                else:
                    raise ValueError(f"Invalid model_type '{model_type}'")
                width, height = x_max - x_min, y_max - y_min
                area = width * height
                gt_annotations.append({
                    "id": gt_id, "image_id": image_id, "category_id": category_id,
                    "bbox": [x_min, y_min, width, height], "area": area, "iscrowd": 0
                })
                gt_id += 1
                gt_boxes.append([x_min, y_min, x_max, y_max])
        gt_boxes_dict[image_id] = gt_boxes

    for image_id, pred in pred_map.items():
        if image_id not in image_ids:
            continue

        # image_path = image_id_to_path.get(str(image_id))
        image_path = image_id_to_path.get(image_id)
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

        pred_boxes = []
        if model_type == "Qwen":
            boxes = pred["bbox"]
            input_width, input_height = pred['input_width'], pred['input_height']
        else:
            boxes = pred
        for category_name, box_list in boxes.items():
            category_name = category_name.lower()
            category_id = category_name_to_id.get(category_name)
            if category_id is None:
                print(f"Category '{category_name}' not found in category_name_to_id.")
                continue
            for bbox in box_list:
                if len(bbox) != 4:
                    print(f"Invalid bbox format for image_id {image_id}: {bbox}")
                    continue
                if model_type == "LLaVA":
                    x_min, y_min, x_max, y_max = [
                        coord * img_width if i % 2 == 0 else coord * img_height 
                        for i, coord in enumerate(bbox)
                    ]
                elif model_type == "Qwen":
                    x_min, y_min, x_max, y_max = [
                        coord * img_width / input_width if i % 2 == 0 else coord * img_height / input_height
                        for i, coord in enumerate(bbox)
                    ]
                width, height = x_max - x_min, y_max - y_min
                area = width * height
                pred_annotations.append({
                    "id": pred_id, "image_id": image_id, "category_id": category_id,
                    "bbox": [x_min, y_min, width, height], "area": area, "score": 0.9
                })
                pred_id += 1
                pred_boxes.append([x_min, y_min, x_max, y_max])
        pred_boxes_dict[image_id] = pred_boxes

    return gt_annotations, pred_annotations, gt_boxes_dict, pred_boxes_dict, images_info

def get_evaluation_metrics(coco_eval, gt_boxes_dict, pred_boxes_dict):
    metrics = {}

    # Get COCO evaluation metrics
    metrics['mAP'] = float(coco_eval.stats[0])
    metrics['AP@0.5'] = float(coco_eval.stats[1])
    metrics['AP@0.75'] = float(coco_eval.stats[2])
    metrics['AR@1'] = float(coco_eval.stats[6])
    metrics['AR@10'] = float(coco_eval.stats[7])
    metrics['AR@100'] = float(coco_eval.stats[8])

    miou_list = []
    tp_05, tp_075 = 0, 0
    total_preds = 0
    total_gts = 0

    for image_id in gt_boxes_dict.keys():
        gt_boxes = torch.tensor(gt_boxes_dict[image_id], dtype=torch.float32)
        pred_boxes = torch.tensor(pred_boxes_dict.get(image_id, []), dtype=torch.float32)
        total_preds += len(pred_boxes)
        total_gts += len(gt_boxes)

        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            miou_list.append(0)
            continue

        # Calculate IoUs
        ious = box_iou(pred_boxes, gt_boxes)  # Shape: [num_preds, num_gts]

        max_iou_per_gt, _ = ious.max(dim=0)  # Shape: [num_gts]
        max_iou_per_pred, _ = ious.max(dim=1)  # Shape: [num_preds]

        # mIoU 
        miou = max_iou_per_gt.mean().item() if max_iou_per_gt.numel() > 0 else 0
        if np.isnan(miou):
            continue
        miou_list.append(miou)

        tp_05 += (max_iou_per_pred >= 0.5).sum().item()
        tp_075 += (max_iou_per_pred >= 0.75).sum().item()

    # Precision@IoU
    metrics['Precision@0.5'] = tp_05 / total_preds if total_preds > 0 else 0
    metrics['Precision@0.75'] = tp_075 / total_preds if total_preds > 0 else 0

    # Recall@IoU
    metrics['Recall@0.5'] = tp_05 / total_gts if total_gts > 0 else 0
    metrics['Recall@0.75'] = tp_075 / total_gts if total_gts > 0 else 0

    # mIoU
    metrics['mIoU'] = float(np.mean(miou_list)) if miou_list else 0

    return metrics

def evaluate_ground_metrics(predictions, ground_truth, kwargs):
    count = 1

    id_mapping = {}

    image_id_to_path = {}
    # image_id_to_path = {item["id"]: item["image"] for item in ground_truth}

    gt_map, pred_map = {}, {}
    model_type = kwargs["model_type"]

    for item in ground_truth:
        try:
            image_id = int(item["id"])
        except ValueError:
            if item["id"] not in id_mapping:
                id_mapping[item["id"]] = count
                count += 1
            image_id = id_mapping[item["id"]]

        image_id_to_path[image_id] = item["image"]

        if model_type == "LLaVA":
            parsed_bboxes = parse_bboxes(item['conversations'][1].get("value", ""))
        elif model_type == "Qwen":
            parsed_bboxes = parse_bboxes_qwen(item['conversations'][1].get("value", ""))
        if parsed_bboxes:
            gt_map[image_id] = parsed_bboxes

    for item in predictions:
        try:
            image_id = int(item["id"])
        except ValueError:
            if item["id"] not in id_mapping:
                id_mapping[item["id"]] = count
                count += 1
            image_id = id_mapping[item["id"]]
        if model_type == "LLaVA":
            parsed_bboxes = parse_bboxes(item.get("text", ""))
        elif model_type == "Qwen":
            parsed_bboxes = parse_bboxes_qwen(item.get("text", ""))
        if model_type == "Qwen" and parsed_bboxes:
            pred_map[image_id] = {"bbox":parsed_bboxes,"input_width":item['input_width'],"input_height":item['input_height']}
        elif parsed_bboxes:
            pred_map[image_id] = parsed_bboxes
    pred_map = {k:v for k,v in pred_map.items() if k in gt_map.keys()}

    category_name_to_id = {"default": 1}
    categories = [{"id": 1, "name": "default"}]

    gt_annotations, pred_annotations, gt_boxes_dict, pred_boxes_dict, images_info = convert_to_coco_format(
        gt_map, pred_map, category_name_to_id, image_id_to_path, kwargs["image_folder"], model_type
    )

    gt_json = {
        'images': images_info,
        'annotations': gt_annotations,
        'categories': categories
    }

    coco_gt = COCO()
    coco_gt.dataset = gt_json
    coco_gt.createIndex()
    try:
        coco_pred = coco_gt.loadRes(pred_annotations)
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return {}

    coco_eval = COCOeval(coco_gt, coco_pred, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = get_evaluation_metrics(coco_eval, gt_boxes_dict, pred_boxes_dict)
    return metrics
