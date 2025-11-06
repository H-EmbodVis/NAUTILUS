import json
import re
import sys

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pprint import pprint

# from google import genai
import time
import os
import random

def process_text(text: str) -> str:
    return re.sub(r'[\W_]+$', '', text).strip()


def exact_match_strategy(gts: dict, res: dict):
    imgIds = gts.keys()
    accurate_num = 0
    for id in imgIds:
        if res[id] == gts[id]:
            accurate_num += 1
    return round(float(float(accurate_num) / len(gts)),3)

def calculate_metrics(gts: dict, res: dict, java = False) -> dict:
    # BLEU
    bleu_scorer = Bleu(n=4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)
    
    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    
    # ROUGE
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)

    # java environment required
    if java:
        # METEOR
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(gts, res)
        
        # SPICE
        # spice_scorer = Spice()
        # spice_score, _ = spice_scorer.compute_score(gts, res)
        spice_score = -1
    else:
        meteor_score = -1
        spice_score = -1

    scores = {
        "bleu_score": [float(bleu) for bleu in bleu_score],
        "cider_score": float(cider_score),
        "rouge_score": float(rouge_score),
        "meteor_score": float(meteor_score),
        "spice_score": float(spice_score),
    }

    return scores

def load_and_process(gt_data: dict, res_data: dict) -> (dict, dict):
    """加载并处理ground truth和预测结果数据"""
    gts = {}
    res = {}

    for entry_gt, entry_res in zip(gt_data, res_data):
        gt_id = entry_gt['id']
        try:
            res_id = entry_res['question_id']
        except:
            res_id = entry_res['id']

        for conversation in entry_gt['conversations']:
            if conversation['from'] == 'gpt':
                desc = process_text(conversation['value'])
                gts[gt_id] = [desc]

        desc = process_text(entry_res['text'])
        res[res_id] = [desc]

    print(f"Total:{len(gts)}")
    return gts, res

def clear_special_tokens(in_str):
    return in_str.replace("</ref>","").replace("<ref>","").replace("</box>","").replace("<box>","").replace("<image>","").replace("\n","")

def evaluate_text_metrics(predictions_data: dict, ground_truth_data: dict, kwargs) -> dict:
    java_enabled = kwargs["java"]
    gts, res = load_and_process(ground_truth_data, predictions_data)

    return calculate_metrics(gts, res, java_enabled)