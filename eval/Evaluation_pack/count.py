import json
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_options(prompt):
    options = re.findall(r"([a-d]):(\d+)", prompt.lower())
    return {option: int(count) for option, count in options}

def build_ground_truth_map(ground_truth):
    gt_map = {}
    for item in ground_truth:
        question_id = item["id"]
        prompt = item["conversations"][0]["value"]
        answer = item["conversations"][-1]["value"].lower()

        if answer in "abcd":
            options = parse_options(prompt)
            gt_map[question_id] = {'type': 'choice', 'answer': answer, 'options': options}
        else:
            try:
                gt_map[question_id] = {'type': 'counting', 'answer': int(answer)}
            except ValueError:
                gt_map[question_id] = {'type': 'counting', 'answer': None}
    return gt_map

def build_prediction_map(predictions):
    pred_map = {}
    for item in predictions:
        question_id = item["id"]
        pred_answer = item["text"].lower()
        prompt = item.get("prompt", "")
        gt = item["gt"].lower()

        if gt in "abcd":
            options = parse_options(prompt)
            pred_map[question_id] = {'type': 'choice', 'answer': pred_answer, 'options': options}
        else:
            try:
                pred_map[question_id] = {'type': 'counting', 'answer': int(pred_answer)}
            except ValueError:
                pred_map[question_id] = {'type': 'counting', 'answer': None}
    return pred_map

def evaluate_choice(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1)
    }


def evaluate_counting(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_total)

    max_error = np.max(np.abs(y_pred - y_true))
    min_error = np.min(np.abs(y_pred - y_true))
    std_error = np.std(np.abs(y_pred - y_true))

    return {
        "Mean Absolute Error (MAE)": float(mae),
        "Mean Squared Error (MSE)": float(mse),
        "Root Mean Squared Error (RMSE)": float(rmse),
        "Mean Absolute Percentage Error (MAPE)": float(mape),
        "R-Squared (R²)": float(r_squared),
        "Max Error": float(max_error),
        "Min Error": float(min_error),
        "Standard Deviation of Errors": float(std_error)
    }


def evaluate_count_metrics(predictions, ground_truth, kwargs):
    gt_map = build_ground_truth_map(ground_truth)
    pred_map = build_prediction_map(predictions)

    y_true_counting = []
    y_pred_counting = []
    y_true_choice = []
    y_pred_choice = []

    for question_id, pred_data in pred_map.items():
        if question_id in gt_map:
            gt_data = gt_map[question_id]
            if gt_data['type'] == 'counting' and pred_data['type'] == 'counting':
                if gt_data['answer'] is not None and pred_data['answer'] is not None:
                    y_true_counting.append(gt_data['answer'])
                    y_pred_counting.append(pred_data['answer'])
            elif gt_data['type'] == 'choice' and pred_data['type'] == 'choice':
                if gt_data['answer'] is not None and pred_data['answer'] is not None:
                    y_true_choice.append(gt_data['answer'])
                    y_pred_choice.append(pred_data['answer'])

    metrics = {}

    if y_true_choice and y_pred_choice:
        metrics["Choice Evaluation Metrics"] = evaluate_choice(y_true_choice, y_pred_choice)
    else:
        metrics["Choice Evaluation Metrics"] = {"Error": "No valid choice data available for evaluation."}
    
    if y_true_counting and y_pred_counting:
        metrics["Counting Evaluation Metrics"] = evaluate_counting(y_true_counting, y_pred_counting)
    else:
        metrics["Counting Evaluation Metrics"] = {"Error": "No valid counting data available for evaluation."}

    return metrics
