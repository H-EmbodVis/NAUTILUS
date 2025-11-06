from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    matthews_corrcoef, 
    cohen_kappa_score, 
    log_loss
)
from sklearn.preprocessing import LabelBinarizer
from concurrent.futures import ThreadPoolExecutor

def evaluate_classification_metrics(predictions, ground_truth, kwargs):
    
    try:
        gt_map = {item["id"]: item["conversations"][-1]["value"].strip().lower() for item in ground_truth}
        pred_map = {item["question_id"]: item["text"].strip().lower() for item in predictions}       
    except:
        gt_map = {item["id"]: item["conversations"][-1]["value"].strip().lower() for item in ground_truth}
        pred_map = {item["id"]: item["text"].strip().lower() for item in predictions}

    y_true = []
    y_pred = []
    for question_id, pred_label in pred_map.items():
        if question_id in gt_map:
            y_true.append(gt_map[question_id])
            y_pred.append(pred_label)

    if len(y_true) != len(y_pred):
        raise ValueError("Mismatch between the number of true labels and predicted labels.")

    for idx in range(len(y_true)):
        if y_true[idx] in y_pred[idx]:
            y_pred[idx] = y_true[idx]

    metrics = {}

    # Accuracy
    metrics["Accuracy"] = float(accuracy_score(y_true, y_pred))

    # Precision, Recall, F1 (Weighted)
    metrics["Precision (weighted)"] = float(precision_score(y_true, y_pred, average='weighted'))
    # metrics["Recall (macro)"] = float(recall_score(y_true, y_pred, average='macro'))
    # metrics["Recall (micro)"] = float(recall_score(y_true, y_pred, average='micro'))
    metrics["F1 Score (weighted)"] = float(f1_score(y_true, y_pred, average='weighted'))

    # Matthews Correlation Coefficient (MCC)
    metrics["Matthews Correlation Coefficient"] = float(matthews_corrcoef(y_true, y_pred))

    # Cohen's Kappa Score
    metrics["Cohen's Kappa Score"] = float(cohen_kappa_score(y_true, y_pred))

    # Log Loss
    try:
        metrics["Log Loss"] = float(log_loss(y_true, y_pred, labels=list(set(y_true))))
    except ValueError:
        metrics["Log Loss"] = None  

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics["Confusion Matrix"] = conf_matrix.tolist()  # 

    # ROC AUC (Weighted)
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    y_pred_bin = lb.transform(y_pred)
    try:
        metrics["AUC (weighted)"] = float(roc_auc_score(y_true_bin, y_pred_bin, average='weighted', multi_class='ovr'))
    except ValueError:
        metrics["AUC (weighted)"] = None

    metrics["Classification Report"] = classification_report(y_true, y_pred, output_dict=True)

    return metrics