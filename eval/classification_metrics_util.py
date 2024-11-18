import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
def compute_optimal_threshold(fprs, tprs, sorted_scores):
    """Find the optimal threshold using Youden's J Index (maximizing TPR - FPR).

    Args:
        fprs: List of false positive rates.
        tprs: List of true positive rates.
        sorted_scores: List of sorted anomaly scores.

    Returns:
        optimal_threshold: Threshold that maximizes TPR - FPR.
    """
    # Calculate Youden’s J Index for each threshold
    youdens_j = [tpr - fpr for tpr, fpr in zip(tprs, fprs)]
    
    # Find index of maximum Youden's J value
    optimal_index = np.argmax(youdens_j)
    
    # Return the corresponding threshold
    return sorted_scores[optimal_index]

def compute_image_level_metrics(
        anomaly_maps,
        scoring_function,
        ground_truth_labels,
        fprs, tprs):
    assert len(anomaly_maps) == len(ground_truth_labels)

    # Compute the anomaly score for each anomaly map.
    anomaly_scores = map(scoring_function, anomaly_maps)

    # Sort samples by anomaly score. Keep track of ground truth label.
    sorted_samples = \
        sorted(zip(anomaly_scores, ground_truth_labels), key=lambda x: x[0])
        
    sorted_scores, sorted_labels = zip(*sorted_samples)
    optimal_threshold = compute_optimal_threshold(fprs, tprs, sorted_scores)
    predictions = np.asarray([1 if score > optimal_threshold else 0 for score in sorted_scores])
    accuracy = accuracy_score(sorted_labels, predictions)
    precision = precision_score(sorted_labels, predictions)
    recall = recall_score(sorted_labels, predictions)
    f1 = f1_score(sorted_labels, predictions)
    
    return {
        'threshold': float(optimal_threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def compute_pixel_level_metrics(anomaly_maps, ground_truth_maps):
     # Find the best threshold based on F1-score
    ground_truth_maps = np.asarray(ground_truth_maps)
    ground_truth_maps = (ground_truth_maps > 0.5).astype(int) 
    anomaly_maps = np.asarray(anomaly_maps)
    precision, recall, thresholds = precision_recall_curve(ground_truth_maps.flatten(), anomaly_maps.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    # Calculate metrics with the best threshold
    tp = fp = tn = fn = 0
    for gt_map, anomaly_map in zip(ground_truth_maps, anomaly_maps):
        anomaly_binary_map = anomaly_map > threshold
        tp += np.sum((anomaly_binary_map == 1) & (gt_map == 1))
        fp += np.sum((anomaly_binary_map == 1) & (gt_map == 0))
        tn += np.sum((anomaly_binary_map == 0) & (gt_map == 0))
        fn += np.sum((anomaly_binary_map == 0) & (gt_map == 1))

    pixel_accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    pixel_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    pixel_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    pixel_f1 = (2 * pixel_precision * pixel_recall) / (pixel_precision + pixel_recall) if (pixel_precision + pixel_recall) > 0 else 0
    
    return {
        'threshold': float(threshold),
        'accuracy':  float(pixel_accuracy),
        'precision':  float(pixel_precision),
        'recall':  float(pixel_recall),
        'f1':  float(pixel_f1)
    }