"""Compute evaluation metrics for a single experiment."""

__author__ = "Paul Bergmann, David Sattlegger"
__copyright__ = "2021, MVTec Software GmbH"

from os import listdir, makedirs, path
import numpy as np

import eval.generic_util as util
from eval.pro_curve_util import compute_pro
from eval.roc_curve_util import compute_classification_roc
from eval.classification_metrics_util import compute_image_level_metrics, compute_pixel_level_metrics


def calculate_metrics(ground_truth,
                      predictions,
                      integration_limit):

    # Compute the PRO curve.
    pro_curve = compute_pro(
        anomaly_maps=predictions,
        ground_truth_maps=ground_truth)

    # Compute the area under the PRO curve.
    au_pro = util.trapezoid(
        pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit
    print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro}")

    pixel_level_metrics = compute_pixel_level_metrics(
        anomaly_maps=predictions,
        ground_truth_maps=ground_truth)
    print('Threshold: {:.4f}'.format(pixel_level_metrics['threshold']))
    print('Pixel-level Accuracy: {:.4f}'.format(pixel_level_metrics['accuracy']))
    print('Pixel-level Precision: {:.4f}'.format(pixel_level_metrics['precision']))
    print('Pixel-level Recall: {:.4f}'.format(pixel_level_metrics['recall']))
    print('Pixel-level F1 Score: {:.4f}'.format(pixel_level_metrics['f1']))

    # Derive binary labels for each input image:
    # (0 = anomaly free, 1 = anomalous).
    binary_labels = [int(np.any(x > 0)) for x in ground_truth]
    del ground_truth

    # Compute the classification ROC curve.
    roc_curve = compute_classification_roc(
        anomaly_maps=predictions,
        scoring_function=np.max,
        ground_truth_labels=binary_labels)

    # Compute the area under the classification ROC curve.
    au_roc = util.trapezoid(roc_curve[0], roc_curve[1])
    print(f"Image-level classification AU-ROC: {au_roc}")

    image_level_metrics = compute_image_level_metrics(
        anomaly_maps=predictions,
        scoring_function=np.max,
        ground_truth_labels=binary_labels,
        fprs=roc_curve[0], tprs=roc_curve[1])

    print('Threshold: {:.4f}'.format(image_level_metrics['threshold']))
    print('Image-level Accuracy: {:.4f}'.format(image_level_metrics['accuracy']))
    print('Image-level Precision: {:.4f}'.format(image_level_metrics['precision']))
    print('Image-level Recall: {:.4f}'.format(image_level_metrics['recall']))
    print('Image-level F1 Score: {:.4f}'.format(image_level_metrics['f1']))
    
    # Return the evaluation metrics.
    return au_pro, au_roc, pro_curve, roc_curve, pixel_level_metrics, image_level_metrics
