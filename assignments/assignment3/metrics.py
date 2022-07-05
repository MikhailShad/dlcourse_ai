import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    overall = ground_truth.shape[0]

    tp = len(list(filter(lambda pred_gt: pred_gt[0] == 1 and pred_gt[1] == 1, zip(prediction, ground_truth))))
    tn = len(list(filter(lambda pred_gt: pred_gt[0] == 0 and pred_gt[1] == 0, zip(prediction, ground_truth))))
    fp = len(list(filter(lambda pred_gt: pred_gt[0] == 1 and pred_gt[1] == 0, zip(prediction, ground_truth))))
    fn = len(list(filter(lambda pred_gt: pred_gt[0] == 0 and pred_gt[1] == 1, zip(prediction, ground_truth))))
    assert tp + tn + fp + fn == overall

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / overall

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    return np.sum(prediction == ground_truth) / len(prediction)
