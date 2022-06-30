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
    # print(f"TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn}, overall = {overall}")
    assert tp + tn + fp + fn == overall

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / overall

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    overall = ground_truth.shape[0]
    confusion_matrix = np.zeros((overall, overall))
    for p, gt in zip(prediction, ground_truth):
        confusion_matrix[gt, p] += 1

    accuracy = np.sum([confusion_matrix[i][i] for i in range(overall)]) / overall
    return accuracy
