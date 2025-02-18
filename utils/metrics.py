import torch
import torch.nn as nn

from typing import Callable
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, r2_score, log_loss


def prc_auc(targets: torch.Tensor, preds: torch.Tensor) -> float:
    r"""
    Computes the area under the precision-recall curve.

    :param targets: A tensor of binary targets.
    :param preds: A tensor of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets.cpu().numpy(), preds.cpu().numpy())
    return auc(recall, precision)


def bce(targets: torch.Tensor, preds: torch.Tensor) -> float:
    r"""
    Computes the binary cross entropy loss.

    :param targets: A tensor of binary targets.
    :param preds: A tensor of prediction probabilities.
    :return: The computed binary cross entropy.
    """
    # Don't use logits because the sigmoid is added in all places except training itself
    bce_func = nn.BCELoss(reduction='mean')
    loss = bce_func(input=preds, target=targets)
    
    return loss


def rmse(targets: torch.Tensor, preds: torch.Tensor) -> float:
    r"""
    Computes the root mean squared error.

    :param targets: A tensor of targets.
    :param preds: A tensor of predictions.
    :return: The computed rmse.
    """
    return torch.sqrt(torch.mean((preds - targets) ** 2))


def mse(targets: torch.Tensor, preds: torch.Tensor) -> float:
    r"""
    Computes the mean squared error.

    :param targets: A tensor of targets.
    :param preds: A tensor of predictions.
    :return: The computed mse.
    """
    return torch.mean((preds - targets) ** 2)


def mae(targets: torch.Tensor, preds: torch.Tensor) -> float:
    r"""
    Computes the mean absolute error.

    :param targets: A tensor of targets.
    :param preds: A tensor of predictions.
    :return: The computed mae.
    """
    return torch.mean(torch.abs(preds - targets))


def accuracy(targets: torch.Tensor, preds: torch.Tensor, threshold: float = 0.5) -> float:
    r"""
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A tensor of binary targets.
    :param preds: A tensor of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    if preds.dim() > 1 and preds.size(1) > 1:  # multiclass
        hard_preds = torch.argmax(preds, dim=1)
    else:  # binary prediction
        hard_preds = (preds > threshold).float()

    return (hard_preds == targets).float().mean().item()


def get_metric_func(metric: str) -> Callable[[torch.Tensor, torch.Tensor], float]:
    r"""
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`auc`: Area under the receiver operating characteristic curve
    * :code:`prc-auc`: Area under the precision recall curve
    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`
    * :code:`accuracy`: Accuracy (using a threshold to binarize predictions)
    * :code:`cross_entropy`: Cross entropy
    * :code:`binary_cross_entropy`: Binary cross entropy

    :param metric: Metric name.
    :return: A metric function which takes as arguments a tensor of targets and a tensor of predictions and returns.
    """
    if metric == 'auc':
        return lambda targets, preds: roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy())

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mse

    if metric == 'mae':
        return mae

    if metric == 'r2':
        return lambda targets, preds: r2_score(targets.cpu().numpy(), preds.cpu().numpy())

    if metric == 'accuracy':
        return accuracy

    if metric == 'cross_entropy':
        return lambda targets, preds: log_loss(targets.cpu().numpy(), preds.cpu().numpy())

    if metric == 'binary_cross_entropy':
        return bce

    raise ValueError(f'Metric "{metric}" not supported.')