# Weather4cast 2023 Starter Kit
#
# This Starter Kit builds on and extends the Weather4cast 2022 Starter Kit,
# the original license for which is included below.
#
# In line with the provisions of this license, all changes and additional
# code are also released unde the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
# 
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


import numpy as np
import torch
import torch.nn.functional as F


def to_one_hot(arr, thresholds=None):
    if thresholds is None:
        thresholds = [0.2, 1, 5, 10, 15]
    num_classes = len(thresholds) + 1
    one_hot = torch.zeros((arr.shape[0], num_classes) + arr.shape[2:], dtype=torch.float32, device=arr.device)

    for i, threshold in enumerate(thresholds):
        if i == 0:
            one_hot[:, i] = (arr < threshold).squeeze(1)
        else:
            one_hot[:, i] = ((arr >= thresholds[i - 1]) & (arr < threshold)).squeeze(1)

    one_hot[:, -1] = (arr >= thresholds[-1]).squeeze(1)
    return one_hot


def combine_metrics(y=None, y_hat=None, cm=None, mode='val', multi_output=False, pred_len=32):
    threshold = np.array([0.2, 1, 5, 10, 15])
    if mode == 'val_step':
        if not multi_output:
            log_mean = np.mean(np.log(threshold + 1e-2))
            ret = torch.zeros((5, pred_len, 4), device=y.device)
            for i in range(5):
                for j in range(pred_len):
                    y_i = y > threshold[i]
                    y_hat_i = y_hat > np.log(threshold[i]) - log_mean
                    tn, fp, fn, tp = get_confusion_matrix(y_i, y_hat_i)
                    ret[i - 1, j] = torch.tensor([tn, fp, fn, tp], device=y.device)
        else:
            num_classes = y.shape[1]
            pred_len = y.shape[2]
            y_hat = F.softmax(y_hat, dim=1)
            y_hat = torch.argmax(y_hat, dim=1)
            y_hat = F.one_hot(y_hat, num_classes=num_classes).permute(0, 4, 1, 2, 3)

            ret = torch.zeros((num_classes - 1, pred_len, 4), device=y.device)
            for i in range(1, num_classes):
                for j in range(pred_len):
                    y_i = torch.sum(y[:, i:, j, :, :], dim=1)
                    y_hat_i = torch.sum(y_hat[:, i:, j, :, :], dim=1)
                    tn, fp, fn, tp = get_confusion_matrix(y_i, y_hat_i)
                    ret[i - 1, j] = torch.tensor([tn, fp, fn, tp], device=y.device)
        return ret
    else:
        values = dict([])
        total_acc, total_recall, total_precision, total_F1, total_csi = [], [], [], [], []
        num_classes = 6
        for i in range(0, num_classes - 1):
            recall, precision, F1, acc, csi = recall_precision_f1_acc(cm=torch.sum(cm[i], dim=0))
            total_acc += [acc]
            total_recall += [recall]
            total_precision += [precision]
            total_F1 += [F1]
            total_csi += [csi]

        # 计算平均值
        avg_acc = sum(total_acc) / (num_classes - 1)
        avg_recall = sum(total_recall) / (num_classes - 1)
        avg_precision = sum(total_precision) / (num_classes - 1)
        avg_F1 = sum(total_F1) / (num_classes - 1)
        avg_csi = sum(total_csi) / (num_classes - 1)

        values.update({
            f'{mode}_avg_acc': avg_acc,
            f'{mode}_avg_recall': avg_recall,
            f'{mode}_avg_precision': avg_precision,
            f'{mode}_avg_F1': avg_F1,
            f'{mode}_avg_csi': avg_csi
        })

        for i in range(num_classes - 1):
            values[f'{mode}_acc_{threshold[i]}'] = total_acc[i]
            values[f'{mode}_recall_{threshold[i]}'] = total_recall[i]
            values[f'{mode}_precision_{threshold[i]}'] = total_precision[i]
            values[f'{mode}_F1_{threshold[i]}'] = total_F1[i]
            values[f'{mode}_csi_{threshold[i]}'] = total_csi[i]
        return values


def get_confusion_matrix(y, y_hat):
    """get confusion matrix from y_true and y_pred

    Args:
        y_true (numpy array): ground truth 
        y_pred (numpy array): prediction 

    Returns:
        confusion matrix
    """

    unique_mapping = (y * 2 + y_hat).to(torch.long).view(-1)
    cm = [0, 0, 0, 0]
    for i in range(len(cm)):
        cm[i] = (unique_mapping == i).sum()

    return cm


def recall_precision_f1_acc(y=None, y_hat=None, cm=None):
    """ returns metrics for recall, precision, f1, accuracy

    Args:
        y (numpy array): ground truth 
        y_hat (numpy array): prediction 

    Returns:
        recall(float): recall/TPR 
        precision(float): precision/PPV
        F1(float): f1-score
        acc(float): accuracy
        csi(float): critical success index
    """

    # pytorch to numpy
    if cm is None:
        cm = get_confusion_matrix(y, y_hat)

    # if len(cm) == 4:
    tn, fp, fn, tp = cm
    recall, precision, F1, acc, csi = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(
        0.), torch.tensor(0.)

    if (tp + fn) > 0:
        recall = tp / (tp + fn)

    if (tp + fp) > 0:
        precision = tp / (tp + fp)

    if (precision + recall) > 0:
        F1 = 2 * (precision * recall) / (precision + recall)

    if (tp + fn + fp) > 0:
        csi = tp / (tp + fn + fp)

    if (tn + fp + fn + tp) > 0:
        acc = (tn + tp) / (tn + fp + fn + tp)

    return recall.item(), precision.item(), F1.item(), acc.item(), csi.item()
