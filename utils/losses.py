import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import get_dict_value


def log_cosh(x: torch.Tensor) -> torch.Tensor:
    return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)


def get_lossfx(loss, params):
    if loss == 'DiceLoss':
        lossfx = DiceLoss(weight=torch.FloatTensor(params['lossfx']['weight']), use_csi=params['lossfx']['use_csi'],
                          use_logcosh=params['lossfx']['use_logcosh'],
                          use_neglog=get_dict_value(params['lossfx'], 'use_neglog', False),
                          image_avg=get_dict_value(params['lossfx'], 'image_avg', False),
                          time_weighted=get_dict_value(params['lossfx'], 'time_weighted', False)
                          )
    else:
        raise ValueError(f'No support loss function {loss}!')

    return lossfx


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, use_csi=False, use_logcosh=False,
                 smooth=1e-5, use_neglog=False, image_avg=False, time_weighted=False):
        super(DiceLoss, self).__init__()
        print("-------------- use dice loss --------------")
        print("use logcosh: ", use_logcosh)
        print("use use_neglog: ", use_neglog)
        print("use image_avg: ", image_avg)
        print("use time weighted: ", time_weighted)
        self.weight = None
        if weight is not None:
            self.weight = (weight / torch.sum(weight)).clone().detach()

        self.use_csi = use_csi
        self.use_logcosh = use_logcosh
        self.use_neglog = use_neglog
        self.smooth = smooth
        self.image_avg = image_avg
        self.time_weighted = time_weighted

    def forward(self, inputs, targets, smooth=1e-5, distribution=False, mask=None):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if mask is not None:
            inputs[mask] = 0
            targets[mask] = 0
        if not distribution:
            inputs = torch.softmax(inputs, dim=1)

        targets = torch.cumsum(torch.flip(targets, dims=[1]), dim=1)[:, :-1]
        inputs = torch.cumsum(torch.flip(inputs, dims=[1]), dim=1)[:, :-1]
        if self.time_weighted:
            class_inputs = torch.flatten(inputs.permute(1, 2, 0, 3, 4), start_dim=2)
            class_targets = torch.flatten(targets.permute(1, 2, 0, 3, 4), start_dim=2)
        else:
            class_inputs = torch.flatten(inputs.permute(1, 2, 0, 3, 4), start_dim=1)
            class_targets = torch.flatten(targets.permute(1, 2, 0, 3, 4), start_dim=1)
        intersection = (class_inputs * class_targets).sum(dim=-1)
        class_inputs_card = class_inputs.sum(dim=-1)
        class_targets_card = class_targets.sum(dim=-1)
        correction = intersection if self.use_csi else 0
        class_dice = (2. * intersection + self.smooth) / (
                class_inputs_card + class_targets_card - correction + self.smooth)
        if self.time_weighted:
            init_weight = 0.9 ** torch.arange(class_dice.size(1), device=class_dice.device).float()
            weight = init_weight / torch.sum(init_weight)
            class_dice = torch.sum(class_dice * weight.view(1, -1), dim=1)
        if self.weight is None or self.image_avg:
            dice = torch.mean(class_dice)
        else:
            self.weight = self.weight.to(class_dice.device)
            dice = torch.sum(class_dice * self.weight)
        # log-cosh transform
        if self.use_logcosh:
            return log_cosh(1 - dice)
        elif self.use_neglog:
            return -torch.log(torch.clamp(dice, min=1e-15))
        else:
            return 1 - dice
