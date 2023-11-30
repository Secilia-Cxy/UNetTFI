# Weather4cast 2023 Starter Kit
#
# The files from this repository make up the Weather4cast 2023 Starter Kit.
# 
# It builds on and extends the Weather4cast 2022 Starter Kit, the
# original copyright and GPL license notices for which are included
# below.
#
# In line with the provisions of that license, all changes and
# additional code are also released under the GNU General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
import datetime
# imports for plotting
import os
import random
from typing import Dict, Any

import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

# models
from models.baseline_UNET3D import UNet as Base_UNET3D  # 3_3_2 model selection
from models.unet2d import UNetWrapper as UNET2D  # 3_3_2 model selection
from utils.data_utils import get_dict_value
from utils.evaluate import *
from utils.evaluate import to_one_hot
from utils.losses import get_lossfx
from utils.viz import plot_sequence, save_pdf

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

VERBOSE = False


# VERBOSE = True

class UNet_Lightning(pl.LightningModule):
    def __init__(self, UNet_params: dict, params: dict,
                 **kwargs):
        super(UNet_Lightning, self).__init__()
        self.plot_results = get_dict_value(params, 'plot_results', False)
        self.in_channel_to_plot = get_dict_value(params, 'in_channel_to_plot', 7)
        self.in_channels = params['in_channels']
        self.start_filts = params['init_filter_size']
        self.dropout_rate = params['dropout_rate']
        self.multi_output = UNet_params['multi_output']
        self.crop_size = UNet_params['crop_size']
        self.rotation_aug = get_dict_value(UNet_params, 'rotation_aug', False)
        self.repeated_aug = get_dict_value(UNet_params, 'repeated_aug', True)
        if self.rotation_aug:
            self.candidate_rotation = [
                lambda x: torch.flip(x, dims=[-1]),
                lambda x: torch.flip(x, dims=[-2]),
                lambda x: torch.flip(x, dims=[-1, -2]),
                lambda x: torch.rot90(x, k=1, dims=[-1, -2]),
                lambda x: torch.rot90(x, k=-1, dims=[-1, -2]),
                lambda x: torch.rot90(x, k=2, dims=[-1, -2])
            ]
        self.center_output = get_dict_value(UNet_params, 'center_output', 0)
        print("use rotation augmentation : ", self.rotation_aug)
        print("use center output : ", self.center_output)
        self.out_channels = params['len_seq_predict']
        self.use_embedding = get_dict_value(UNet_params, 'use_embedding', False)
        self.use_time_mix = get_dict_value(UNet_params, 'use_time_mix', False)
        if self.use_embedding:
            self.embedding = nn.Embedding(7 * 2, 252 * 252)
            self.in_channels += 1

        backbone = get_dict_value(params, "backbone", "3D_UNET_base")
        if backbone == "3D_UNET_base":
            self.model = Base_UNET3D(in_channels=self.in_channels, start_filts=self.start_filts,
                                     dropout_rate=self.dropout_rate, out_channels=self.out_channels,
                                     multi_output=self.multi_output, crop_input=self.crop_size,
                                     crop_output=self.center_output)

        elif params["backbone"] == "UNET2D":
            self.model = UNET2D(input_channels=self.in_channels, input_step=4,
                                crop_input=self.crop_size, crop_output=self.center_output,
                                num_class=6 if self.multi_output else 1, forecast_step=32)


        else:
            raise NotImplementedError(f"model {params['backbone']} not implemented")

        self.save_hyperparameters()
        self.params = params

        self.val_batch = 0

        self.prec = 7

        pos_weight = torch.tensor(params['pos_weight'])
        if VERBOSE: print("Positive weight:", pos_weight)

        self.loss = params['loss']
        self.thres = None
        self.bs = params['batch_size']
        self.loss_fn = get_lossfx(self.loss, params)
        self.main_metric = {
            'DiceLoss': 'Dice',
        }[self.loss]

        self.relu = nn.ReLU()  # None
        t = f"============== n_workers: {params['n_workers']} | batch_size: {params['batch_size']} \n" + \
            f"============== loss: {self.loss}"
        print(t)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        print("loaded checkpoints")

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        if self.use_embedding:
            for name, param in self.embedding.named_parameters():
                param.requires_grad = False

    def on_fit_start(self):
        """ create a placeholder to save the results of the metric per variable """
        metric_placeholder = {self.main_metric: -1}
        self.logger.log_hyperparams(self.hparams)
        self.logger.log_metrics(metric_placeholder)

    def forward(self, x, metadata=None):
        if self.use_embedding:
            emb_idx = get_emb_idx(metadata, x.device)
            emb = self.embedding(emb_idx)
            emb = emb.reshape(-1, 1, 1, 252, 252)
            emb = emb.repeat(x.shape[0] // emb.shape[0], 1, x.shape[2], 1, 1)
            x = torch.cat([x, emb], dim=1)
        x = self.model(x)
        return x

    def retrieve_only_valid_pixels(self, x, m):
        """ we asume 1s in mask are invalid pixels """
        ##print(f"x: {x.shape} | mask: {m.shape}")
        return x[~m]

    def get_target_mask(self, metadata):
        mask = metadata['target']['mask']
        # print("mask---->", mask.shape)
        return mask

    def _compute_loss(self, y_hat, y, mask=None):
        loss = self.loss_fn(y_hat, y, mask=mask)
        return loss

    def training_step(self, batch, batch_idx, phase='train'):
        x, y, metadata = batch
        mask = self.get_target_mask(metadata)
        if self.use_time_mix:
            alpha = np.random.beta(1, 1)
            x_mix = (1 - alpha) * x[:, :, :-1] + alpha * x[:, :, 1:]
            y_mix = (1 - alpha) * y[:, :, :-1] + alpha * y[:, :, 1:]
            mask_mix = mask[:, :, :-1] | mask[:, :, 1:]
            x = torch.cat([x[:, :, :-1], x[:, :, 1:], x_mix], dim=0)
            y = torch.cat([y[:, :, :-1], y[:, :, 1:], y_mix], dim=0)
            mask = torch.cat([mask[:, :, :-1], mask[:, :, 1:], mask_mix], dim=0)
        else:
            x = torch.cat([x[:, :, :-1], x[:, :, 1:]], dim=0)
            y = torch.cat([y[:, :, :-1], y[:, :, 1:]], dim=0)
            mask = torch.cat([mask[:, :, :-1], mask[:, :, 1:]], dim=0)
        if self.multi_output:
            mask = torch.repeat_interleave(mask, repeats=6, dim=1)
            y = to_one_hot(y)

        if self.rotation_aug:
            if self.repeated_aug:
                x = torch.cat([x, torch.flip(x, dims=[-1]), torch.flip(x, dims=[-2]), torch.flip(x, dims=[-2, -1])],
                              dim=0)
                y = torch.cat([y, torch.flip(y, dims=[-1]), torch.flip(y, dims=[-2]), torch.flip(y, dims=[-2, -1])],
                              dim=0)
                mask = torch.cat(
                    [mask, torch.flip(mask, dims=[-1]), torch.flip(mask, dims=[-2]), torch.flip(mask, dims=[-2, -1])],
                    dim=0)
            else:
                randi = random.randint(0, len(self.candidate_rotation) - 1)
                x = torch.cat([x, self.candidate_rotation[randi](x)], dim=0)
                y = torch.cat([y, self.candidate_rotation[randi](y)], dim=0)
                mask = torch.cat(
                    [mask, self.candidate_rotation[randi](mask)],
                    dim=0)

        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')

        y_hat = self.forward(x, metadata)

        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')

        loss = self._compute_loss(y_hat, y, mask=mask)

        # LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, phase='val'):
        x, y, metadata = batch
        x_raw = x.detach().clone()
        mask = self.get_target_mask(metadata)

        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')

        if self.multi_output:
            mask = torch.repeat_interleave(mask, repeats=6, dim=1)
            y = to_one_hot(y)

        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')

        y_hat = self.forward(x, metadata)

        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')

        loss = self._compute_loss(y_hat, y, mask=mask)

        if mask is not None:
            y_hat[mask] = 0
            y[mask] = 0

        cm = combine_metrics(y=y, y_hat=y_hat, mode='val_step', multi_output=self.multi_output)

        # LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)

        if self.plot_results:
            title = f'batch {self.val_batch}'
            for channel in range(11):
                # self.in_channel_to_plot = channel
                self.plot_batch(x_raw, y, y_hat, metadata, title, 'test', vmax=1., channel=channel)
            self.val_batch += 1

        return {'loss': loss.cpu().item(), 'N': x.shape[0], "cm": cm}

    def validation_epoch_end(self, outputs, phase='val'):
        print("Validation epoch end average over batches: ",
              [batch['N'] for batch in outputs])
        values = {}
        cm = torch.sum(torch.stack([batch["cm"] for batch in outputs], 0), 0)
        values.update({f"{k}_epoch": v for k, v in
                       combine_metrics(cm=cm, mode=phase, multi_output=self.multi_output).items()})

        values[f"{phase}_loss_epoch"] = np.average([batch['loss'] for batch in outputs],
                                                   weights=[batch['N'] for batch in outputs])
        self.log_dict(values, batch_size=self.bs, sync_dist=True)
        self.log(self.main_metric, values[f"{phase}_loss_epoch"], batch_size=self.bs, sync_dist=True)
        # print(values)

    def test_step(self, batch, batch_idx, phase='test'):
        x, y, metadata = batch
        x_raw = x.detach().clone()
        y_raw = y.detach().clone()
        mask = self.get_target_mask(metadata)

        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')

        if self.multi_output:
            mask = torch.repeat_interleave(mask, repeats=6, dim=1)
            y = to_one_hot(y)

        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')

        y_hat = self.forward(x, metadata)

        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')

        loss = self._compute_loss(y_hat, y, mask=mask)

        if mask is not None:
            y_hat[mask] = 0
            y[mask] = 0

        cm = combine_metrics(y=y, y_hat=y_hat, mode='val_step', multi_output=self.multi_output)

        # LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)

        if self.plot_results:
            title = f'batch {self.val_batch}'
            self.plot_batch(x_raw, y_raw, y_hat, metadata, title, 'test', vmax=1., channel=self.in_channel_to_plot)

            self.val_batch += 1

        return {'loss': loss.cpu().item(), 'N': x.shape[0], "cm": cm, "y_hat": y_hat, "y": y}

    def plot_batch(self, xs, ys, y_hats, metadata, loss, phase, vmax=0.01, vmin=0, channel=0, mix=False):
        figures = []

        # ys = to_number(ys)
        y_hats = to_number(y_hats)

        # pytorch to numpy
        xs, y_hats = [o.cpu() for o in [xs, y_hats]]
        xs, y_hats = [np.asarray(o) for o in [xs, y_hats]]

        if phase in ["test"]:
            ys = ys.cpu()
            ys = np.asarray(ys)
        else:
            ys = y_hats  # it's going to be empty - just to make life easier while passing values to other functions

        print(f"\nplot batch of size {len(xs)}")
        for i in range(len(xs)):
            print(f"plot, {i + 1}/{len(xs)}")
            texts_in = [t[i] for t in metadata['input']['timestamps']]
            # print(texts_in)
            texts_ta = [t[i] for t in metadata['target']['timestamps']]
            # title = self.seq_metrics(ys[i].ravel(), y_hats[i].ravel())
            if VERBOSE:
                print("inputs")
                print(np.shape(xs[i]))
                if (phase == "test"):
                    print("target")
                    print(np.shape(ys[i]))
                print("prediction")
                print(np.shape(y_hats[i]))
            self.collapse_time = True

            fig = plot_sequence(xs[i], ys[i], y_hats[i], texts_in, texts_ta,
                                self.params, phase, self.collapse_time, vmax=vmax, vmin=vmin,
                                channel=channel, title=loss)
            figures.append(fig)
            # save individual image to tensorboard
            # self.logger.experiment.add_figure(f"preds_{self.trainer.global_step}_{self.val_batch}_{i}", fig)
            # self.logger.log_image(f"preds_{self.trainer.global_step}_{self.val_batch}_{i}", fig)
        # save all figures to disk
        date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
        channel_names = ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008',
                         'WV_062', 'WV_073']
        channel_name = channel_names[channel]
        if mix:
            fname = f"batch_{self.val_batch}_channel_{channel_name}_{date_time}_mix"
        else:
            fname = f"batch_{self.val_batch}_channel_{channel_name}_{date_time}"
        dir_path = os.path.join('plots', f"{self.params['name']}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f_path = os.path.join(dir_path, fname)
        save_pdf(figures, f_path)
        if (phase == "test"):
            print(f'saved figures at: {fname} | {loss}')
        else:
            print(f'saved figures at: {fname}')
        # self.val_batch += 1
        return figures

    def predict_step(self, batch, batch_idx, phase='predict'):
        x, y, metadata = batch
        mask = self.get_target_mask(metadata)

        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')

        y_hat = self.forward(x, metadata)

        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        if self.plot_results:
            self.plot_batch(x, y, y_hat, metadata, f'batch: {self.val_batch} | prediction results', phase, vmax=1.)
        return y_hat, y, mask

    def configure_optimizers(self):
        print("config optimizers")
        optim_params = self.params[self.params['optim']]

        model_parameters = self.set_model_params_optimizer()

        if self.params['optim'].lower() == 'adam':
            optimizer = optim.Adam(model_parameters, lr=float(self.params["lr"]), **optim_params)
        elif self.params['optim'].lower() == 'adamw':
            optimizer = optim.AdamW(model_parameters, lr=float(self.params["lr"]), **optim_params)
        elif self.params['optim'].lower() == 'sgd':
            optimizer = optim.SGD(model_parameters, lr=float(self.params["lr"]), **optim_params)
        else:
            raise ValueError(f'No support {self.params.optim} optimizer!')

        ## configure scheduler
        lr_params = self.params[self.params['scheduler']]

        print("Learning rate:", self.params["lr"],
              "optimizer: ", self.params["optim"], "optimier parameters: ", optim_params,
              "scheduler: ", self.params['scheduler'], "scheduler paramsters: ", lr_params)

        if self.params['scheduler'] == 'exp':
            scheduler = lr_scheduler.ExponentialLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'cosinewarm':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'onecycle':
            scheduler = lr_scheduler.OneCycleLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'reducelr':
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer, **lr_params),
                    'monitor': self.params['reducelr_monitor'],
                }
            }
        else:
            raise ValueError(f"No support {self.params['scheduler']} scheduler!")

    def set_model_params_optimizer(self):
        if 'no_bias_decay' in self.params and self.params.get('no_bias_decay'):
            if 'encoder_lr_ratio' in self.params:
                encoder_lr_ratio = self.params.get('encoder_lr_ratio')
                group_decay_encoder, group_no_decay_encoder = group_weight(self.model.down_convs)
                group_decay_decoder, group_no_decay_decoder = group_weight(self.model.up_convs)
                base_lr = self.params['lr']
                params = [{'params': group_decay_decoder},
                          {'params': group_no_decay_decoder, 'weight_decay': 0.0},
                          {'params': group_decay_encoder, 'lr': base_lr * encoder_lr_ratio},
                          {'params': group_no_decay_encoder, 'lr': base_lr * encoder_lr_ratio, 'weight_decay': 0.0}]
                print(
                    f'separately set lr with no_bias_decay for encoder {base_lr} and decoder {base_lr * encoder_lr_ratio}...')
            else:
                group_decay, group_no_decay = group_weight(self.model)
                params = [{'params': group_decay},
                          {'params': group_no_decay, 'weight_decay': 0.0}]
                print(f'set params with no_bias_decay...')
        elif 'encoder_lr_ratio' in self.params:
            encoder_lr_ratio = self.params.get('encoder_lr_ratio')
            base_lr = float(self.params['lr'])
            print(encoder_lr_ratio, base_lr)
            print(f'separately set lr for encoder {base_lr} and decoder {base_lr * encoder_lr_ratio}...')
            params = [{'params': self.model.up_convs.parameters()},
                      {'params': self.model.reduce_channels.parameters()},
                      {'params': self.model.down_convs.parameters(), 'lr': base_lr * encoder_lr_ratio}]
        else:
            params = list(filter(lambda x: x.requires_grad, self.parameters()))

        return params

    def freeze_model_params(self):
        if 'freeze_encoder' in self.params and self.params.get('freeze_encoder'):
            print('freezing the parameters of encoder...')
            for name, param in self.model.down_convs.named_parameters():
                param.requires_grad = False

        if 'freeze_decoder' in self.params and self.params.get('freeze_decoder'):
            print('freezing the parameters of decoder...')
            for name, param in self.model.down_convs.named_parameters():
                param.requires_grad = False

        if 'freeze_output' in self.params and self.params.get('freeze_output'):
            print('freezing the parameters of final output...')
            for name, param in self.model.reduce_channels.named_parameters():
                param.requires_grad = False


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return group_decay, group_no_decay


def to_number(y_hat, nums=None):
    if nums is None:
        nums = torch.tensor([0, 0.6, 3, 7.5, 12.5, 16]).reshape(1, 6, 1, 1, 1).to(y_hat.device)
    num_classes = 6
    y_hat = F.softmax(y_hat, dim=1)
    y_hat = torch.argmax(y_hat, dim=1)
    y_hat = F.one_hot(y_hat, num_classes=num_classes).permute(0, 4, 1, 2, 3)
    ret = torch.sum(y_hat * nums, axis=1, keepdim=True)
    return ret


def get_emb_idx(metadata, device):
    x_region = metadata['region']
    x_year = metadata['year']
    regions = ['boxi_0015', 'boxi_0034', 'boxi_0076', 'roxi_0004', 'roxi_0005', 'roxi_0006', 'roxi_0007']
    years = ['2019', '2020']
    emb_idx = [regions.index(region) * 2 + years.index(year) for region, year in zip(x_region, x_year)]
    emb_idx = torch.tensor(emb_idx, dtype=torch.long, device=device)
    return emb_idx


def main():
    print("running")


if __name__ == 'main':
    main()
