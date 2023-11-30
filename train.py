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


import argparse
import copy

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import os
import torch
import torch.nn.functional as F
import wandb
from utils.evaluate import recall_precision_f1_acc, get_confusion_matrix
from models.unet_lightning_w4c23 import UNet_Lightning as UNetModel
from utils.data_utils import load_config
from utils.data_utils import get_cuda_memory_usage
from utils.data_utils import tensor_to_submission_file
from utils.data_utils import get_dict_value
from utils.w4c_dataloader import RainData
from utils.evaluate import to_one_hot


class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """

    def __init__(self, params, training_params, mode):
        super().__init__()
        self.params = params
        self.training_params = training_params
        concat_train_val = get_dict_value(training_params, 'concat_train_val', False)
        print("-----------------------  concat_train_val: ", concat_train_val)
        if mode in ['train']:
            print("Loading TRAINING/VALIDATION dataset -- as test")
            if concat_train_val:
                self.val_ds = RainData('validation', **self.params)
                self.train_ds = ConcatDataset([RainData('training', **self.params), self.val_ds])
            else:
                self.train_ds = RainData('training', **self.params)
                self.val_ds = RainData('validation', **self.params)
            print(f"Training dataset size: {len(self.train_ds)}")
        if mode in ['val']:
            print("Loading VALIDATION dataset -- as test")
            self.val_ds = RainData('validation', **self.params)
        if mode in ['predict']:
            print("Loading PREDICTION/TEST dataset -- as test")
            self.test_ds = RainData('test', **self.params)

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(dataset,
                        batch_size=self.training_params['batch_size'],
                        num_workers=self.training_params['n_workers'],
                        shuffle=shuffle,
                        pin_memory=pin, prefetch_factor=2,
                        persistent_workers=False)
        return dl

    def train_dataloader(self):
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)

    def val_dataloader(self):
        return self.__load_dataloader(self.val_ds, shuffle=False, pin=True)

    def test_dataloader(self):
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)


def load_model(Model, params, checkpoint_path='') -> pl.LightningModule:
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    p = {**params['experiment'], **params['dataset'], **params['train']}
    if checkpoint_path == '':
        print('-> Modelling from scratch!  (no checkpoint loaded)')
        model = Model(params['model'], p)
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path, UNet_params=params['model'], params=p)
    return model


def get_trainer(gpus, params, mode):
    """ get the trainer, modify here its options:
        - save_top_k
     """
    max_epochs = params['train']['max_epochs']
    # max_epochs = 1
    print("Trainig for", max_epochs, "epochs")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', save_top_k=90, save_last=True,
                                          filename='{epoch:02d}-{val_loss_epoch:.6f}')

    parallel_training = None
    ddpplugin = None
    if gpus[0] == -1:
        gpus = None
    elif len(gpus) > 1:
        parallel_training = 'ddp'
    ##        ddpplugin = DDPPlugin(find_unused_parameters=True)
    print(f"====== process started on the following GPUs: {gpus} ======")
    date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
    version = params['experiment']['name']
    version = version + '_' + date_time

    # SET LOGGER
    # if params['experiment']['logging']: 
    #     tb_logger = pl_loggers.TensorBoardLogger(save_dir=params['experiment']['experiment_folder'],name=params['experiment']['sub_folder'], version=version, log_graph=True)
    # else: 
    #     tb_logger = False

    if params['experiment']['logging'] and mode != "predict" and mode != "val":
        # Create a WandbLogger instead of TensorBoardLogger
        wandb_logger = WandbLogger(
            project='w4c23',
            save_dir=params['experiment']['experiment_folder'],
            name=params['experiment']['sub_folder'],
        )
    else:
        wandb_logger = False
    if mode == "predict" or mode == "val" or len(gpus) <= 1:
        strategy = None
    else:
        strategy = "ddp"
    if params['train']['early_stopping']:
        early_stop_callback = EarlyStopping(monitor="val_loss_epoch",
                                            patience=params['train']['patience'],
                                            mode="min")
        callback_funcs = [checkpoint_callback, ModelSummary(max_depth=2), early_stop_callback]
    else:
        callback_funcs = [checkpoint_callback, ModelSummary(max_depth=2)]

    trainer = pl.Trainer(devices=gpus, max_epochs=max_epochs,
                         gradient_clip_val=params['model']['gradient_clip_val'],
                         gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
                         accelerator="gpu",
                         callbacks=callback_funcs, logger=wandb_logger,
                         # profiler='simple',
                         # fast_dev_run=3,
                         # log_every_n_steps=1,
                         precision=params['experiment']['precision'],
                         strategy=strategy
                         )

    return trainer


def to_number(y_hat, nums=None, thres=None):
    if nums is None:
        nums = torch.tensor([0, 0.6, 3, 7.5, 12.5, 16]).reshape(1, 6, 1, 1, 1).to(y_hat.device)
    num_classes = 6
    y_hat = F.softmax(y_hat, dim=1)
    if thres is not None:
        y_sum = 1 - torch.cumsum(y_hat, dim=1)
        y_hat = torch.argmax((y_sum < torch.tensor(thres + [2], device=y_sum.device).reshape(1, 6, 1, 1, 1)).long(),
                             dim=1)
    else:
        y_hat = torch.argmax(y_hat, dim=1)
    y_hat = F.one_hot(y_hat, num_classes=num_classes).permute(0, 4, 1, 2, 3)
    ret = torch.sum(y_hat * nums, axis=1, keepdim=True)
    return y_hat, ret


def do_predict(trainer, model, predict_params, test_data):
    ret = 0
    test_batch = trainer.predict(model, dataloaders=test_data)
    scores = torch.cat([b[0] for b in test_batch])
    _, scores = to_number(scores)
    tensor_to_submission_file(scores, predict_params)
    return ret


def do_test(trainer, model, test_data):
    scores = trainer.test(model, dataloaders=test_data)

def do_val(trainer, model, test_data):
    scores = trainer.validate(model, dataloaders=test_data)

def train(params, gpus, mode, checkpoint_path, model=UNetModel, tune=True):
    """ main training/evaluation method
    """
    # ------------
    # model & data
    # ------------
    get_cuda_memory_usage(gpus)
    data = DataModule(params['dataset'], params['train'], mode)
    model = load_model(model, params, checkpoint_path)
    # model.summary()
    # ------------
    # Add your models here
    # ------------

    # ------------
    # trainer
    # ------------
    trainer = get_trainer(gpus, params, mode)
    get_cuda_memory_usage(gpus)
    # ------------
    # train & final validation
    # ------------
    ret = None
    if mode == 'train':
        print("------------------")
        print("--- TRAIN MODE ---")
        print("------------------")
        trainer.fit(model, data)

    if mode == "val":
        # ------------
        # VALIDATE
        # ------------
        print("---------------------")
        print("--- VALIDATE MODE ---")
        print("---------------------")
        do_val(trainer, model, data.val_dataloader())

    if mode == 'predict':
        # ------------
        # PREDICT
        # ------------
        print("--------------------")
        print("--- PREDICT MODE ---")
        print("--------------------")
        print("REGIONS!:: ", params["dataset"]["regions"], params["predict"]["region_to_predict"])
        if params["predict"]["region_to_predict"] not in params["dataset"]["regions"]:
            print(
                "EXITING... \"regions\" and \"regions to predict\" must indicate the same region name in your config file.")
        else:
            model.eval()
            ret = do_predict(trainer, model, params["predict"], data.test_dataloader())

    get_cuda_memory_usage(gpus)
    return ret


def update_params_based_on_args(options):
    config_p = os.path.join('models/configurations', options.config_path)
    params = load_config(config_p)

    if options.name != '':
        print(params['experiment']['name'])
        params['experiment']['name'] = options.name
    # print(params['model'])
    return params


def set_parser():
    """ set custom parser """

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--config_path", type=str, required=False, default='./configurations/config_basline.yaml',
                        help="path to config-yaml")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=1,
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train',
                        help="choose mode: train (default) / val / predict")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='',
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=False, default='',
                        help="Set the name of the experiment")
    parser.add_argument("-a", "--generate_all", action="store_true", required=False, default=False,
                        help="Set the name of the experiment")
    parser.add_argument("--tune", action="store_true", required=False, default=False,
                        help="Set the name of the experiment")

    parser.add_argument("--test_region", type=str, required=False, default=None)
    parser.add_argument("--test_year", type=str, required=False, default=None)
    parser.add_argument("--test_bz", type=int, required=False, default=None)

    return parser


def main():
    parser = set_parser()
    options = parser.parse_args()

    params = update_params_based_on_args(options)
    if options.test_region:
        params['dataset']['regions'] = [options.test_region]
        params['predict']['region_to_predict'] = options.test_region
    if options.test_year:
        params['dataset']['years'] = [options.test_year]
        params['predict']['year_to_predict'] = options.test_year
    if options.test_bz:
        params['train']['batch_size'] = options.test_bz
    if options.generate_all:
        print("generate on all the regions and years")
        original_regions = copy.deepcopy(params['dataset']['regions'])
        original_year = copy.deepcopy(params['dataset']['years'])
        cms = []
        description = []
        for region in original_regions:
            for year in original_year:
                params['dataset']['regions'] = [region]
                params['predict']['region_to_predict'] = region
                params['dataset']['years'] = [year]
                params['predict']['year_to_predict'] = year
                ret = train(params, options.gpus, options.mode, options.checkpoint, options.tune)
                description.append([region, year])
                cms.append(ret)

        for j in range(len(cms)):
            ret = cms[j]
            print(f"{description[j][0]},{description[j][1]}: ")
            csi_list = []
            for i in range(ret.size(0)):
                recall, precision, F1, acc, csi = recall_precision_f1_acc(cm=ret[i])
                csi_list.append(csi)
            print(f"csi : {np.mean(csi_list)}")
        csi_list = []
        for class_cm in torch.sum(torch.stack(cms, dim=0), dim=0):
            recall, precision, F1, acc, csi = recall_precision_f1_acc(cm=class_cm)
            csi_list.append(csi)
        print(f"total csi : {np.mean(csi_list)}")
    else:

        train(params, options.gpus, options.mode, options.checkpoint, tune=options.tune)


if __name__ == "__main__":
    main()
    """ examples of usage:

    1) train from scratch on one GPU
    python train.py --gpus 2 --mode train --config_path config_baseline.yaml --name baseline_train

    2) train from scratch on four GPUs
    python train.py --gpus 0 1 2 3 --mode train --config_path config_baseline.yaml --name baseline_train
    
    3) fine tune a model from a checkpoint on one GPU
    python train.py --gpus 1 --mode train  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_tune
    
    4) evaluate a trained model from a checkpoint on two GPUs
    python train.py --gpus 0 1 --mode val  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_validate

    5) generate predictions (plese note that this mode works only for one GPU)
    python train.py --gpus 1 --mode predict  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"

    """
