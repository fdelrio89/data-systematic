#!/usr/bin/env python
# coding: utf-8
import os

import comet_ml

import torch
from torch.utils.data import DataLoader

from config import Config, load_config
from data import CLEVRSplit, CLEVRTextSplit
from model import Model, TextualModel, TrainingModel


import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def log_to_comet():
    return 'COMET_API_KEY' in os.environ and 'COMET_WORKSPACE' in os.environ

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = Config()
config = load_config()
if config.use_txt_scene:
    train_dataset, test_dataset, systematic_dataset = CLEVRTextSplit.build_splits(config)
else:
    train_dataset, test_dataset, systematic_dataset = CLEVRSplit.build_splits(config)

config.pad_idx = train_dataset.pad_idx

dlkwargs = {
    'batch_size': config.batch_size,
    'num_workers': int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", 4)),
    'pin_memory': torch.cuda.is_available(),
}

train_loader = DataLoader(train_dataset, shuffle=True, **dlkwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **dlkwargs)
systematic_loader = DataLoader(systematic_dataset, shuffle=False, **dlkwargs)

if config.use_txt_scene:
    model = TextualModel(config)
else:
    model = Model(config)
training_model = TrainingModel(model, config)

experiment_name = os.environ.get("EXP_NAME", "default")

checkpoint_path = f"outputs/{experiment_name}/"
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path, save_top_k=1, monitor="val_loss/dataloader_idx_0", every_n_epochs=1, save_last=True)

resume_from_path = None
if config.resume_training:
    resume_from_path = f'{checkpoint_path}/last.ckpt'

comet_logger = None
if log_to_comet():
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),
        project_name='systematic-text-representation',
        experiment_key=os.environ.get("COMET_EXPERIMENT_KEY"),
        experiment_name=experiment_name,
    )
    comet_logger.log_hyperparams(vars(config))

if config.profile:
    trainer = Trainer(max_steps=2000, accelerator="gpu", devices=1, profiler="simple",
                      enable_progress_bar=False)
else:
    trainer = Trainer(max_epochs=config.max_epochs, accelerator="gpu", devices=1,
                        logger=comet_logger, callbacks=[checkpoint_callback])


trainer.fit(training_model, train_loader, val_dataloaders=[test_loader, systematic_loader],
            ckpt_path=resume_from_path)

trainer.test(training_model, dataloaders=[test_loader, systematic_loader])
