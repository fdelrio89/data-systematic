#!/usr/bin/env python
# coding: utf-8
import os

try:
    import comet_ml
except ImportError:
    comet_ml = None
try:
    import wandb
except ImportError:
    wandb = None

import torch

import lightning as L
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from config import load_config
from data import build_datasets, build_loader, build_detailed_test_dataloaders
from model import build_model
from compute_results import compute_results


PROJECT_NAME='data-systematicity'


def log_to_comet():
    if comet_ml is None:
        return False
    return 'COMET_API_KEY' in os.environ and 'COMET_WORKSPACE' in os.environ


def log_to_wandb():
    if wandb is None:
        return False
    return 'LOG_TO_WANDB' in os.environ


def build_trainer(config, experiment_name, checkpoint_path, callbacks=None):
    callbacks = [] if callbacks is None else callbacks
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    save_top_k = -1
    every_n_epochs = 50
    callbacks.append(ModelCheckpoint(dirpath=checkpoint_path,
                                     save_top_k=save_top_k,
                                     monitor="val_loss/dataloader_idx_0",
                                     every_n_epochs=every_n_epochs,
                                     save_last=True))

    loggers = []
    if log_to_comet():
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name=PROJECT_NAME,
            experiment_name=experiment_name,
            experiment_key=config.comet_experiment_key,
        )
        comet_logger.log_hyperparams(vars(config))
        config.comet_experiment_key = comet_logger.experiment.get_key()
        loggers.append(comet_logger)

    if log_to_wandb():
        try:
            wandb_logger = WandbLogger(
                project=PROJECT_NAME,
                name=experiment_name,
                version=config.wandb_experiment_id,
            )
            wandb_logger.log_hyperparams(vars(config))
            config.wandb_experiment_id = wandb_logger.version
            loggers.append(wandb_logger)
        except wandb.errors.errors.CommError as e:
            print(e)
            pass

    base_val_checks = 10
    check_val_every_n_epoch = int(base_val_checks / config.trainset_subset)

    print(f'Working with: {torch.cuda.device_count()} GPUs')
    if torch.cuda.device_count() > 1:
        return Trainer(max_epochs=config.max_epochs,
                       accelerator="gpu",
                       devices=torch.cuda.device_count(),
                       strategy='ddp_find_unused_parameters_false',
                       precision="16",
                       logger=loggers,
                       callbacks=callbacks,
                       check_val_every_n_epoch=check_val_every_n_epoch)

    else:
        return Trainer(max_epochs=config.max_epochs,
                       accelerator="gpu",
                       devices=torch.cuda.device_count(),
                       precision="16",
                       logger=loggers,
                       callbacks=callbacks,
                       check_val_every_n_epoch=check_val_every_n_epoch)


def build_tester(config, experiment_name):
    loggers = []
    if log_to_comet():
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name=PROJECT_NAME,
            experiment_key=config.comet_experiment_key,
            experiment_name=experiment_name,
        )
        config.comet_experiment_key = comet_logger.experiment.get_key()
        loggers.append(comet_logger)

    if log_to_wandb():
        try:
            wandb_logger = WandbLogger(
                project=PROJECT_NAME,
                name=experiment_name,
                version=config.wandb_experiment_id,
            )
            config.wandb_experiment_id = wandb_logger.version
            loggers.append(wandb_logger)
        except wandb.errors.errors.CommError as e:
            print(e)
            pass

    print(f'Working with: {torch.cuda.device_count()} GPU')
    return Trainer(max_epochs=config.max_epochs,
                   accelerator="gpu",
                   devices=1,
                   precision="16",
                   logger=loggers)


def update_config_with_data(config, dataset):
    config.pad_idx = dataset.pad_idx
    config.n_tokens = dataset.n_tokens


def build_data_for_training(config, train_callbacks):
    train_dataset, test_dataset, systematic_dataset, cmn_systematic_dataset = build_datasets(config)
    update_config_with_data(config, train_dataset)

    train_loader = build_loader(train_dataset, config, shuffle=True)
    test_loader = build_loader(test_dataset, config, shuffle=False)
    systematic_loader = build_loader(systematic_dataset, config, shuffle=False)
    cmn_systematic_loader = build_loader(cmn_systematic_dataset, config, shuffle=False)
    test_detailed_loaders = build_detailed_test_dataloaders(test_dataset, config) # type_of_tokens_to_test
    systematic_detailed_loaders = build_detailed_test_dataloaders(systematic_dataset, config) # type_of_tokens_to_test
    cmn_systematic_detailed_loaders = build_detailed_test_dataloaders(cmn_systematic_dataset, config) # type_of_tokens_to_test
    train_data_args = {
        'train_dataloaders': train_loader,
        'val_dataloaders': [
                test_loader, systematic_loader,
                test_detailed_loaders['color'],
                systematic_detailed_loaders['color'],
                test_detailed_loaders['shapes'],
                systematic_detailed_loaders['shapes'],
                cmn_systematic_detailed_loaders['color'],
                cmn_systematic_detailed_loaders['shapes'],
                ],
    }
    test_data_args = {
        'dataloaders': [test_loader, systematic_loader, cmn_systematic_loader],
    }

    print('train_dataset size = ', len(train_dataset))
    print('batch_size size    = ', config.batch_size)
    print('train_loader size  = ', len(train_loader))

    return train_data_args, test_data_args, test_detailed_loaders, systematic_detailed_loaders


def main(config):
    seed_everything(config.seed, workers=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = config.experiment_name
    train_callbacks = []
    (train_data_args,
     test_data_args,
     test_detailed_loaders,
     systematic_detailed_loaders) = build_data_for_training(config, train_callbacks)

    training_model = build_model(config)

    checkpoint_path = f"{config.outputs_path}/{experiment_name}/"
    resume_from_path = None
    if config.resume_training:
        resume_from_path = f'{checkpoint_path}/last.ckpt'

    trainer = build_trainer(config, experiment_name, checkpoint_path, callbacks=train_callbacks)
    tester = build_tester(config, experiment_name)

    trainer.fit(
        training_model,
        ckpt_path=resume_from_path,
        **train_data_args,
    )
    tester.test(
        training_model,
        ckpt_path=f'{checkpoint_path}/last.ckpt',
        **test_data_args,
    )

    for entity_to_test in test_detailed_loaders:
        detailed_test_loader = test_detailed_loaders[entity_to_test]
        detailed_systematic_loader = systematic_detailed_loaders[entity_to_test]

        training_model.set_exp_prefix(entity_to_test)
        tester.test(training_model,
                    ckpt_path=f'{checkpoint_path}/last.ckpt',
                    dataloaders=[detailed_test_loader, detailed_systematic_loader])


if __name__ == "__main__":
    config = load_config()
    main(config)
    print('Computing Results...')
    compute_results(config.experiment_name, only_performance=(not config.multimodal_pretraining))
