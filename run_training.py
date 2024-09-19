#!/usr/bin/env python
# coding: utf-8
import os

import comet_ml
import torch

import lightning as L
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from config import load_config
from data import build_datasets, build_loader, build_detailed_test_dataloaders
from data import CurriculumData, CurriculumScheduler, build_common_colors_subset
from model import build_model
from compute_results import compute_results


def log_to_comet():
    # return 'COMET_API_KEY' in os.environ and 'COMET_WORKSPACE' in os.environ
    return False

def log_to_wandb():
    # return 'COMET_API_KEY' in os.environ and 'COMET_WORKSPACE' in os.environ
    return True

def log_to_csv():
    return False

def build_trainer(config, experiment_name, checkpoint_path, callbacks=None):
    callbacks = [] if callbacks is None else callbacks
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    save_top_k = -1 if config.multimodal_pretraining else 1 # if pre-training save all checkpoints
    callbacks.append(ModelCheckpoint(dirpath=checkpoint_path,
                                     save_top_k=save_top_k,
                                     monitor="val_loss/dataloader_idx_0",
                                     every_n_epochs=50,
                                     save_last=True))
    # if config.use_curriculum:
    #     schedule = [tuple(range(3,i+1)) for i in range(3,10)] + [(0,)] # 3 to 10 ojects
    #     callbacks.append(CurriculumScheduler(schedule, config.max_epochs))

    loggers = []
    if log_to_comet():
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name='systematic-text-representation',
            experiment_name=experiment_name,
            experiment_key=config.comet_experiment_key,
        )
        comet_logger.log_hyperparams(vars(config))
        config.comet_experiment_key = comet_logger.experiment.get_key()
        loggers.append(comet_logger)

    if log_to_wandb():
        wandb_logger = WandbLogger(
            project='systematic-text-representation',
            name=experiment_name,
            version=config.wandb_experiment_id,
        )
        wandb_logger.log_hyperparams(vars(config))
        config.wandb_experiment_id = wandb_logger.version
        loggers.append(wandb_logger)

    if log_to_csv():
        output_path = f"outputs/{experiment_name}/"
        csv_logger = CSVLogger(
            output_path,
            name='restults.csv'
        )
        csv_logger.log_hyperparams(vars(config))
        loggers.append(csv_logger)

    reload_dataloaders_every_n_epochs = 1 if config.use_curriculum else 0
    if torch.cuda.device_count() > 1:
        print(f'Working with: {torch.cuda.device_count()} GPUs')
        return Trainer(max_epochs=config.max_epochs,
                       accelerator="gpu",
                       devices=torch.cuda.device_count(),
                       strategy='ddp_find_unused_parameters_false',
                       precision="16",
                       logger=loggers,
                       callbacks=callbacks,
                       reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs)

    else:
        print(f'Working with: {torch.cuda.device_count()} GPU')
        return Trainer(max_epochs=config.max_epochs,
                       accelerator="gpu",
                       # devices="auto",
                       devices=torch.cuda.device_count(),
                       logger=loggers,
                       callbacks=callbacks,
                       reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs)

def build_tester(config, experiment_name):
    loggers = []
    if log_to_comet():
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name='systematic-text-representation',
            experiment_key=config.comet_experiment_key,
            experiment_name=experiment_name,
        )
        config.comet_experiment_key = comet_logger.experiment.get_key()
        loggers.append(comet_logger)

    if log_to_wandb():
        wandb_logger = WandbLogger(
            project='systematic-text-representation',
            name=experiment_name,
            version=config.wandb_experiment_id,
        )
        # wandb_logger.log_hyperparams(vars(config))
        config.wandb_experiment_id = wandb_logger.version
        loggers.append(wandb_logger)

    print(f'Working with: {torch.cuda.device_count()} GPU')
    return Trainer(max_epochs=config.max_epochs,
                   accelerator="gpu",
                   devices=1,
                   precision="16",
                   logger=loggers)


def main(config):
    seed_everything(config.seed, workers=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    experiment_name = os.environ.get("EXP_NAME", "default")
    train_callbacks = []

    train_dataset, test_dataset, systematic_dataset, cmn_systematic_dataset = build_datasets(config)
    config.pad_idx = train_dataset.pad_idx
    config.n_tokens = train_dataset.n_tokens
    
    if config.use_curriculum:
        train_subsets = build_common_colors_subset(train_dataset, config)
        train_dataset = CurriculumData(train_subsets)
        train_callbacks.append(CurriculumScheduler(train_dataset, config.max_epochs))

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

    training_model = build_model(config)

    checkpoint_path = f"outputs/{experiment_name}/"
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
    experiment_name = os.environ.get("EXP_NAME", "default")
    compute_results(experiment_name)