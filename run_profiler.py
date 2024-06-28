#!/usr/bin/env python
# coding: utf-8
import os
import time

import comet_ml
from tqdm.auto import tqdm
import torch

import lightning as L
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from config import Config, load_config
from data import build_datasets, build_loader, build_detailed_test_dataloaders
from model import build_model


def log_to_comet():
    return False

def log_to_wandb():
    return False

def log_to_csv():
    return False

PROFILING_MAX_STEPS = 100

def build_trainer(config, experiment_name, checkpoint_path, max_steps=PROFILING_MAX_STEPS):
    loggers = []
    if torch.cuda.device_count() > 1:
        print(f'Working with: {torch.cuda.device_count()} GPUs')
        return Trainer(max_steps=max_steps, # Profiling
                       profiler="simple", # Profiling
                       accelerator="gpu",
                       devices=torch.cuda.device_count(),
                       strategy='ddp_find_unused_parameters_false',
                       precision="16",
                       logger=loggers,
                       )

    else:
        print(f'Working with: {torch.cuda.device_count()} GPU')
        return Trainer(max_steps=max_steps, # Profiling
                       profiler="simple", # Profiling
                       accelerator="gpu",
                       devices=torch.cuda.device_count(),
                       logger=loggers,
                       )

def build_tester(config, experiment_name, max_steps=PROFILING_MAX_STEPS):
    loggers = []
    print(f'Working with: {torch.cuda.device_count()} GPU')
    return Trainer(max_steps=max_steps, # Profiling
                   profiler="simple", # Profiling
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

    train_dataset, test_dataset, systematic_dataset = build_datasets(config)
    config.pad_idx = train_dataset.pad_idx

    train_loader = build_loader(train_dataset, config, shuffle=True)
    test_loader = build_loader(test_dataset, config, shuffle=False)
    systematic_loader = build_loader(systematic_dataset, config, shuffle=False)
    test_detailed_loaders = build_detailed_test_dataloaders(test_dataset, config) # type_of_tokens_to_test
    systematic_detailed_loaders = build_detailed_test_dataloaders(systematic_dataset, config) # type_of_tokens_to_test

    training_model = build_model(config)

    checkpoint_path = f"outputs/{experiment_name}/"
    resume_from_path = None
    # if config.resume_training:
    #     resume_from_path = f'{checkpoint_path}/last.ckpt'

    trainer = build_trainer(config, experiment_name, checkpoint_path)
    trainer_one_step = build_trainer(config, experiment_name, checkpoint_path, max_steps=1)
    tester = build_tester(config, experiment_name)

    print('='*90)
    print('='*90)
    print('Profile time to iterate through all training samples:')
    start = time.time()
    for _ in tqdm(train_loader):
        pass
    end = time.time()
    duration = end - start
    steps_per_second =  len(train_loader) / duration
    print(f'Iterate through training set in: {duration:.1f} seconds')
    print(f'{steps_per_second:.1f} step per seconds')
    print('='*90)
    print('='*90)

    print()

    print('='*90)
    print('='*90)
    print('Profile One Step on Train:')
    trainer_one_step.fit(training_model, train_loader,
                val_dataloaders=[
                    test_loader, systematic_loader,
                    test_detailed_loaders['color'], systematic_detailed_loaders['color'],
                    test_detailed_loaders['shapes'], systematic_detailed_loaders['shapes'],
                    ])
    print('='*90)
    print('='*90)

    print()

    print('='*90)
    print('='*90)
    print('Profile by Lightning Train:')
    trainer.fit(training_model, train_loader,
                val_dataloaders=[
                    test_loader, systematic_loader,
                    test_detailed_loaders['color'], systematic_detailed_loaders['color'],
                    test_detailed_loaders['shapes'], systematic_detailed_loaders['shapes'],
                    ])
    print('='*90)
    print('='*90)

    print()

    print('='*90)
    print('='*90)
    print('Profile by Lightning Train:')
    tester.test(training_model,
                dataloaders=[test_loader, systematic_loader])

    for entity_to_test in test_detailed_loaders:
        detailed_test_loader = test_detailed_loaders[entity_to_test]
        detailed_systematic_loader = systematic_detailed_loaders[entity_to_test]

        training_model.set_exp_prefix(entity_to_test)
        tester.test(training_model,
                    dataloaders=[detailed_test_loader, detailed_systematic_loader])


if __name__ == "__main__":
    config = load_config()
    main(config)