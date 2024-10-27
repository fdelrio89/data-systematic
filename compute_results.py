from argparse import ArgumentParser
from itertools import product
from functools import partial
import json
import os
import pprint
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics.cluster import normalized_mutual_info_score

from config import load_config
from data import build_datasets
from data import CollatorForMaskedSelectedTokens, CollatorForMaskedRandomSelectedTokens, CollatorForMaskedVQA
from data import ALL_POSSIBLE_COLORS
from model import MultimodalModel, MultimodalPretrainingModel
from utils import load_checkpoint
from lightning import Trainer
from tqdm.auto import tqdm

from p_score import compute_p_scores
from o_score import compute_o_scores
from probing import compute_probe_metrics


pp = pprint.PrettyPrinter(indent=2)

def load_training_model(config, checkpoint):
    model = MultimodalModel(config)
    training_model = MultimodalPretrainingModel(model, config)
    training_model.load_state_dict(checkpoint['state_dict'])
    return training_model

def build_collate_fns(dataset, config, dont_mask_spheres=False):
    processor = dataset.processor
    vocab = processor.vocabulary

    relation_tokens = sorted(
        [vocab[w] for w in ['left', 'right', 'behind', 'front'] if w in vocab])
    color_tokens = sorted(
        [vocab[w] for w in ALL_POSSIBLE_COLORS if w in vocab])
    #     [vocab[w] for w in ['blue', 'brown', 'cyan', 'green', 'red', 'purple', 'yellow', 'gray']])
    shapes_tokens = sorted(
        [vocab[w] for w in ['cylinder', 'sphere', 'cube'] if w in vocab])
    materials_tokens = sorted(
        [vocab[w] for w in ['metal', 'rubber'] if w in vocab])
    size_tokens = sorted(
        [vocab[w] for w in ['small', 'large'] if w in vocab])

    random_baseline = {
        'relation':  1 / len(relation_tokens),
        'color':  1 / len(color_tokens),
        'shapes':  1 / len(shapes_tokens),
        'materials':  1 / len(materials_tokens),
        'size':  1 / len(size_tokens),
        'identity':  1 / len(processor.vocabulary),
    }
    
    if not config.multimodal_pretraining:
        vqa_collator = CollatorForMaskedVQA(config, dataset.processor)
        return {
            'random_without_testing_spheres': [
                ('color', vqa_collator),
                ('shapes', vqa_collator),
                ('materials', vqa_collator),
                ('size', vqa_collator),
        ]}, random_baseline

    collate_fns = {
        'selected': [
            ('color', CollatorForMaskedSelectedTokens(
                config, processor, tokens=color_tokens, dont_mask_spheres=dont_mask_spheres)),
            ('shapes', CollatorForMaskedSelectedTokens(
                config, processor, tokens=shapes_tokens, dont_mask_spheres=dont_mask_spheres)),
            ('materials', CollatorForMaskedSelectedTokens(
                config, processor, tokens=materials_tokens, dont_mask_spheres=dont_mask_spheres)),
            ('size', CollatorForMaskedSelectedTokens(
                config, processor, tokens=size_tokens, dont_mask_spheres=dont_mask_spheres)),
        ],
        'random': [
            ('color', CollatorForMaskedRandomSelectedTokens(
                config, processor, tokens=color_tokens, p=0.3, dont_mask_spheres=dont_mask_spheres)),
            ('shapes', CollatorForMaskedRandomSelectedTokens(
                config, processor, tokens=shapes_tokens, p=0.3, dont_mask_spheres=dont_mask_spheres)),
            ('materials', CollatorForMaskedRandomSelectedTokens(
                config, processor, tokens=materials_tokens, p=0.3, dont_mask_spheres=dont_mask_spheres)),
            ('size', CollatorForMaskedRandomSelectedTokens(
                config, processor, tokens=size_tokens, p=0.3, dont_mask_spheres=dont_mask_spheres)),
        ],
        'random_without_testing_spheres' : [
            ('color', CollatorForMaskedRandomSelectedTokens(
                config, processor, tokens=color_tokens, p=0.3, dont_mask_spheres=True)),
            ('shapes', CollatorForMaskedRandomSelectedTokens(
                config, processor, tokens=shapes_tokens, p=0.3, dont_mask_spheres=True)),
            ('materials', CollatorForMaskedRandomSelectedTokens(
                config, processor, tokens=materials_tokens, p=0.3, dont_mask_spheres=True)),
            ('size', CollatorForMaskedRandomSelectedTokens(
                config, processor, tokens=size_tokens, p=0.3, dont_mask_spheres=True)),
        ]
    }
    return collate_fns, random_baseline

def build_loader(dataset, collate_fn):
    dlkwargs = {
        'batch_size': 512,
        'num_workers': int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
        'pin_memory': torch.cuda.is_available(),
    }
    return DataLoader(dataset, shuffle=True, collate_fn=collate_fn, **dlkwargs)


def build_test_data(config):
    test_datasets = {}
    systematic_datasets = {}
    cmn_systematic_datasets = {}
    
    if config.multimodal_pretraining:
        train_dataset, test_dataset, systematic_dataset, common_systematic_dataset = build_datasets(config)
        config.pad_idx = train_dataset.pad_idx
        config.n_tokens = train_dataset.n_tokens
        
        property_queries = ['shapes', 'size', 'color', 'materials']
        test_datasets = {prop: test_dataset for prop in property_queries}
        systematic_datasets = {prop: systematic_dataset for prop in property_queries}
        cmn_systematic_datasets = {prop: common_systematic_dataset for prop in property_queries}
    else:
        (train_dataset, 
         test_datasets['all'],
         test_datasets['shapes'],
         test_datasets['size'],
         test_datasets['color'],
         test_datasets['materials'], 
         systematic_datasets['all'],
         systematic_datasets['shapes'],
         systematic_datasets['size'],
         systematic_datasets['color'],
         systematic_datasets['materials'], 
         cmn_systematic_datasets['all'],
         cmn_systematic_datasets['shapes'],
         cmn_systematic_datasets['size'],
         cmn_systematic_datasets['color'],
         cmn_systematic_datasets['materials']
         ) = build_datasets(config)
        
        config.pad_idx = train_dataset.pad_idx
        config.n_tokens = train_dataset.n_tokens

    return train_dataset, test_datasets, systematic_datasets, cmn_systematic_datasets

def compute_performance_results(exp_name):
    checkpoint = load_checkpoint(exp_name)
    config = load_config(exp_name)

    train_dataset, test_datasets, systematic_datasets, common_systematic_datasets = build_test_data(config)

    training_model = load_training_model(config, checkpoint)
    trainer = Trainer(max_epochs=config.max_epochs,
                    accelerator="gpu",
                    devices=torch.cuda.device_count()
    )

    all_results = {}
    collate_fns, random_baseline = build_collate_fns(train_dataset, config)
    for type_, fns_by_category in collate_fns.items():
        results = {}
        for name, collate_fn in fns_by_category:
            train_loader = build_loader(train_dataset, collate_fn=collate_fn)
            test_loader = build_loader(test_datasets[name], collate_fn=collate_fn)
            systematic_loader = build_loader(systematic_datasets[name], collate_fn=collate_fn)
            common_systematic_loader = build_loader(common_systematic_datasets[name], collate_fn=collate_fn)

            test_results = trainer.test(training_model, dataloaders=[test_loader, systematic_loader])
            raw_results = trainer.test(training_model, dataloaders=[train_loader, common_systematic_loader])
            train_results = [
                {k.replace('test_', 'train_'): v for k, v in raw_results[0].items()}]
            common_results = [
                {k.replace('systematic_test', 'common_systematic_test'): v for k, v in raw_results[1].items()}]
            results[name] = test_results + train_results + common_results

        all_results[type_] = results
        

    all_results['config'] = vars(config)
    all_results['random_baseline'] = random_baseline
        
    return all_results


def compute_results(exp_name, only_performance=False):
    all_results = compute_performance_results(exp_name)

    if not only_performance:
        print('Computing NMI Scores')
        all_results['sampled_nmi_scores'] = compute_nmi(exp_name, use_complete_dataset=False)
        all_results['nmi_scores'] = compute_nmi(exp_name, use_complete_dataset=True)

        print('Computing P-Scores')
        all_results['p_scores'] = compute_p_score(exp_name)
        all_results['p_scores_within_task'] = compute_p_score(exp_name, within_task=True)
        
        print('Computing O-Scores')
        all_results['o_scores'] = compute_o_score(exp_name)
        all_results['o_scores_within_task'] = compute_o_score(exp_name, triplet_within_task=True)
        all_results['o_scores_within_color_shape'] = compute_o_score(exp_name, triplet_within_color_shape=True)
        
        # all_results['correlation_scores'] = compute_correlation(exp_name, use_complete_dataset=True)
        print('Computing Probing')
        all_results['probing_metrics'] = compute_probing(exp_name)
    
    with open(f'outputs/results/{exp_name}.json', 'w') as fp:
        json.dump(all_results, fp)


def update_results_without_spheres(exp_name):
    checkpoint = load_checkpoint(exp_name)
    config = load_config(exp_name)

    # workspace_path = ''
    config.vocabulary_path = config.vocabulary_path.replace('/workspace1/' ,'/workspace/')
    config.base_path = config.base_path.replace('/workspace1/' ,'/workspace/')
    # config.vocabulary_path = config.vocabulary_path.replace('/workspace/' ,'/workspace1/')
    # config.base_path = config.base_path.replace('/workspace/' ,'/workspace1/')

    train_dataset, test_dataset, systematic_dataset, common_systematic_dataset = build_datasets(config)
    config.pad_idx = train_dataset.pad_idx

    training_model = load_training_model(config, checkpoint)
    trainer = Trainer(max_epochs=config.max_epochs,
                    accelerator="gpu",
                    devices=torch.cuda.device_count()
    )

    try:
        with open(f'outputs/results/{exp_name}.json') as fp:
            all_results = json.load(fp)
    except FileNotFoundError as e:
        all_results = {}
        
    if 'random_without_testing_spheres' in all_results:
        return
    
    collate_fns, random_baseline = build_collate_fns(train_dataset, config, dont_mask_spheres=True)
    new_collate_fns = {}
    new_collate_fns['random_without_testing_spheres'] = collate_fns['random']
    for type_, fns_by_category in new_collate_fns.items():
        results = {}
        for name, collate_fn in fns_by_category:
            train_loader = build_loader(train_dataset, collate_fn=collate_fn)
            test_loader = build_loader(test_dataset, collate_fn=collate_fn)
            systematic_loader = build_loader(systematic_dataset, collate_fn=collate_fn)
            common_systematic_loader = build_loader(common_systematic_dataset, collate_fn=collate_fn)

            test_results = trainer.test(training_model, dataloaders=[test_loader, systematic_loader])
            raw_results = trainer.test(training_model, dataloaders=[train_loader, common_systematic_loader])
            train_results = [
                {k.replace('test_', 'train_'): v for k, v in raw_results[0].items()}]
            common_results = [
                {k.replace('systematic_test', 'common_systematic_test'): v for k, v in raw_results[1].items()}]
            results[name] = test_results + train_results + common_results

        all_results[type_] = results

    print(f'Storing {exp_name} new updated results with: "random_without_testing_spheres"')
    with open(f'outputs/results/{exp_name}.json.tmp', 'w') as fp:
        json.dump(all_results, fp)
        
    os.rename(f'outputs/results/{exp_name}.json.tmp', f'outputs/results/{exp_name}.json')
        
def update_results_with_permuted_pixels(exp_name):
    checkpoint = load_checkpoint(exp_name)
    config = load_config(exp_name)
    
    config.permute_pixels = True

    # workspace_path = ''
    config.vocabulary_path = config.vocabulary_path.replace('/workspace/' ,'/workspace1/')
    config.base_path = config.base_path.replace('/workspace/' ,'/workspace1/')

    train_dataset, test_dataset, systematic_dataset, common_systematic_dataset = build_datasets(config)
    config.pad_idx = train_dataset.pad_idx

    training_model = load_training_model(config, checkpoint)
    trainer = Trainer(max_epochs=config.max_epochs,
                    accelerator="gpu",
                    devices=torch.cuda.device_count()
    )

    try:
        with open(f'outputs/results/{exp_name}.json') as fp:
            all_results = json.load(fp)
    except FileNotFoundError as e:
        all_results = {}
    
    collate_fns, random_baseline = build_collate_fns(train_dataset, config)
    
    new_collate_fns = {}
    new_collate_fns['permuted_pixels'] = collate_fns['random']
    for type_, fns_by_category in new_collate_fns.items():
        results = {}
        for name, collate_fn in fns_by_category:
            train_loader = build_loader(train_dataset, collate_fn=collate_fn)
            test_loader = build_loader(test_dataset, collate_fn=collate_fn)
            systematic_loader = build_loader(systematic_dataset, collate_fn=collate_fn)
            common_systematic_loader = build_loader(common_systematic_dataset, collate_fn=collate_fn)

            test_results = trainer.test(training_model, dataloaders=[test_loader, systematic_loader])
            raw_results = trainer.test(training_model, dataloaders=[train_loader, common_systematic_loader])
            train_results = [
                {k.replace('test_', 'train_'): v for k, v in raw_results[0].items()}]
            common_results = [
                {k.replace('systematic_test', 'common_systematic_test'): v for k, v in raw_results[1].items()}]
            results[name] = test_results + train_results + common_results

        all_results[type_] = results

    with open(f'outputs/results/{exp_name}.json.tmp', 'w') as fp:
        json.dump(all_results, fp)
        
    os.rename(f'outputs/results/{exp_name}.json.tmp', f'outputs/results/{exp_name}.json')


def compute_nmi(exp_name, use_complete_dataset=True, store_result=False):
    print(f'Computing NMIS Scores for {exp_name} (use_complete_dataset={use_complete_dataset})')
    
    n_samples = 5_000

    config = load_config(exp_name)
    # pp.pprint(config)
    config.vocabulary_path = config.vocabulary_path.replace('/storage-otro/', '/workspace1/')
    config.base_path = config.base_path.replace('/storage-otro/', '/workspace1/')

    train_dataset, *_ = build_datasets(config)
    config.pad_idx = train_dataset.pad_idx
    config.n_tokens = train_dataset.n_tokens

    pad_idx = train_dataset.processor.vocabulary['[PAD]']

    data_to_iterate = train_dataset
    if not use_complete_dataset:
        indexes = random.sample(list(range(len(train_dataset))), k=n_samples)
        train_subset = Subset(train_dataset, indexes)
        data_to_iterate = train_subset
        
    loader = DataLoader(
        data_to_iterate, batch_size=64, num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))

    all_sizes = []
    all_colors = []
    all_materials = []
    all_shapes = []
    for _, scene in tqdm(loader):
        sizes = scene[:,1:][:,0::5]
        colors = scene[:,1:][:,1::5]
        materials = scene[:,1:][:,2::5]
        shapes = scene[:,1:][:,3::5]
        
        sizes = sizes[sizes != pad_idx].numpy()
        colors = colors[colors != pad_idx].numpy()
        materials = materials[materials != pad_idx].numpy()
        shapes = shapes[shapes != pad_idx].numpy()
        
        all_sizes.append(sizes)
        all_colors.append(colors)
        all_materials.append(materials)
        all_shapes.append(shapes)

    all_sizes = np.concatenate(all_sizes)
    all_colors = np.concatenate(all_colors)
    all_materials = np.concatenate(all_materials)
    all_shapes = np.concatenate(all_shapes)

    task_variables = [
        ('size', all_sizes), ('color', all_colors), ('material', all_materials), ('shape', all_shapes)]

    all_nmi_scores = {}
    for (t0_name, t0_variables), (t1_name, t1_variables) in product(task_variables, repeat=2):
        all_nmi_scores[f'{t0_name}:{t1_name}'] = normalized_mutual_info_score(
                                                                t0_variables, t1_variables)

    if not store_result:
        return all_nmi_scores
        
    with open(f'outputs/results/{exp_name}.json') as fp:
        all_results = json.load(fp)
    
    k_name = 'nmi_scores' if use_complete_dataset else 'sampled_nmi_scores'
    all_results[k_name] = all_nmi_scores
    with open(f'outputs/results/{exp_name}.json.tmp', 'w') as fp:
        json.dump(all_results, fp)
        
    os.rename(f'outputs/results/{exp_name}.json.tmp', f'outputs/results/{exp_name}.json')


def compute_correlation(exp_name, use_complete_dataset=True, store_result=False):
    assert False, "Not implemented"
    print(f'Computing NMIS Scores for {exp_name} (use_complete_dataset={use_complete_dataset})')
    
    n_samples = 5_000

    config = load_config(exp_name)
    # pp.pprint(config)
    config.vocabulary_path = config.vocabulary_path.replace('/storage-otro/', '/workspace1/')
    config.base_path = config.base_path.replace('/storage-otro/', '/workspace1/')

    train_dataset, *_ = build_datasets(config)
    config.pad_idx = train_dataset.pad_idx
    config.n_tokens = train_dataset.n_tokens

    pad_idx = train_dataset.processor.vocabulary['[PAD]']

    data_to_iterate = train_dataset
    if not use_complete_dataset:
        indexes = random.sample(list(range(len(train_dataset))), k=n_samples)
        train_subset = Subset(train_dataset, indexes)
        data_to_iterate = train_subset
        
    loader = DataLoader(
        data_to_iterate, batch_size=64, num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))

    all_sizes = []
    all_colors = []
    all_materials = []
    all_shapes = []
    for _, scene in tqdm(loader):
        sizes = scene[:,1:][:,0::5]
        colors = scene[:,1:][:,1::5]
        materials = scene[:,1:][:,2::5]
        shapes = scene[:,1:][:,3::5]
        
        sizes = sizes[sizes != pad_idx].numpy()
        colors = colors[colors != pad_idx].numpy()
        materials = materials[materials != pad_idx].numpy()
        shapes = shapes[shapes != pad_idx].numpy()
        
        all_sizes.append(sizes)
        all_colors.append(colors)
        all_materials.append(materials)
        all_shapes.append(shapes)

    all_sizes = np.concatenate(all_sizes)
    all_colors = np.concatenate(all_colors)
    all_materials = np.concatenate(all_materials)
    all_shapes = np.concatenate(all_shapes)

    task_variables = [
        ('size', all_sizes), ('color', all_colors), ('material', all_materials), ('shape', all_shapes)]

    all_nmi_scores = {}
    for (t0_name, t0_variables), (t1_name, t1_variables) in product(task_variables, repeat=2):
        all_nmi_scores[f'{t0_name}:{t1_name}'] = normalized_mutual_info_score(
                                                                t0_variables, t1_variables)

    if not store_result:
        return all_nmi_scores
        
    with open(f'outputs/results/{exp_name}.json') as fp:
        all_results = json.load(fp)
    
    k_name = 'nmi_scores' if use_complete_dataset else 'sampled_nmi_scores'
    all_results[k_name] = all_nmi_scores
    with open(f'outputs/results/{exp_name}.json.tmp', 'w') as fp:
        json.dump(all_results, fp)
        
    os.rename(f'outputs/results/{exp_name}.json.tmp', f'outputs/results/{exp_name}.json')
    

def compute_p_score(exp_name, store_result=False, within_task=False):
    print(f'Computing P-Score for {exp_name}')
    
    metric_key = 'p_scores_within_task' if within_task else 'p_scores'
    
    # Updating results
    if store_result and os.path.exists(f'outputs/results/{exp_name}.json'):
        with open(f'outputs/results/{exp_name}.json') as fp:
            all_results = json.load(fp)
        if metric_key in all_results:
            return all_results[metric_key]
    
    all_p_scores = compute_p_scores(exp_name, n_samples=1024, n_seeds_to_try=5,
                                    n_sampled_vertices=10, n_sampled_dichotomies=3500,
                                    dichotomy_within_task=within_task)

    if not store_result:
        return all_p_scores

    with open(f'outputs/results/{exp_name}.json') as fp:
        all_results = json.load(fp)
    
    all_results[metric_key] = all_p_scores
    with open(f'outputs/results/{exp_name}.json.tmp', 'w') as fp:
        json.dump(all_results, fp)
        
    os.rename(f'outputs/results/{exp_name}.json.tmp', f'outputs/results/{exp_name}.json')
    

def compute_o_score(exp_name, store_result=False, triplet_within_task=False, triplet_within_color_shape=False):
    print(f'Computing O-Score for {exp_name}')
    
    if triplet_within_task:
        metric_key = 'o_scores_within_task'
    elif triplet_within_color_shape:
        metric_key = 'o_scores_within_color_shape'
    else:
        metric_key = 'o_scores'

    
    # # Updating results
    # if store_result and os.path.exists(f'outputs/results/{exp_name}.json'):
    #     with open(f'outputs/results/{exp_name}.json') as fp:
    #         all_results = json.load(fp)
    #     if metric_key in all_results:
    #         return all_results[metric_key]
    
    all_o_scores = compute_o_scores(exp_name,
                                    n_samples=1024,
                                    n_seeds_to_try=5,
                                    n_sampled_vertices=10,
                                    n_sampled_triplets=3500,
                                    triplet_within_task=triplet_within_task,
                                    triplet_within_color_shape=triplet_within_color_shape)

    if not store_result:
        return all_o_scores

    with open(f'outputs/results/{exp_name}.json') as fp:
        all_results = json.load(fp)
    
    all_results[metric_key] = all_o_scores
    with open(f'outputs/results/{exp_name}.json.tmp', 'w') as fp:
        json.dump(all_results, fp)
        
    os.rename(f'outputs/results/{exp_name}.json.tmp', f'outputs/results/{exp_name}.json')
    
    
def compute_probing(exp_name, store_result=False):
    print(f'Computing Probing for {exp_name}')
    
    # Updating results
    if store_result and os.path.exists(f'outputs/results/{exp_name}.json'):
        with open(f'outputs/results/{exp_name}.json') as fp:
            all_results = json.load(fp)
        if 'probing_metrics' in all_results:
            return all_results['probing_metrics']
    
    all_probe_metrics = compute_probe_metrics(exp_name,
                                              n_seeds_to_try=5,
                                              n_samples=15_000,
                                              n_objects_to_sample=15_000)

    if not store_result:
        return all_probe_metrics

    with open(f'outputs/results/{exp_name}.json') as fp:
        all_results = json.load(fp)
    
    all_results['probing_metrics'] = all_probe_metrics
    with open(f'outputs/results/{exp_name}.json.tmp', 'w') as fp:
        json.dump(all_results, fp)
        
    os.rename(f'outputs/results/{exp_name}.json.tmp', f'outputs/results/{exp_name}.json')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument('--update_results_without_spheres', action='store_true', default=False)
    parser.add_argument('--update_with_p_scores', action='store_true', default=False)
    parser.add_argument('--update_with_p_scores_within_task', action='store_true', default=False)
    parser.add_argument('--update_with_probe_metrics', action='store_true', default=False)
    parser.add_argument('--update_results_with_permuted_pixels', action='store_true', default=False)
    parser.add_argument('--update_with_o_scores', action='store_true', default=False)
    parser.add_argument('--update_with_o_scores_within_task', action='store_true', default=False)
    parser.add_argument('--update_with_o_scores_within_color_shape', action='store_true', default=False)
    args = parser.parse_args()
        
    # compute_nmi(args.exp_name, use_complete_dataset=False, store_result=True)
    # compute_nmi(args.exp_name, use_complete_dataset=True, store_result=True)

    fn_name = 'Complete Results'
    fn_to_apply = compute_results
    if args.update_results_without_spheres:
        fn_name = 'Results Without Spheres'
        fn_to_apply = update_results_without_spheres
    elif args.update_with_p_scores:
        fn_name = 'P-Score'
        fn_to_apply = partial(compute_p_score, store_result=True)
    elif args.update_with_p_scores_within_task:
        fn_name = 'P-Score Within Task'
        fn_to_apply = partial(compute_p_score, store_result=True, within_task=True)
    elif args.update_with_o_scores:
        fn_name = 'O-Score'
        fn_to_apply = partial(compute_o_score, store_result=True)
    elif args.update_with_o_scores_within_task:
        fn_name = 'O-Score Within Task'
        fn_to_apply = partial(compute_o_score, store_result=True, triplet_within_task=True)
    elif args.update_with_o_scores_within_color_shape:
        fn_name = 'O-Score Within Task'
        fn_to_apply = partial(compute_o_score, store_result=True, triplet_within_color_shape=True)
    elif args.update_with_probe_metrics:
        fn_name = 'Probing'
        fn_to_apply = partial(compute_probing, store_result=True)
    elif args.update_results_with_permuted_pixels:
        fn_name = 'Permuted Pixels'
        fn_to_apply = update_results_with_permuted_pixels

    path_obj = Path(args.exp_name)
    if path_obj.is_dir():
        print(f'Iterating in {args.exp_name}')
        exp_files = list(path_obj.glob('*.json'))
        random.shuffle(exp_files)
        for exp_file in exp_files:
            exp_name = exp_file.stem
            if 'multimodal-pretraining' in exp_name:
                continue
            if 'overloading' in exp_name and 'overloading_to=8.json' not in exp_name:
                continue
            # if 'seed' in exp_name:
            #     continue
            print(f'Updating {fn_name} for {exp_name}')
            try:
                start = time.time()
                fn_to_apply(exp_name)
                end = time.time()
                print(f'Result computed in {end-start:.1f} seconds')
            except FileNotFoundError as e:
                print(f'Unable to update {exp_name}')
                print(str(e))
    else:
        try:
            start = time.time()
            fn_to_apply(args.exp_name)
            end = time.time()
            print(f'Result computed in {end-start:.1f} seconds')
        except FileNotFoundError as e:
            print(f'Unable to update {args.exp_name}')
            print(str(e))
 