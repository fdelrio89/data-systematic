import os
# os.chdir('/mnt/ialabnas/homes/fidelrio/systematic-text-representations/')

from collections import defaultdict
import json
import random
import pprint
from itertools import product, combinations
import random

import torch

from config import load_config
from data import build_datasets
from data import CollatorForMaskedSelectedTokens
from data import ALL_POSSIBLE_COLORS
from model import MultimodalModel, MultimodalPretrainingModel
from utils import load_checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset

import numpy as np
from sklearn.decomposition import PCA

pp = pprint.PrettyPrinter(indent=2)


def all_possible_triplets(unique_tokens, only_task_mod_ids=None):
    only_task_mod_idx1 = only_task_mod_idx2 = None
    if only_task_mod_ids is not None:
        only_task_mod_idx1, only_task_mod_idx2 = only_task_mod_ids
        
    task_ids = list(range(len(unique_tokens)))
    
    triplets = set()
    context_tokens = unique_tokens.copy()
    for context in product(*context_tokens):
        for task_mod_idx1, task_mod_idx2 in combinations(task_ids, r=2):
            if task_mod_idx1 == task_mod_idx2:
                    continue

            if only_task_mod_ids is not None:
                if only_task_mod_idx1 is not None and task_mod_idx1 != only_task_mod_idx1: 
                    continue
                if only_task_mod_idx2 is not None and task_mod_idx2 != only_task_mod_idx2:
                    continue
                    
            tokens_mod1 = unique_tokens[task_mod_idx1]
            tokens_mod2 = unique_tokens[task_mod_idx2]
            for token_mod_idx1, token_mod_idx2 in product(tokens_mod1, tokens_mod2):
                if context[task_mod_idx1] == token_mod_idx1:
                    continue
                if context[task_mod_idx2] == token_mod_idx2:
                    continue
                
                pivot = list(context).copy()
                v1 = list(context).copy()
                v1[task_mod_idx1] = token_mod_idx1
                v2 = list(context).copy()
                v2[task_mod_idx2] = token_mod_idx2
                triplets.add((tuple(pivot),
                              tuple(v1),
                              tuple(v2)))
    return list(triplets)


def is_possible(pivot, v0, v1, prop_index):
    if not prop_index[pivot]:
        return False
    if not prop_index[v0]:
        return False
    if not prop_index[v1]:
        return False
    return True


def compute_o_scores(exp_name, n_samples=1024, n_seeds_to_try=10, n_sampled_vertices=20,
                     n_sampled_triplets=3500, triplet_within_task=False, triplet_within_color_shape=False):
    device = torch.device('cuda')

    checkpoint = load_checkpoint(exp_name, epoch=None)
    print('Checkpoint loaded from epoch:', checkpoint['epoch'])
    
    config = load_config(exp_name)

    # config.vocabulary_path = config.vocabulary_path.replace('/workspace/' ,'/workspace1/')
    # config.base_path = config.base_path.replace('/workspace/' ,'/workspace1/')

    train_dataset, test_dataset, systematic_dataset, common_systematic_dataset = build_datasets(config)
    config.pad_idx = train_dataset.pad_idx
    config.n_tokens = train_dataset.n_tokens

    complete_dataset = ConcatDataset((test_dataset, systematic_dataset))

    processor = test_dataset.processor
    mask_token_idx = processor.vocabulary['[MASK]']
    pad_token_idx = processor.vocabulary['[PAD]']
    
    def get_props(scenes):
        sizes = scenes[:,1:][:,0::5]
        colors = scenes[:,1:][:,1::5]
        materials = scenes[:,1:][:,2::5]
        shapes = scenes[:,1:][:,3::5]

        sizes = sizes[sizes != pad_token_idx].cpu()
        colors = colors[colors != pad_token_idx].cpu()
        materials = materials[materials != pad_token_idx].cpu()
        shapes = shapes[shapes != pad_token_idx].cpu()
        return sizes, colors, materials, shapes

    model = MultimodalModel(config).to(device)
    training_model = MultimodalPretrainingModel(model, config).to(device)
    training_model.load_state_dict(checkpoint['state_dict'])
    
    vocab = processor.vocabulary

    colors_tokens = sorted(
        [vocab[w] for w in ALL_POSSIBLE_COLORS if w in vocab])
    shapes_tokens = sorted(
        [vocab[w] for w in ['cylinder', 'sphere', 'cube'] if w in vocab])
    materials_tokens = sorted(
        [vocab[w] for w in ['metal', 'rubber'] if w in vocab])
    sizes_tokens = sorted(
        [vocab[w] for w in ['small', 'large'] if w in vocab])

    batch_size = n_samples
    test_indices = random.sample(range(len(test_dataset)), k=batch_size)
    pc_subset_test = Subset(test_dataset, test_indices)
    systematic_indices = random.sample(range(len(systematic_dataset)), k=batch_size)
    pc_subset_systematic = Subset(systematic_dataset, systematic_indices)
    complete_indices = random.sample(range(len(complete_dataset)), k=batch_size)
    pc_subset_complete = Subset(complete_dataset, complete_indices)

    colors_collator = CollatorForMaskedSelectedTokens(config, processor, tokens=colors_tokens)
    shapes_collator = CollatorForMaskedSelectedTokens(config, processor, tokens=shapes_tokens)
    materials_collator = CollatorForMaskedSelectedTokens(config, processor, tokens=materials_tokens)
    sizes_collator = CollatorForMaskedSelectedTokens(config, processor, tokens=sizes_tokens)
    dlkwargs = {
        'batch_size': batch_size,
        'num_workers': int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
        'pin_memory': torch.cuda.is_available(),
    }

    test_loaders = {}
    systematic_loaders = {}
    complete_loaders = {}
    for task, collator in [('colors', colors_collator),
                        ('shapes', shapes_collator),
                        ('materials', materials_collator),
                        ('sizes', sizes_collator)]:
        
        test_loaders[task] = DataLoader(
            pc_subset_test, shuffle=True, collate_fn=collator, **dlkwargs)
        systematic_loaders[task] = DataLoader(
            pc_subset_systematic, shuffle=True, collate_fn=collator, **dlkwargs)
        complete_loaders[task] = DataLoader(
            pc_subset_complete, shuffle=True, collate_fn=collator, **dlkwargs)
        
    
    feature_maps = []  # This will be a list of Tensors, each representing a feature map

    def hook_feat_map(mod, inp, out):
        feature_maps.clear()
        feature_maps.append(out)

    model.transformer.register_forward_hook(hook_feat_map)
    
    all_o_scores = {}
    tasks = ['sizes', 'colors', 'materials', 'shapes']
    for test_name, loader in [
            ('test', test_loaders), ('systematic', systematic_loaders), ('complete', complete_loaders)]:
        all_o_scores[test_name] = {}
        for task_idx, task in enumerate(tasks):
            all_o_scores[test_name][task] = []
            triplets = None
            for _ in range(n_seeds_to_try):
                images, scenes, labels = next(iter(loader[task]))
                images, scenes, labels = images.to(device), scenes.to(device), labels.to(device)
                with torch.no_grad():
                    output_logits = model(images, scenes)

                    features = feature_maps[0]
                    scene_features = features.transpose(1,0)[:,-config.max_scene_size:]

                    mask_idxs = (scenes == mask_token_idx)
                    gt = labels[:,-config.max_scene_size:][mask_idxs].cpu()
                    feats = scene_features[mask_idxs].cpu()
                    props = get_props(scenes)
                    sizes, colors, materials, shapes = props

                prop_stack = [sizes, colors, materials, shapes]
                prop_stack[task_idx] = gt
                prop_stack = torch.stack(prop_stack, dim=-1).tolist()
                prop_index = defaultdict(list)
                for idx, props in enumerate(prop_stack):
                    props = tuple(props)
                    prop_index[props].append(idx)

                unique_sizes = sizes.unique().tolist()
                unique_colors = colors.unique().tolist()
                unique_materials = materials.unique().tolist()
                unique_shapes = shapes.unique().tolist()
                unique_tokens = [
                    unique_sizes,
                    unique_colors,
                    unique_materials,
                    unique_shapes,
                ]

                unique_gts = gt.unique().tolist()
                unique_tokens[task_idx] = unique_gts
                
                
                only_task_mod_ids = None
                if triplet_within_task: # task against everything else
                    only_task_mod_ids = (task_idx, None)
                elif triplet_within_color_shape:
                    only_task_mod_ids = (tasks.index('colors'), tasks.index('shapes'))
                
                if not triplets:
                    triplets = all_possible_triplets(unique_tokens,
                                                     only_task_mod_ids=only_task_mod_ids)
                possible_triplets = [p for p in triplets if is_possible(*p, prop_index=prop_index)]
                
                if n_sampled_triplets < len(possible_triplets):
                    sampled_triplets = random.sample(possible_triplets, k=n_sampled_triplets)
                else:
                    sampled_triplets = possible_triplets

                o_scores = []
                for pivot, v0, v1 in tqdm(sampled_triplets):
                    for _ in range(n_sampled_vertices):
                        pivot_idx = random.choice(prop_index[pivot])
                        v0_idx = random.choice(prop_index[v0])
                        v1_idx = random.choice(prop_index[v1])

                        vec0 = feats[v0_idx] - feats[pivot_idx]
                        vec1 = feats[v1_idx] - feats[pivot_idx]

                        o_score = vec0 @ vec1 / (vec0.norm() * vec1.norm())
                        o_scores.append(np.sqrt(1 - float(o_score)**2))
                
                all_o_scores[test_name][task].append(np.mean(o_scores))

    return all_o_scores


if __name__ == '__main__':
    n_colors = 8
    exp_name = f'mmlm--n_colors={n_colors}c--mlm_probability=0.15'
    all_o_scores = compute_o_scores(exp_name)
    with open(f'outputs/results/{exp_name}_o_scores.json', 'w') as fp:
        json.dump(all_o_scores, fp)