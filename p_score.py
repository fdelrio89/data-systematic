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


def all_possible_dichotomies(unique_tokens):
    dichotomies = set()
    for dichotomic_tokens_idx in range(len(unique_tokens)):
        context_tokens = unique_tokens[:dichotomic_tokens_idx] + unique_tokens[dichotomic_tokens_idx+1:]
        dichotomic_tokens = unique_tokens[dichotomic_tokens_idx]
        for dichotomic_pair in combinations(dichotomic_tokens, r=2):
            for context0 in product(*context_tokens):
                for context1 in product(*context_tokens):
                    if context0 == context1:
                        continue

                    a, b = dichotomic_pair
                    if a == b:
                        continue
                    from_v0 = list(context0).copy()
                    from_v0.insert(dichotomic_tokens_idx, a)
                    from_v1 = list(context1).copy()
                    from_v1.insert(dichotomic_tokens_idx, a)
                    to_v0 = list(context0).copy()
                    to_v0.insert(dichotomic_tokens_idx, b)
                    to_v1 = list(context1).copy()
                    to_v1.insert(dichotomic_tokens_idx, b)

                    dichotomies.add((tuple(from_v0),
                                        tuple(from_v1),
                                        tuple(to_v0),
                                        tuple(to_v1)))

    return list(dichotomies)

def is_possible(from_v0, from_v1, to_v0, to_v1, prop_index):
    if not prop_index[from_v0]:
        return False
    if not prop_index[from_v1]:
        return False
    if not prop_index[to_v0]:
        return False
    if not prop_index[to_v1]:
        return False
    return True


def compute_p_scores(exp_name, n_samples=1024, n_seeds_to_try=10, n_sampled_vertices=20, n_sampled_dichotomies=3500):
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
    
    all_p_scores = {}
    tasks = ['sizes', 'colors', 'materials', 'shapes']
    for test_name, loader in [
            ('test', test_loaders), ('systematic', systematic_loaders), ('complete', complete_loaders)]:
        all_p_scores[test_name] = {}
        for task_idx, task in enumerate(tasks):
            all_p_scores[test_name][task] = []
            dichotomies = None
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
                
                if not dichotomies:
                    dichotomies = all_possible_dichotomies(unique_tokens)
                possible_dichotomies = [p for p in dichotomies if is_possible(*p, prop_index=prop_index)]
                
                if n_sampled_dichotomies < len(possible_dichotomies):
                    sampled_dichotomies = random.sample(possible_dichotomies, k=n_sampled_dichotomies)
                else:
                    sampled_dichotomies = possible_dichotomies

                p_scores = []
                for from_v0, from_v1, to_v0, to_v1 in tqdm(sampled_dichotomies):
                    for _ in range(n_sampled_vertices):
                        from_v0_idx = random.choice(prop_index[from_v0])
                        from_v1_idx = random.choice(prop_index[from_v1])
                        to_v0_idx = random.choice(prop_index[to_v0])
                        to_v1_idx = random.choice(prop_index[to_v1])

                        vec0 = feats[to_v0_idx] - feats[from_v0_idx]
                        vec1 = feats[to_v1_idx] - feats[from_v1_idx]

                        p_score = vec0 @ vec1 / (vec0.norm() * vec1.norm())
                        p_scores.append(float(p_score))
                
                all_p_scores[test_name][task].append(np.mean(p_scores))

    return all_p_scores


if __name__ == '__main__':
    n_colors = 8
    exp_name = f'mmlm--n_colors={n_colors}c--mlm_probability=0.15'
    all_p_scores = compute_p_scores(exp_name)
    with open(f'outputs/results/{exp_name}_p_scores.json', 'w') as fp:
        json.dump(all_p_scores, fp)