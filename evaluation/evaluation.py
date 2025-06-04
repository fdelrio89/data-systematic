import random
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from config import load_config
from data import build_datasets, ALL_POSSIBLE_COLORS
from data import CollatorForMaskedSelectedTokens
from model import MultimodalModel, MultimodalPretrainingModel
from utils import load_checkpoint

class MetricEvaluator:
    def __init__(self, exp_name):
        self.exp_name = exp_name

        self.device = torch.device('cuda')
        self.checkpoint = load_checkpoint(exp_name, epoch=None)
        self.config = load_config(exp_name)
        print(self.config.base_path)
        print(self.config.vocabulary_path)

        self.tasks = ['sizes', 'colors', 'materials', 'shapes']

        self.build_datasets()
        self.build_models()
        self.build_auxiliary()

    def build_datasets(self):
        self.train_dataset, self.test_dataset, self.systematic_dataset, _ = build_datasets(self.config)
        self.original_train_dataset = self.train_dataset
        self.original_test_dataset = self.test_dataset
        self.original_systematic_dataset = self.systematic_dataset

        self.config.pad_idx = self.train_dataset.pad_idx
        self.config.n_tokens = self.train_dataset.n_tokens

        self.processor = self.test_dataset.processor
        self.mask_token_idx = self.processor.vocabulary['[MASK]']
        self.pad_token_idx = self.processor.vocabulary['[PAD]']

        self.complete_dataset = ConcatDataset((self.test_dataset, self.systematic_dataset))
        self.original_complete_dataset = self.complete_dataset

    def random_dataset_subset(self, dataset, k):
        indices = random.sample(range(len(dataset)), k=k)
        return Subset(dataset, indices)

    def sample_subset_datasets(self, scenes_to_sample):
        if scenes_to_sample is not None:
            self.test_dataset = self.random_dataset_subset(self.original_test_dataset, k=scenes_to_sample)
            self.systematic_dataset = self.random_dataset_subset(self.original_systematic_dataset, k=scenes_to_sample)
            self.complete_dataset = self.random_dataset_subset(self.original_complete_dataset, k=scenes_to_sample)

    def build_models(self):
        self.model = MultimodalModel(self.config).to(self.device)
        self.training_model = MultimodalPretrainingModel(self.model, self.config).to(self.device)
        self.training_model.load_state_dict(self.checkpoint['state_dict'])

    def build_auxiliary(self):

        self.vocab = self.processor.vocabulary
        self.token_types = {
            # 'relations': sorted([vocab[w] for w in ['left', 'right', 'behind', 'front'] if w in vocab]),
            'sizes': sorted([self.vocab[w] for w in ['small', 'large'] if w in self.vocab]),
            'colors': sorted([self.vocab[w] for w in ALL_POSSIBLE_COLORS if w in self.vocab]),
            'materials': sorted([self.vocab[w] for w in ['metal', 'rubber'] if w in self.vocab]),
            'shapes': sorted([self.vocab[w] for w in ['cylinder', 'sphere', 'cube'] if w in self.vocab]),
        }
        self.random_baseline = {
            'identity':  1 / len(self.vocab),
            **{type_: len(tokens) for type_, tokens in self.token_types.items()}
        }

    def build_dataloaders(self, batch_size, shuffle, num_workers):
        self.collators = {
            type_: CollatorForMaskedSelectedTokens(self.config, self.processor, tokens=tokens)
            for type_, tokens in self.token_types.items()
        }

        dlkwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
        }
        print(f'Building DataLoader with num_workers: {dlkwargs["num_workers"]}')
        test_loaders = {}
        systematic_loaders = {}
        complete_loaders = {}
        for task in self.tasks:
            collator = self.collators[task]
            test_loaders[task] = DataLoader(
                self.test_dataset, shuffle=shuffle, collate_fn=collator, **dlkwargs)
            systematic_loaders[task] = DataLoader(
                self.systematic_dataset, shuffle=shuffle, collate_fn=collator, **dlkwargs)
            complete_loaders[task] = DataLoader(
                self.complete_dataset, shuffle=shuffle, collate_fn=collator, **dlkwargs)

        return {'test': test_loaders, 'systematic': systematic_loaders, 'complete': complete_loaders}

    def get_properties(self, scenes, indices=None):
        sizes = scenes[:,1:][:,0::5]
        colors = scenes[:,1:][:,1::5]
        materials = scenes[:,1:][:,2::5]
        shapes = scenes[:,1:][:,3::5]

        sizes = sizes[sizes != self.pad_token_idx]
        colors = colors[colors != self.pad_token_idx]
        materials = materials[materials != self.pad_token_idx]
        shapes = shapes[shapes != self.pad_token_idx]
        if indices is not None:
            sizes = sizes[indices]
            colors = colors[indices]
            materials = materials[indices]
            shapes = shapes[indices]

        return sizes.cpu(), colors.cpu(), materials.cpu(), shapes.cpu()


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.feature_location = []

        def hook_feat_map(mod, inp, out):
            self.feature_location.clear()
            self.feature_location.append(out)

        self.model.transformer.register_forward_hook(hook_feat_map)

    def __call__(self, *args, **kwargs):
        _ = self.model(*args, **kwargs)
        features = self.feature_location[0]
        return features
