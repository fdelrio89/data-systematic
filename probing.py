import copy
from collections import defaultdict
from itertools import product
import os
import random


from config import load_config
from data import build_datasets
from data import CollatorForMaskedSelectedTokens
from data import ALL_POSSIBLE_COLORS
from model import MultimodalModel, MultimodalPretrainingModel
from utils import load_checkpoint
from tqdm.auto import tqdm


import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset


class MetricEvaluator:
    def __init__(self, exp_name):
        self.exp_name = exp_name

        self.device = torch.device('cuda')
        self.checkpoint = load_checkpoint(exp_name, epoch=None)
        self.config = load_config(exp_name)

        self.tasks = ['sizes', 'colors', 'materials', 'shapes']

        self.build_datasets()
        self.build_models()
        self.build_auxiliary()

    def build_datasets(self):
        self.train_dataset, self.test_dataset, self.systematic_dataset, _ = build_datasets(self.config)
        self.config.pad_idx = self.train_dataset.pad_idx
        self.config.n_tokens = self.train_dataset.n_tokens

        self.processor = self.test_dataset.processor
        self.mask_token_idx = self.processor.vocabulary['[MASK]']
        self.pad_token_idx = self.processor.vocabulary['[PAD]']

        self.complete_dataset = ConcatDataset((self.test_dataset, self.systematic_dataset))

    def sample_subset_datasets(self, scenes_to_sample):
        if scenes_to_sample is not None:
            test_indices = random.sample(range(len(self.test_dataset)), k=scenes_to_sample)
            self.test_dataset = Subset(self.test_dataset, test_indices)
            systematic_indices = random.sample(range(len(self.systematic_dataset)), k=scenes_to_sample)
            self.systematic_dataset = Subset(self.systematic_dataset, systematic_indices)
            complete_indices = random.sample(range(len(self.complete_dataset)), k=scenes_to_sample)
            self.complete_dataset = Subset(self.complete_dataset, complete_indices)

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

    def build_dataloaders(self, batch_size, shuffle):
        self.collators = {
            type_: CollatorForMaskedSelectedTokens(self.config, self.processor, tokens=tokens)
            for type_, tokens in self.token_types.items()
        }
        dlkwargs = {
            'batch_size': batch_size,
            'num_workers': int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
            'pin_memory': torch.cuda.is_available(),
        }
        test_loaders = {}
        systematic_loaders = {}
        complete_loaders = {}
        for task, collator in self.collators.items():
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


class ProbeEvaluator(MetricEvaluator):
    def __init__(self, exp_name):
        super().__init__(exp_name)
        self.fe = FeatureExtractor(self.model)

        # self.sample_subset_datasets(scenes_to_sample=15_000)
        
        self.batch_size = 256
        self.probe_batch_size = 256
        self.hidden_sizes_to_try = [128]
        self.lrs_to_try = [1e-2, 1e-3]

    def extract_features_and_properties(self):
        batch_size = 256
        shuffle = True

        feats_by_set = {}
        gt_by_set = {}
        props_by_set = {}
        for test_name, loaders in self.build_dataloaders(batch_size, shuffle).items():
            feats_by_task = defaultdict(list)
            gt_by_task = defaultdict(list)
            props_by_task = defaultdict(list)
            for task in self.tasks:
                for images, scenes, labels in tqdm(loaders[task]):
                    images = images.to(self.device)
                    scenes = scenes.to(self.device)
                    labels = labels.to(self.device)
                    with torch.no_grad():
                        features = self.fe(images, scenes)

                        scene_features = features.transpose(1,0)[:,-self.config.max_scene_size:]
                        mask_idxs = (scenes == self.mask_token_idx)

                        gts = labels[:,-self.config.max_scene_size:][mask_idxs].cpu()
                        feats = scene_features[mask_idxs].cpu()
                        props = self.get_properties(scenes)

                        gt_by_task[task].append(gts)
                        feats_by_task[task].append(feats)
                        props_by_task[task].append(props)

            feats_by_set[test_name] = feats_by_task
            gt_by_set[test_name] = gt_by_task
            props_by_set[test_name] = props_by_task

        for test_name, feats_by_tasks in feats_by_set.items():
            for task_name, feats in feats_by_tasks.items():
                task_idx = self.tasks.index(task_name)
                feats_by_set[test_name][task_name] = torch.cat(feats)
                gt_by_set[test_name][task_name] = torch.cat(gt_by_set[test_name][task_name])
                props_by_set[test_name][task_name] = torch.stack(
                    [torch.cat(prop) for prop in zip(*props_by_set[test_name][task_name])], dim=-1)

                props_by_set[test_name][task_name][:,task_idx] = gt_by_set[test_name][task_name]

        return feats_by_set, gt_by_set, props_by_set

    def create_probe(self, n_features, hidden_size, n_targets):
        if isinstance(hidden_size, list):
            hidden_sizes = hidden_size
        else:
            hidden_sizes = [hidden_size]

        prev_hidden_size = hidden_sizes.pop(0)
        layers = [nn.Linear(n_features, prev_hidden_size)]
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_hidden_size, hidden_size))
            prev_hidden_size = hidden_size
        layers.append(nn.Linear(prev_hidden_size, n_targets))
        linear_probe = nn.Sequential(*layers)

        for p in linear_probe:
            try:
                nn.init.kaiming_uniform_(p.weight)
            except AttributeError:
                pass

        linear_probe = linear_probe.to(self.device)
        return linear_probe

    def train_probe(self,
                    linear_probe,
                    optimizer,
                    train_loader,
                    val_loader,
                    num_epochs,
                    patience):

        N = len(val_loader.dataset)

        wait = 0
        best_model = None
        best_val_relevant_metric = 0
        criterion = F.cross_entropy

        linear_probe.train()

        metrics = defaultdict(list)
        for epoch in tqdm(range(num_epochs)):
            linear_probe.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                logit_pred = linear_probe(x).squeeze()
                loss = criterion(logit_pred, y)
                loss.backward()
                optimizer.step()

                metrics['loss'].append(float(loss))
                pred = logit_pred.argmax(-1)
                train_acc = (pred == y).sum() / y.shape[0]
                metrics['train_acc'].append(float(train_acc))


            linear_probe.eval()
            cum_sum = 0
            cum_loss = 0
            cum_logit_pred, cum_y = [], []
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                bsz = x.shape[0]

                logit_pred = linear_probe(x).squeeze()
                loss = criterion(logit_pred, y)
                cum_loss += float(loss) * bsz

                cum_logit_pred.append(logit_pred)
                cum_y.append(y)

            logit_pred = torch.cat(cum_logit_pred)
            y = torch.cat(cum_y)

            val_loss = cum_loss / N
            metrics['val_loss'].append(float(val_loss))
            pred = logit_pred.argmax(-1)
            val_acc = (pred == y).sum() / y.shape[0]
            metrics['val_acc'].append(float(val_acc))

            # Early stopping
            val_relevant_metric = metrics['val_acc'][-1]
            if val_relevant_metric > best_val_relevant_metric:
                best_val_relevant_metric = val_relevant_metric
                best_model = copy.deepcopy(linear_probe)
                wait = 0
            else:
                wait += 1

            if wait > patience:
                break

        print(f'Finished in epoch {epoch}')
        return best_model, metrics, best_val_relevant_metric

    def calculate_probe_metrics(self, seeds=10, n_objects_to_sample=15_000):
        all_probe_metrics = {}
        feats_by_set, gt_by_set, props_by_set = self.extract_features_and_properties()

        for test_name in feats_by_set:
            all_probe_metrics[test_name] = {}
            for feat_task_name, gt_task_name in product(self.tasks, repeat=2):
                probe_metrics = []

                feats = feats_by_set[test_name][feat_task_name]
                props = props_by_set[test_name][feat_task_name]
                print(f'Running Probe for {feat_task_name}:{gt_task_name}')
                for _ in range(seeds):
                    object_indices = random.sample(range(len(feats)), k=n_objects_to_sample)
                    X = feats[object_indices]
                    gt_task_idx = self.tasks.index(gt_task_name)
                    y = props[object_indices][:,gt_task_idx]

                    tokens_to_class_map = {t: i for i, t in enumerate(set(y.tolist()))}
                    y = y.apply_(tokens_to_class_map.get) # transform to classes

                    n_features = X.size(1)
                    n_targets = len(tokens_to_class_map)

                    (train_loader,
                     val_loader,
                     test_loader) = self.build_probe_loaders(X, y, batch_size=self.probe_batch_size)
                    print('Grid Search')
                    best_model,*_ = self.grid_search_best_probe(
                        train_loader, val_loader, n_features, n_targets)
                    print('Eval Best Model')
                    run_metrics = self.eval_probe(best_model, test_loader)
                    probe_metrics.append(run_metrics['test_acc'])

                all_probe_metrics[test_name][f'{feat_task_name}:{gt_task_name}'] = probe_metrics

        return all_probe_metrics

    def build_probe_loaders(self, X, y, batch_size, n_train=10_000):
        n = X.size(0)
        n_val = (n - n_train) // 2

        X_train = X[:n_train,:].to(self.device)
        y_train = y[:n_train].to(self.device)
        X_val = X[n_train:n_train+n_val,:].to(self.device)
        y_val = y[n_train:n_train+n_val].to(self.device)
        X_test = X[n_train+n_val:,:].to(self.device)
        y_test = y[n_train+n_val:].to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def grid_search_best_probe(self, train_loader, val_loader, n_features, n_targets):
        relevant_metric_name = 'acc'

        all_metrics = {}
        grid_best_relevant_metric = 0
        grid_best_model = None
        grid_best_metrics = None
        best_hidden_size = None
        best_lr = None
        for exp_idx, (hidden_size, lr) in enumerate(product(self.hidden_sizes_to_try,
                                                            self.lrs_to_try), start=1):

            print(f'Start experiment: {exp_idx}')

            linear_probe = self.create_probe(n_features, [hidden_size, hidden_size], n_targets)
            optimizer = SGD(linear_probe.parameters(), lr=lr, momentum=0.9)

            best_model, metrics, best_relevant_metric = self.train_probe(linear_probe,
                                                                         optimizer,
                                                                         train_loader,
                                                                         val_loader,
                                                                         num_epochs=500,
                                                                         patience=20,
                                                                         )

            all_metrics[(hidden_size, lr)] = metrics
            if  best_relevant_metric > grid_best_relevant_metric:
                best_hidden_size = hidden_size
                best_lr = lr
                grid_best_model = best_model
                grid_best_metrics = metrics
                grid_best_relevant_metric = best_relevant_metric

            linear_probe = optimizer = None

            print(f'hidden_size={hidden_size} lr={lr}')
            print(f'Best val {relevant_metric_name}: ', float(best_relevant_metric))
            print('\n')

        print('best_hidden_size:', best_hidden_size)
        print('best_lr:', best_lr)
        grid_best_params = {
            'hidden_size': best_hidden_size,
            'lr': best_lr,
        }

        return grid_best_model, grid_best_metrics, grid_best_params

    def eval_probe(self, model, test_loader, metrics=None):
        criterion = F.cross_entropy

        if metrics is None:
            metrics = {}

        model.eval()
        cum_loss = 0
        cum_logit_pred = []
        cum_y = []
        for x, y in tqdm(test_loader):
            x, y = x.to(self.device), y.to(self.device)
            bsz = x.shape[0]
            logit_pred = model(x).squeeze()
            cum_logit_pred.append(logit_pred)
            cum_y.append(y)
            loss = criterion(logit_pred, y)
            cum_loss += float(loss) * bsz


        logit_pred = torch.cat(cum_logit_pred)
        y = torch.cat(cum_y)
        metrics['test_loss'] = (cum_loss / len(test_loader.dataset))
        pred = logit_pred.argmax(-1)
        test_acc = (pred == y).sum() / y.shape[0]
        metrics['test_acc'] = float(test_acc)

        return metrics

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


def compute_probe_metrics(exp_name, n_samples=15_000, n_seeds_to_try=10, n_objects_to_sample=15_000):
    probe_evaluator = ProbeEvaluator(exp_name)
    probe_evaluator.sample_subset_datasets(n_samples)
    return probe_evaluator.calculate_probe_metrics(n_seeds_to_try, n_objects_to_sample)
