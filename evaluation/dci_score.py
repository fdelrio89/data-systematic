from collections import defaultdict
import numpy as np
import torch
from tqdm.auto import tqdm
import scipy
from sklearn import ensemble
from evaluation import MetricEvaluator, FeatureExtractor


class DCIEvaluator(MetricEvaluator):
    fast_mode = True
    def __init__(self, exp_name):
        super().__init__(exp_name)

        self.fe = FeatureExtractor(self.model)
        self.batch_size = 256

    def extract_features_and_properties(self):
        print('extract_features_and_properties')
        shuffle = True

        feats_by_set = {}
        gt_by_set = {}
        props_by_set = {}

        num_workers = 0
        # num_workers = max(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 2, 0)
        for test_name, loaders in self.build_dataloaders(
                self.batch_size, shuffle, num_workers=num_workers).items():

            if self.fast_mode and test_name != 'complete':
                continue

            print(f'Extracting features for: {test_name} split')
            feats_by_task = defaultdict(list)
            gt_by_task = defaultdict(list)
            props_by_task = defaultdict(list)
            for task in tqdm(self.tasks):
                if self.fast_mode:
                    if task not in ('colors', 'shapes'):
                        continue

                print(f'Extracting {task} features.')
                for images, scenes, labels in tqdm(loaders[task], leave=False):
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

    def calculate_dci_metrics(self, seeds=1):
        # all_p_scores[test_name][task].append(np.mean(p_scores))
        all_dci_metrics = {}
        feats_by_set, gt_by_set, props_by_set = self.extract_features_and_properties()

        for test_name in feats_by_set:
            all_dci_metrics[test_name] = {}
            for task_name in self.tasks:
                if self.fast_mode:
                    if task_name not in ('colors', 'shapes'):
                        continue
                feats = feats_by_set[test_name][task_name].numpy()
                props = props_by_set[test_name][task_name].numpy()

                all_dci_metrics[test_name][task_name] = {
                    'disentanglement': [],
                    'completeness': [],
                }
                for _ in range(seeds):
                    matrix, _, _, _ = self.compute_importance_gbt(
                                                feats, props.T, feats[:1,:], props[:1,:].T)

                    all_dci_metrics[test_name][task_name]['disentanglement'].append(self.disentanglement(matrix))
                    all_dci_metrics[test_name][task_name]['completeness'].append(self.completeness(matrix))

        return all_dci_metrics

    def _compute_dci(self, mus_train, ys_train, mus_test, ys_test):
        """Computes score based on both training and testing codes and factors."""
        scores = {}
        importance_matrix, train_err, test_err = self.compute_importance_gbt(
            mus_train, ys_train, mus_test, ys_test)
        assert importance_matrix.shape[0] == mus_train.shape[0]
        assert importance_matrix.shape[1] == ys_train.shape[0]
        scores["informativeness_train"] = train_err
        scores["informativeness_test"] = test_err
        scores["disentanglement"] = self.disentanglement(importance_matrix)
        scores["completeness"] = self.completeness(importance_matrix)
        return scores

    @staticmethod
    def disentanglement_per_code(importance_matrix):
        """Compute disentanglement score of each code."""
        # importance_matrix is of shape [num_codes, num_factors].
        return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                        base=importance_matrix.shape[1])

    @staticmethod
    def disentanglement(importance_matrix):
        """Compute the disentanglement score of the representation."""
        per_code = DCIEvaluator.disentanglement_per_code(importance_matrix)
        if importance_matrix.sum() == 0.:
            importance_matrix = np.ones_like(importance_matrix)
        code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

        return np.sum(per_code*code_importance)

    @staticmethod
    def compute_importance_gbt(x_train, y_train, x_test, y_test):
        """Compute importance based on gradient boosted trees."""
        models = []
        num_factors = y_train.shape[0]
        num_codes = x_train.T.shape[0]
        importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                    dtype=np.float64)

        train_loss = []
        test_loss = []
        for i in range(num_factors):
            print(f"Training for factor {i+1}")
            model = ensemble.GradientBoostingClassifier(verbose=1)
            model.fit(x_train, y_train[i, :])
            models.append(model)
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            train_loss.append(np.mean(model.predict(x_train) == y_train[i, :]))
            test_loss.append(np.mean(model.predict(x_test) == y_test[i, :]))
        return importance_matrix, np.mean(train_loss), np.mean(test_loss), models

    @staticmethod
    def completeness_per_factor(importance_matrix):
        """Compute completeness of each factor."""
        # importance_matrix is of shape [num_codes, num_factors].
        return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                        base=importance_matrix.shape[0])


    @staticmethod
    def completeness(importance_matrix):
        """"Compute completeness of the representation."""
        per_factor = DCIEvaluator.completeness_per_factor(importance_matrix)
        if importance_matrix.sum() == 0.:
            importance_matrix = np.ones_like(importance_matrix)
        factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
        return np.sum(per_factor*factor_importance)


def compute_dci_metrics(exp_name, n_samples=15_000, n_seeds_to_try=1):
    probe_evaluator = DCIEvaluator(exp_name)
    probe_evaluator.sample_subset_datasets(n_samples)
    return probe_evaluator.calculate_dci_metrics(n_seeds_to_try)
