from utils import only_in_amd_cluster
from collections import defaultdict
import math
import os
import json
import h5py
import random
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import default_collate
from torch.utils.data import DataLoader
from torch.utils.data import Subset, ConcatDataset, Sampler
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ColorJitter
import lightning as L
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)


def build_datasets(config):
    if config.multimodal_pretraining:
        if config.use_embedding_loaded or config.use_vit_embedding_loaded:
            print('Building CLEVRMultimodalFromFeaturesSplit')
            return CLEVRMultimodalFromFeaturesSplit.build_splits(config)
        else:
            print('Building CLEVRMultimodalSplit')
            return CLEVRMultimodalSplit.build_splits(config)
    elif config.multimodal_training:
        print('Building CLEVRMultimodalTrainingSplit')
        return CLEVRMultimodalTrainingSplit.build_splits(config)
    elif config.image_pretraining:
        print('Building CLEVRMultimodalSplit')
        return CLEVRMultimodalSplit.build_splits(config)
    elif config.use_txt_scene:
        print('Building CLEVRTextSplit')
        return CLEVRTextSplit.build_splits(config)
    else:
        print('Building CLEVRSplit')
        return CLEVRSplit.build_splits(config)


def build_loader(dataset, config, shuffle=True, collate_fn=None, episodic_training=False):
    in_amd_cluster = lambda: os.environ.get('IS_AMD_CLUSTER')
    cpus_for_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    num_workers = cpus_for_task // 4 if in_amd_cluster else cpus_for_task
    dlkwargs = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if collate_fn:
        dlkwargs['collate_fn'] = collate_fn
    elif config.multimodal_pretraining:
        dlkwargs['collate_fn'] = CollatorForMaskedLanguageModeling(
            config, dataset.processor, mlm_probability=config.mlm_probability)

    if episodic_training:
        if isinstance(dataset, CLEVRMultimodalSplit):
            scenes_or_questions = dataset.scenes
        elif isinstance(dataset, CLEVRSplit):
            scenes_or_questions = dataset.questions

        dlkwargs['batch_sampler'] = EpisodicBatchSampler(scenes_or_questions, config.batch_size, shuffle=shuffle)
    else:
        dlkwargs['batch_size'] = config.batch_size
        dlkwargs['shuffle'] = shuffle

    return DataLoader(dataset, **dlkwargs)


def build_detailed_test_dataloaders(dataset, config, type_of_tokens_to_test=None):
    processor = dataset.processor
    vocab = processor.vocabulary
    tokens_to_test = {
        'relation': ['left', 'right', 'behind', 'front'],
        # 'color': ['blue', 'brown', 'cyan', 'green', 'red', 'purple', 'yellow', 'gray'],
        'color': ALL_POSSIBLE_COLORS,
        'shapes': ['cylinder', 'sphere', 'cube'],
        'materials': ['metal', 'rubber'],
        'size': ['small', 'large'],
    }
    tokens_idx_to_test = {
        k: sorted([vocab[w] for w in tokens if w in vocab]) for k, tokens in tokens_to_test.items()
        if type_of_tokens_to_test is None or k in type_of_tokens_to_test
        }

    collators_for_testing = {
        k: CollatorForMaskedSelectedTokens(config, processor, tokens=tokens)
        for k, tokens in tokens_idx_to_test.items()
        }
    collators_for_testing['identity'] = IdentityCollator(config, processor)

    return {k: build_loader(dataset, config, shuffle=False, collate_fn=collate_fn)
            for k, collate_fn in collators_for_testing.items()}


class ResponsiveSubset(Subset):
    def __getattr__(self, attr):
        return getattr(self.dataset, attr)


class ResponsiveConcatDataset(ConcatDataset):
    def __getattr__(self, attr):
        return getattr(self.datasets[0], attr)


class RandomPixelShuffle(object):
    def __call__(self, img):
        channels, height, width = img.size()
        indices = np.random.permutation(height * width)
        shuffled_img = img.view(channels, -1)[:, indices].view(channels, height, width)
        return shuffled_img


class EpisodicBatchSampler(Sampler):
    def __init__(self, scenes_or_questions, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ds_len = len(scenes_or_questions)
        self.samples_by_episode_id = defaultdict(list)
        for index, s_or_q in enumerate(scenes_or_questions):
            self.samples_by_episode_id[s_or_q['episode_id']].append(index)
        
    def __iter__(self):
        iter_samples_by_episode_id = {k: v.copy() for k, v in self.samples_by_episode_id.items()}
        episode_ids = list(iter_samples_by_episode_id.keys())
        
        next_batch = []
        while len(episode_ids) > 0:
            index_to_sample = random.randrange(len(episode_ids)) if self.shuffle else 0
            episode_id = episode_ids[index_to_sample]

            sample_indices = iter_samples_by_episode_id[episode_id]
            if self.shuffle:
                random.shuffle(sample_indices)
            
            num_missing_samples = self.batch_size - len(next_batch)
            next_batch.extend(
                [sample_indices.pop(0) for _ in range(num_missing_samples) if len(sample_indices) > 0])
            
            if not sample_indices:
                episode_ids.pop(index_to_sample)
            
            if len(next_batch) >= self.batch_size:
                yield next_batch
                next_batch = []
                
    
    def __len__(self):
        return math.ceil(self.ds_len / self.batch_size)


class CollatorForMaskedLanguageModeling:
    def __init__(self, config, processor, mlm_probability=0.15):
        self.config = config
        self.mlm_probability = mlm_probability
        self.special_token_idxs = torch.tensor(processor.special_token_idxs).long()
        self.non_special_token_idxs = torch.tensor(processor.non_special_token_idxs).long()
        self.mask_token_idx = processor.mask_token_idx
        self.image_patch_sizes = config.patch_height, config.patch_width

    def __call__(self, batch):
        images, scenes = default_collate(batch)
        scenes, scenes_labels = self.build_mlm_targets(scenes)
        images_labels = self.build_null_image_targets(images)
        labels = torch.cat((images_labels, scenes_labels), dim=1)
        return images, scenes, labels

    def build_null_image_targets(self, images):
        b, *_ = images.shape
        # n_patches = int(h / self.image_patch_sizes[0]) * int(w / self.image_patch_sizes[1])
        n_patches = self.config.n_patches
        return torch.full((b, n_patches), -100)

    def random_tokens_like(self, input_, tokens):
        p = torch.ones_like(tokens) / len(tokens)
        idx = p.multinomial(num_samples=input_.numel(), replacement=True)
        random_tokens = tokens[idx].reshape(input_.shape)
        return random_tokens

    def build_mlm_targets(self, inputs):
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = torch.isin(labels, self.special_token_idxs)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = self.random_tokens_like(labels, tokens=self.non_special_token_idxs)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class CollatorForMaskedVQA:
    def __init__(self, config, processor):
        self.config = config
        self.special_token_idxs = torch.tensor(processor.special_token_idxs).long()
        self.non_special_token_idxs = torch.tensor(processor.non_special_token_idxs).long()
        self.mask_token_idx = processor.mask_token_idx
        self.image_patch_sizes = config.patch_height, config.patch_width
        self.pad_token_idx = processor.pad_token_idx

    def __call__(self, batch):
        images, question_answers = default_collate(batch)
        scenes, answer_labels = self.build_answer_targets(question_answers)
        images_labels = self.build_null_image_targets(images)
        labels = torch.cat((images_labels, answer_labels), dim=1)
        return images, scenes, labels

    def build_null_image_targets(self, images):
        b, *_ = images.shape
        # n_patches = int(h / self.image_patch_sizes[0]) * int(w / self.image_patch_sizes[1])
        n_patches = self.config.n_patches
        return torch.full((b, n_patches), -100)

    def build_answer_targets(self, inputs):
        labels = inputs.clone()
        # labels[:,:-1] = -100  # We only compute loss on answer token
        last_non_padded_indices = (inputs != self.pad_token_idx).sum(dim=-1) - 1
        answer_indices = (torch.arange(inputs.size(0)), last_non_padded_indices)
        labels = torch.full_like(inputs, -100)
        labels[answer_indices] = inputs[answer_indices]
        inputs[answer_indices] = self.mask_token_idx
        return inputs, labels


class CollatorForMaskedSelectedTokens:
    def __init__(self, config, processor, tokens, dont_mask_spheres=False):
        self.config = config
        self.token_to_mask_idxs = torch.tensor(tokens).long()
        self.special_token_idxs = torch.tensor(processor.special_token_idxs).long()
        self.mask_token_idx = processor.mask_token_idx
        self.image_patch_sizes = config.patch_height, config.patch_width
        self.dont_mask_spheres = dont_mask_spheres
        self.sphere_token_idx = processor.vocabulary['sphere']

    def __call__(self, batch):
        images, scenes = default_collate(batch)
        scenes, scenes_labels = self.build_targets(scenes)
        images_labels = self.build_null_image_targets(images)
        labels = torch.cat((images_labels, scenes_labels), dim=1)
        return images, scenes, labels

    def build_null_image_targets(self, images):
        b, *_ = images.shape
        n_patches = self.config.n_patches
        return torch.full((b, n_patches), -100)

    def build_targets(self, inputs):
        labels = inputs.clone()
        masked_indices = torch.isin(labels, self.token_to_mask_idxs)
        if self.dont_mask_spheres:
            sphere_token_mask = inputs == self.sphere_token_idx
            sphere_mask = (torch.roll(sphere_token_mask, shifts=-3, dims=1) |
                           torch.roll(sphere_token_mask, shifts=-2, dims=1) |
                           torch.roll(sphere_token_mask, shifts=-1, dims=1) |
                           sphere_token_mask)
            masked_indices = masked_indices & ~sphere_mask
            
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.mask_token_idx

        return inputs, labels


class CollatorForMaskedRandomSelectedTokens:
    def __init__(self, config, processor, tokens, p, dont_mask_spheres=False):
        self.config = config
        self.token_to_mask_idxs = torch.tensor(tokens).long()
        self.special_token_idxs = torch.tensor(processor.special_token_idxs).long()
        self.mask_token_idx = processor.mask_token_idx
        self.image_patch_sizes = config.patch_height, config.patch_width
        self.p = p
        self.dont_mask_spheres = dont_mask_spheres
        self.sphere_token_idx = processor.vocabulary['sphere']

    def __call__(self, batch):
        images, scenes = default_collate(batch)
        scenes, scenes_labels = self.build_targets(scenes)
        images_labels = self.build_null_image_targets(images)
        labels = torch.cat((images_labels, scenes_labels), dim=1)
        return images, scenes, labels

    def build_null_image_targets(self, images):
        b, *_ = images.shape
        n_patches = self.config.n_patches
        return torch.full((b, n_patches), -100)

    def build_targets(self, inputs):
        labels = inputs.clone()
        masked_indices = torch.isin(labels, self.token_to_mask_idxs)
        is_selected = torch.bernoulli(torch.full_like(labels, self.p, dtype=torch.float)).bool()
        masked_indices = masked_indices & is_selected
        if self.dont_mask_spheres:
            sphere_token_mask = inputs == self.sphere_token_idx
            sphere_mask = (torch.roll(sphere_token_mask, shifts=-3, dims=1) |
                           torch.roll(sphere_token_mask, shifts=-2, dims=1) |
                           torch.roll(sphere_token_mask, shifts=-1, dims=1) |
                           sphere_token_mask)
            masked_indices = masked_indices & ~sphere_mask
            
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.mask_token_idx

        return inputs, labels


class IdentityCollator:
    def __init__(self, config, processor):
        self.config = config
        self.special_token_idxs = torch.tensor(processor.special_token_idxs).long()
        self.image_patch_sizes = config.patch_height, config.patch_width

    def __call__(self, batch):
        images, scenes = default_collate(batch)
        scenes, scenes_labels = self.build_targets(scenes)
        images_labels = self.build_null_image_targets(images)
        labels = torch.cat((images_labels, scenes_labels), dim=1)
        return images, scenes, labels

    def build_null_image_targets(self, images):
        b, *_ = images.shape
        n_patches = self.config.n_patches
        return torch.full((b, n_patches), -100)

    def build_targets(self, inputs):
        labels = inputs.clone()
        special_token_indices = torch.isin(labels, self.special_token_idxs)
        labels[special_token_indices] = -100

        return inputs, labels


def build_common_colors_subset(dataset, config):
    with open(config.base_path + f'/CoGenT_A.json') as fp:
        color_dist = json.load(fp)

    common_colors = set(color_dist['cube']) & set(color_dist['cylinder'])
    cmn_colors_in_image = np.array([
        np.mean([o['color'] in common_colors for o in scene['objects']])
        for scene in dataset.scenes
    ])

    # CUTS = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -0.00001]
    CUTS = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,-0.0]

    subsets = []
    for cut in CUTS:
        indices = np.argwhere(cmn_colors_in_image >= cut)[:,0].tolist()
        subsets.append(Subset(dataset, indices))
        
    return subsets


class CurriculumData:
    def __init__(self, train_subsets):
        super().__init__()
        self.step = 0
        self.train_subsets = train_subsets

    @property 
    def processor(self):
        return self.current_subset.dataset.processor
        
    @property 
    def n_steps(self):
        return len(self.train_subsets)

    @property 
    def current_subset(self):
        return self.train_subsets[self.step]

    def __len__(self):
        return len(self.current_subset)

    def __getitem__(self, idx):
        return self.current_subset[idx]


class CurriculumScheduler(L.Callback):
    def __init__(self, curriculum: CurriculumData, max_epochs: int):
        self.curriculum = curriculum
        self.max_epochs = max_epochs

        n_steps = self.curriculum.n_steps # uniform segments
        self.intervals = [(s[0],s[-1]+1) for s in np.array_split(range(self.max_epochs+1), n_steps)]

    def _prepare_epoch(self, trainer, model, epoch):
        current_step = [
            step for step, interval in enumerate(self.intervals) if epoch in range(*interval)][0]
        self.curriculum.step = current_step

    def setup(self, trainer, model, stage):
        # print('CurriculumScheduler.setup')
        self._prepare_epoch(trainer, model, 0)

    def on_train_epoch_end(self, trainer, model):
        # print('CurriculumScheduler.on_train_epoch_end')
        self._prepare_epoch(trainer, model, trainer.current_epoch+1)


class NObjectsCurriculumScheduler(L.Callback):
    def __init__(self, schedule: list, max_epochs: int):
        self.schedule = schedule
        self.max_epochs = max_epochs

        n_segments = len(self.schedule) # uniform segments
        self.intervals = [(s[0],s[-1]+1) for s in np.array_split(range(self.max_epochs), n_segments)]

    def _prepare_epoch(self, trainer, model, epoch):
        current_stage = [stage for stage, interval in enumerate(self.intervals) if epoch in range(*interval)]
        current_stage = current_stage[0]
        n_objects_for_epoch = self.schedule[current_stage]
        trainer.datamodule.train_with_n_objects = n_objects_for_epoch

    def setup(self, trainer, model, stage):
        # print('CurriculumScheduler.setup')
        self._prepare_epoch(trainer, model, 0)

    def on_train_epoch_end(self, trainer, model):
        # print('CurriculumScheduler.on_train_epoch_end')
        self._prepare_epoch(trainer, model, trainer.current_epoch + 1)
 
      
class NObjectsCurriculumData(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_with_n_objects = 0 # 0: all

    def setup(self, stage: str):
        print('Setting CV learning data up')
        self.train_dataset, self.test_dataset, self.systematic_dataset = build_datasets(self.config)
        all_combinations = [tuple(range(3,i+1)) for i in range(3,10+1)] + [(0,)]
        self.train_subsets = {
            combination: self.build_subset(self.train_dataset, combination) for combination in all_combinations
        }

    def build_len_index(self, dataset):
        object_lens = defaultdict(list)
        object_lens[0] = list(range(len(dataset)))
        for idx, scene in enumerate(dataset.scenes):
            n_objects_in_scene = len(scene['objects'])
            object_lens[n_objects_in_scene].append(idx)
        return object_lens

    # @functools.lru_cache()
    def build_subset(self, dataset, size_or_sizes):
        if size_or_sizes == 0 or size_or_sizes == [0]:
            return dataset

        object_lens = self.build_len_index(dataset)
        if isinstance(size_or_sizes, int):
            sizes = [size_or_sizes]
        elif isinstance(size_or_sizes, (tuple,list)):
            sizes = size_or_sizes

        indices = sorted([idx for size in sizes for idx in object_lens[size]])
        return dataset.subset(indices)

    def train_dataloader(self):
        dataset = self.train_dataset
        if self.train_with_n_objects:
            # print(self.train_with_n_objects)
            # print('Building loader with CV subset data')
            # dataset = self.build_subset(dataset, self.train_with_n_objects)
            dataset = self.train_subsets[self.train_with_n_objects]
            # print(f'train_dataset len={len(dataset)}')
        train_loader = build_loader(dataset, self.config, shuffle=True)
        # print(f'train_loader len={len(train_loader)}')
        return train_loader

    def val_dataloader(self):
        test_loader = build_loader(self.test_dataset, self.config, shuffle=False)
        systematic_loader = build_loader(self.systematic_dataset, self.config, shuffle=False)
        test_detailed_loaders = build_detailed_test_dataloaders(self.test_dataset, self.config) # type_of_tokens_to_test
        systematic_detailed_loaders = build_detailed_test_dataloaders(self.systematic_dataset, self.config) # type_of_tokens_to_test
        return [
            test_loader, systematic_loader,
            test_detailed_loaders['color'], systematic_detailed_loaders['color'],
            test_detailed_loaders['shapes'], systematic_detailed_loaders['shapes'],
        ]

    def val_dataloader(self):
        test_loader = build_loader(self.test_dataset, self.config, shuffle=False)
        systematic_loader = build_loader(self.systematic_dataset, self.config, shuffle=False)
        test_detailed_loaders = build_detailed_test_dataloaders(self.test_dataset, self.config) # type_of_tokens_to_test
        systematic_detailed_loaders = build_detailed_test_dataloaders(self.systematic_dataset, self.config) # type_of_tokens_to_test
        return [
            test_loader, systematic_loader,
            test_detailed_loaders['color'], systematic_detailed_loaders['color'],
            test_detailed_loaders['shapes'], systematic_detailed_loaders['shapes'],
        ]


class CLEVRSplit:
    def __init__(self, questions_path, images_dir, processor=None):
        self.questions_path = questions_path
        self.images_dir = images_dir
        self.processor = processor

        print('Loading questions')
        with open(questions_path, 'r') as fp:
            self.questions = json.load(fp)['questions']

    @only_in_amd_cluster(cache)
    def load_image(self, image_filename):
        image_path = f'{self.images_dir}/{image_filename}'
        return Image.open(image_path).convert('RGB')

    def retrieve_raw(self, idx):
        question = self.questions[idx]

        question_str = question['question']
        answer_str = question['answer']
        image_filename = question['image_filename']

        image = self.load_image(image_filename)

        return image, question_str, answer_str
    
    def __getitem__(self, idx):
        image, question, answer = self.retrieve_raw(idx)
        image, question_answer = self.processor(image, question, answer)
        return image, question_answer

    @property
    def pad_idx(self):
        return self.processor.vocabulary[self.processor.pad_token]

    @property
    def n_tokens(self):
        return self.processor.n_tokens

    def iter_qa(self):
        for question in self.questions:
            yield question['question'], question['answer']

    def __len__(self):
        return len(self.questions)

    @classmethod
    def build_splits(cls, config):
        train_split = 'trainA'
        val_split = 'valA'
        test_split = 'valB'
        # common_test_split = 'valB'
        common_test_split = 'valC'
        processor = None
        property_queries = ['shape', 'size', 'color', 'material']

        questions_path = f'{config.base_path}/questions/CLEVR_{train_split}_questions.json'
        images_dir = f'{config.base_path}/images/{train_split}'     
        train_dataset = cls(questions_path, images_dir)
        
        image_transform = [ToTensor(), Resize((224,224))]
        if config.color_jitter:
            image_transform.append(ColorJitter(
                brightness=config.color_jitter_brightness, hue=config.color_jitter_hue,
                saturation=config.color_jitter_saturation, contrast=config.color_jitter_contrast,
                ))
        if not config.not_normalize_image:
            image_transform.append(Normalize(0.5, 1))
        if config.permute_pixels:
            image_transform.append(RandomPixelShuffle())
        image_transform = Compose(image_transform)
        processor = CLEVRProcessor(train_dataset, config, image_transform=image_transform)
        train_dataset.processor = processor

        if config.trainset_subset < 1. and split == train_split:
            assert False, "Subset Not Yet Implemented"

        yield train_dataset
        
        for split in [val_split, test_split, common_test_split]:
            test_datasets = []
            for property_query in property_queries:
                questions_path = f'{config.base_path}/questions/CLEVR_{split}_{property_query}_questions.json'
                images_dir = f'{config.base_path}/images/{split}'
                test_datasets.append(cls(questions_path, images_dir, processor=processor))
                       
            yield ResponsiveConcatDataset(test_datasets)
            yield from test_datasets


class CLEVRProcessor:
    def __init__(self,
                 dataset,
                 config,
                 image_transform=None,
                 pad_questions=True,
                 max_question_size=45):

        self.pad_questions = pad_questions
        self.max_question_size = config.max_question_size

        # self.aug_zero = max(config.aug_zero, 1) # older experiment used 0 as base
        # self.aug_zero_independent = config.aug_zero_independent
        # self.aug_zero_color = config.aug_zero_color

        # self.token_translations = self.load_token_translations(config)

        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        # self.special_tokens = ['[CLS]', '[PAD]', '[SEP]', '[MASK]']
        self.special_tokens = ['[CLS]', '[PAD]', '[SEP]']
        self.image_transform = image_transform
        
        self.vocabulary, self.inv_vocabulary = self.load_or_build_vocabulary(config, dataset)
        self.base_n_tokens = len(self.vocabulary)
        # self.answers_index, self.inv_answers_index = self.build_answers_index(dataset)
        
        self.non_special_tokens = [t for t in self.vocabulary if t not in self.special_tokens]
        self.special_token_idxs = [self.to_token_idx(t) for t in self.special_tokens]
        self.special_token_idxs_t = torch.tensor(self.special_token_idxs)
        self.non_special_token_idxs = [self.to_token_idx(t) for t in self.non_special_tokens]
        self.non_special_token_idxs_t = torch.tensor(self.non_special_token_idxs)
        self.cls_token_idx = self.to_token_idx(self.cls_token)
        self.pad_token_idx = self.to_token_idx(self.pad_token)
        self.mask_token_idx = self.to_token_idx(self.mask_token)
        self.color_token_idxs = [self.to_token_idx(t) for t in self.vocabulary if t in ALL_POSSIBLE_COLORS]
        self.min_color_idx = min(self.color_token_idxs)
        self.color_token_idxs_t = torch.tensor(self.color_token_idxs)
        self.n_color_tokens = len(self.color_token_idxs)

    def load_or_build_vocabulary(self, config, dataset):
        if os.path.exists(config.vocabulary_path):
            return self.load_vocabulary(config.vocabulary_path)
        else:
            return self.build_vocabulary(dataset)

    def load_vocabulary(self, vocabulary_path):
        with open(vocabulary_path) as fp:
            vocabulary = list(map(str.strip, fp.readlines()))

        vocabulary = dict(zip(vocabulary, range(len(vocabulary))))
        inv_vocabulary = dict(enumerate(vocabulary))
        return vocabulary, inv_vocabulary

    def build_vocabulary(self, dataset):
        vocabulary = set()
        
        print('Building vocabulary')
        for question_str, _ in tqdm(dataset.iter_qa(), total=len(dataset)):
            vocabulary.update(self.tokenize(question_str))

        vocabulary = sorted(list(set(vocabulary)))
        vocabulary = [self.cls_token, self.pad_token] + vocabulary

        vocabulary = dict(zip(vocabulary, range(len(vocabulary))))
        inv_vocabulary = dict(enumerate(vocabulary))
        return vocabulary, inv_vocabulary

    def build_answers_index(self, dataset):
        answers_index = []
        print('Building answers index')
        for _, answer_str in tqdm(dataset.iter_qa(), total=len(dataset)):
            answers_index.extend(answer_str.split())

        answers_index = sorted(list(set(answers_index)))
        answers_index = dict(zip(answers_index, range(len(answers_index))))
        inv_answers_index = dict(enumerate(answers_index))
        return answers_index, inv_answers_index

    def pad_question(self, question_seq):
        pads_to_use = self.max_question_size - len(question_seq)
        pads_to_use = max(pads_to_use, 0)

        padded_question = question_seq + [self.vocabulary[self.pad_token]]*pads_to_use
        padded_question = padded_question[:self.max_question_size] # if question is longer than should be

        return padded_question

    def __call__(self, image, question, answer):
        question_answer = question + ' ' + answer
        tokenized_question_answer = self.tokenize_sequence(
            question_answer, self.max_question_size, self.pad_questions)
        tokenized_question_answer = torch.tensor(tokenized_question_answer).long()

        # answer_idx = self.answers_index[answer]
        if self.image_transform:
            image = self.image_transform(image)

        return image, tokenized_question_answer

    def to_token_idx(self, word):
        # if self.token_translations:
        #     word = self.token_translations.get(word, word)
        return self.vocabulary[word]

    def tokenize_sequence(self, str_seq, max_seq_size, pad_seq, lower=True):
        tokenized_seq = [self.to_token_idx(w) for w in self.tokenize(str_seq, lower=lower)]
        tokenized_seq = [self.to_token_idx(self.cls_token)] + tokenized_seq
        if pad_seq:
            tokenized_seq = self.pad_sequence(tokenized_seq, max_seq_size)
        return tokenized_seq

    def tokenize(self, str_, lower=True):
        if lower:
            str_ = str_.lower()
        str_ = str_.replace(';', ' ;').replace('?', ' ?')
        return str_.split()

    def pad_sequence(self, sequence, max_sequence_size):
        pads_to_use = max_sequence_size - len(sequence)
        pads_to_use = max(pads_to_use, 0)

        padded_sequence = sequence + pads_to_use*[self.to_token_idx(self.pad_token)]
        padded_sequence = padded_sequence[:max_sequence_size] # if question is longer than should be

        return padded_sequence

    @property
    def n_tokens(self):
        # if self.aug_zero_color:
        #     return self.base_n_tokens + (self.aug_zero-1)*self.n_color_tokens
        # return self.aug_zero*self.base_n_tokens
        return self.base_n_tokens


class CLEVRMultimodalSplit:
    def __init__(self, scenes_path, images_dir, processor=None):
        # self.questions_path = questions_path
        self.scenes_path = scenes_path
        self.images_dir = images_dir
        self.processor = processor

        with open(scenes_path, 'r') as fp:
            self.scenes = json.load(fp)['scenes']

    def subset(self, indices):
        subset =  CLEVRMultimodalSplit(
            scenes_path=self.scenes_path, images_dir=self.images_dir, processor=self.processor)
        subset.scenes = [subset.scenes[idx]for idx in indices]
        return subset

    @only_in_amd_cluster(cache)
    def load_image(self, image_filename):
        image_path = f'{self.images_dir}/{image_filename}'
        return Image.open(image_path).convert('RGB')
    
    def retrieve_raw(self, idx):
        scene = self.scenes[idx]

        image_filename = scene['image_filename']
        image = self.load_image(image_filename)

        return image, scene

    def __getitem__(self, idx):
        image, scene = self.retrieve_raw(idx)
        image, scene_str = self.processor(image, scene)
        return image, scene_str

    @property
    def pad_idx(self):
        return self.processor.vocabulary[self.processor.pad_token]

    @property
    def n_tokens(self):
        return self.processor.n_tokens

    def __len__(self):
        return len(self.scenes)

    @classmethod
    def build_splits(cls, config):
        train_split = 'trainA'
        val_split = 'valA'
        test_split = 'valB'
        # common_test_split = 'valB'
        common_test_split = 'valC'
        processor = None

        for split in [train_split, val_split, test_split, common_test_split]:
            scenes_path = f'{config.base_path}/scenes/CLEVR_{split}_scenes.json'
            images_dir = f'{config.base_path}/images/{split}'

            if processor:
                dataset = cls(scenes_path, images_dir, processor=processor)
            else:
                dataset = cls(scenes_path, images_dir)
                image_transform = [ToTensor(), Resize((224,224))]
                if config.color_jitter:
                    image_transform.append(ColorJitter(
                        brightness=config.color_jitter_brightness, hue=config.color_jitter_hue,
                        saturation=config.color_jitter_saturation, contrast=config.color_jitter_contrast,
                        ))
                if not config.not_normalize_image:
                    image_transform.append(Normalize(0.5, 1))
                if config.permute_pixels:
                    image_transform.append(RandomPixelShuffle())
                image_transform = Compose(image_transform)
                processor = CLEVRMultimodalProcessor(dataset, config, image_transform=image_transform)
                dataset.processor = processor
                
            if config.trainset_subset < 1. and split == train_split:
                len_ = len(dataset)
                k = int(config.trainset_subset * len_)
                indices = sorted(random.sample(list(range(len_)), k=k))
                dataset = ResponsiveSubset(dataset, indices)
                print(f'Creating subset of training set of N={len(dataset)}')
                
            yield dataset


class CLEVRMultimodalFromFeaturesSplit:
    def __init__(self, scenes_path, images_features_path, processor=None):
        # self.questions_path = questions_path
        self.scenes_path = scenes_path
        self.images_features_path = images_features_path
        self.processor = processor

        with h5py.File(self.images_features_path) as ds:
            self.image_name_to_idx = json.loads(ds["image_features"].attrs["image_name_to_idx"])

        with open(scenes_path, 'r') as fp:
            self.scenes = json.load(fp)['scenes']

    def retrieve_raw(self, idx):
        scene = self.scenes[idx]

        image_filename = scene['image_filename']
        image_features = self.read_features(image_filename)

        return image_features, scene

    def read_features(self, image_name):
        with h5py.File(self.images_features_path) as ds:
            idx = self.image_name_to_idx[image_name]
            features = ds['image_features'][idx]
        return features

    def __getitem__(self, idx):
        image, scene = self.retrieve_raw(idx)

        image, scene_str = self.processor(image, scene)

        return image, scene_str

    @property
    def pad_idx(self):
        return self.processor.vocabulary[self.processor.pad_token]

    def __len__(self):
        return len(self.scenes)

    @classmethod
    def build_splits(cls, config):
        train_split = 'trainA'
        val_split = 'valA'
        test_split = 'valB'
        processor = None

        for split in [train_split, val_split, test_split]:
            embedding_type = 'vit' if config.use_vit_embedding_loaded else config.use_embedding_loaded
            scenes_path = f'{config.base_path}/scenes/CLEVR_{split}_scenes.json'
            features_path = f'{config.base_path}/images/{split}-{embedding_type}.h5'

            if processor:
                dataset = cls(scenes_path, features_path, processor=processor)
            else:
                dataset = cls(scenes_path, features_path)
                image_transform = torch.from_numpy
                processor = CLEVRMultimodalProcessor(dataset, config, image_transform=image_transform)
                dataset.processor = processor

            if config.trainset_subset < 1. and split == train_split:
                len_ = len(dataset)
                k = int(config.trainset_subset * len_)
                indices = sorted(random.sample(list(range(len_)), k=k))
                dataset = ResponsiveSubset(dataset, indices)
                print(f'Creating subset of training set of N={len(dataset)}')
            
            yield dataset


class CLEVRMultimodalTrainingSplit:
    def __init__(self, scenes_path, questions_path, images_dir, processor=None):
        # self.questions_path = questions_path
        self.scenes_path = scenes_path
        self.images_dir = images_dir
        self.processor = processor

        with open(scenes_path, 'r') as fp:
            self.scenes = json.load(fp)['scenes']

        self.indexed_scenes = {scene['image_index']: scene for scene in self.scenes}

        with open(questions_path, 'r') as fp:
            self.questions = json.load(fp)['questions']

    def retrieve_raw(self, idx):
        question = self.questions[idx]
        image_idx = question['image_index']
        scene = self.indexed_scenes[image_idx]

        image_filename = scene['image_filename']

        image_path = f'{self.images_dir}/{image_filename}'
        image = Image.open(image_path).convert('RGB')

        return image, scene, question

    def __getitem__(self, idx):
        image, scene, question = self.retrieve_raw(idx)
        return self.processor(image, scene, question)

    @property
    def pad_idx(self):
        return self.processor.vocabulary[self.processor.pad_token]

    def __len__(self):
        return len(self.scenes)

    def iter_qa(self):
        for question in self.questions:
            yield question['question'], question['answer']

    @classmethod
    def build_splits(cls, config):
        train_split = 'trainA'
        val_split = 'valA'
        test_split = 'valB'
        processor = None

        for split in [train_split, val_split, test_split]:
            scenes_path = f'{config.base_path}/scenes/CLEVR_{split}_scenes.json'
            questions_path = f'{config.base_path}/questions/CLEVR_{split}_questions.json'
            images_dir = f'{config.base_path}/images/{split}'

            if processor:
                yield cls(scenes_path, questions_path, images_dir, processor=processor)
            else:
                dataset = cls(scenes_path, questions_path, images_dir)
                image_transform = [ToTensor(), Resize((224,224))]
                if not config.not_normalize_image:
                    image_transform.append(Normalize(0.5, 1))
                image_transform = Compose(image_transform)
                processor = CLEVRMultimodalProcessor(dataset, config, image_transform=image_transform)
                dataset.processor = processor
                yield dataset


class CLEVRMultimodalProcessor:
    def __init__(self,
                 dataset,
                 config,
                 image_transform=None,
                 pad_scenes=True,
                 pad_questions=True,
                ):
        self.pad_scenes = pad_scenes
        self.pad_questions = pad_questions
        self.max_scene_size = config.max_scene_size
        self.max_question_size = config.max_question_size
        self.rels_to_sample = config.rels_to_sample
        self.only_front_right_relations = config.only_front_right_relations
        self.filter_symmetric_relations = config.filter_symmetric_relations
        self.display_object_properties = config.display_object_properties
        self.shuffle_object_identities = config.shuffle_object_identities


        self.aug_zero = max(config.aug_zero, 1) # older experiment used 0 as base
        self.aug_zero_independent = config.aug_zero_independent
        self.aug_zero_color = config.aug_zero_color

        self.token_translations = self.load_token_translations(config)

        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        # self.special_tokens = ['[CLS]', '[PAD]', '[SEP]', '[MASK]']
        self.special_tokens = ['[CLS]', '[PAD]', '[SEP]', '[MASK]'] + [f'[O{i}]' for i in range(10)]
        self.image_transform = image_transform

        self.vocabulary, self.inv_vocabulary = self.load_or_build_vocabulary(config, dataset)
        self.base_n_tokens = len(self.vocabulary)
        # if self.token_translations:
        #     self.vocabulary, self.inv_vocabulary = self.adjust_vocab_to_token_translations(self.vocabulary)

        self.non_special_tokens = [t for t in self.vocabulary if t not in self.special_tokens]
        self.special_token_idxs = [self.to_token_idx(t) for t in self.special_tokens]
        self.special_token_idxs_t = torch.tensor(self.special_token_idxs)
        self.non_special_token_idxs = [self.to_token_idx(t) for t in self.non_special_tokens]
        self.non_special_token_idxs_t = torch.tensor(self.non_special_token_idxs)
        self.cls_token_idx = self.to_token_idx(self.cls_token)
        self.pad_token_idx = self.to_token_idx(self.pad_token)
        self.mask_token_idx = self.to_token_idx(self.mask_token)
        self.color_token_idxs = [self.to_token_idx(t) for t in self.vocabulary if t in ALL_POSSIBLE_COLORS]
        self.min_color_idx = min(self.color_token_idxs)
        self.color_token_idxs_t = torch.tensor(self.color_token_idxs)
        self.n_color_tokens = len(self.color_token_idxs)

        if config.multimodal_training:
            self.answers_index, self.inv_answers_index = self.build_answers_index(dataset)

    def load_token_translations(self, config):
        if not config.token_translation_path:
            return None
        with open(config.token_translation_path) as fp:
            return json.load(fp)

    def adjust_prevocab_to_token_translations(self, vocabulary):
        original_size = len(vocabulary)
        unique_vocab = set(self.token_translations.get(t, t) for t in vocabulary)
        vocabulary = [t for t in vocabulary if t in unique_vocab]
        end_size = len(vocabulary)
        print(f'Adapting vocab from {original_size} to {end_size}')
        # vocabulary = {t: i for t, i in vocabulary.items() if t in unique_vocab}
        # inv_vocabulary = dict(enumerate(vocabulary))
        return vocabulary#, inv_vocabulary

    def build_answers_index(self, dataset):
        answers_index = []
        print('Building answers index')
        for _, answer_str in tqdm(dataset.iter_qa(), total=len(dataset)):
            answers_index.extend(answer_str.split())

        answers_index = sorted(list(set(answers_index)))
        answers_index = dict(zip(answers_index, range(len(answers_index))))
        inv_answers_index = dict(enumerate(answers_index))
        return answers_index, inv_answers_index

    def load_or_build_vocabulary(self, config, dataset):
        if os.path.exists(config.vocabulary_path):
            return self.load_vocabulary(config.vocabulary_path)
        else:
            return self.build_vocabulary(dataset)

    def load_vocabulary(self, vocabulary_path):
        with open(vocabulary_path) as fp:
            vocabulary = list(map(str.strip, fp.readlines()))

        if self.token_translations:
            vocabulary = self.adjust_prevocab_to_token_translations(vocabulary)

        vocabulary = dict(zip(vocabulary, range(len(vocabulary))))
        inv_vocabulary = dict(enumerate(vocabulary))
        return vocabulary, inv_vocabulary

    def build_vocabulary(self, dataset):
        vocabulary = set()

        print('Building vocabulary')
        for question_str, _ in tqdm(dataset.iter_qa(), total=len(dataset)):
            vocabulary.update(self.tokenize(question_str))

        for scene in dataset.scenes:
            scene_text = self.scene_to_txt(scene)
            vocabulary.update(self.tokenize(scene_text, lower=False))

        vocabulary = sorted(list(set(vocabulary)))
        vocabulary = [self.cls_token, self.pad_token] + vocabulary

        vocabulary = dict(zip(vocabulary, range(len(vocabulary))))
        inv_vocabulary = dict(enumerate(vocabulary))
        return vocabulary, inv_vocabulary

    def __call__(self, image, scene, question=None):
        tokenized_scene = self.process_scene(scene)
        image = self.process_image(image)
        # If multimodal_pretraining
        if question is None:
            return image, tokenized_scene

        # If multimodal_training
        tokenized_question = self.process_question(question)
        answer_idx = self.process_answer(question)
        return image, tokenized_scene, tokenized_question, answer_idx

    def process_image(self, image):
        if self.image_transform:
            image = self.image_transform(image)
        return image
            
    def process_scene(self, scene):
        scene_str = Scene.from_dict(scene,
                                    shuffle_relations=True,
                                    relations_to_sample=self.rels_to_sample,
                                    only_front_right=self.only_front_right_relations,
                                    filter_symmetric=self.filter_symmetric_relations,
                                    always_display_properties=self.display_object_properties,
                                    shuffle_object_identities=self.shuffle_object_identities)
        scene_str = str(scene_str)
        tokenized_scene = self.tokenize_sequence(
            scene_str, self.max_scene_size, self.pad_scenes, lower=False)
        tokenized_scene = torch.tensor(tokenized_scene).long()
        if self.aug_zero > 1:
            tokenized_scene = self.virtual_augment_scene(tokenized_scene)    
        
        return tokenized_scene
    
    @property
    def n_tokens(self):
        if self.aug_zero_color:
            return self.base_n_tokens + (self.aug_zero-1)*self.n_color_tokens
        return self.aug_zero*self.base_n_tokens
    
    def virtual_augment_scene(self, tokenized_scene):
        if self.aug_zero_color:
            tokens_not_to_augment = torch.isin(tokenized_scene, self.special_token_idxs_t)
            tokens_not_to_augment = tokens_not_to_augment | ~torch.isin(tokenized_scene, self.color_token_idxs_t)
        else:
            tokens_not_to_augment = torch.isin(tokenized_scene, self.special_token_idxs_t)

        if self.aug_zero_independent:
            aug_offset_vocab = torch.randint_like(tokenized_scene, 0, self.aug_zero)
        else:
            aug_offset_vocab = random.randint(0, self.aug_zero-1)
        
        if self.aug_zero_color:
            new_vocab_start = self.base_n_tokens - self.min_color_idx
            aug_offset = new_vocab_start + (aug_offset_vocab-1)*self.n_color_tokens
            if self.aug_zero_independent:
                aug_offset[aug_offset_vocab == 0] = 0  
            else:
                aug_offset = 0 if aug_offset_vocab == 0 else aug_offset
        else:
            aug_offset = self.base_n_tokens * aug_offset_vocab
        
        augmented_scene = tokenized_scene + aug_offset
        augmented_scene[tokens_not_to_augment] = tokenized_scene[tokens_not_to_augment]
        
        return augmented_scene
    
    def process_answer(self, question):
        answer_str = question['answer']
        answer_idx = self.answers_index[answer_str]
        return answer_idx

    def process_question(self, question):
        question_str = question['question']
        tokenized_question = self.tokenize_sequence(
            question_str, self.max_question_size, self.pad_questions)
        tokenized_question = torch.tensor(tokenized_question).long()
        return tokenized_question

    def to_token_idx(self, word):
        if self.token_translations:
            word = self.token_translations.get(word, word)
        return self.vocabulary[word]

    def tokenize_sequence(self, str_seq, max_seq_size, pad_seq, lower=True):
        tokenized_seq = [self.to_token_idx(w) for w in self.tokenize(str_seq, lower=lower)]
        tokenized_seq = [self.to_token_idx(self.cls_token)] + tokenized_seq
        if pad_seq:
            tokenized_seq = self.pad_sequence(tokenized_seq, max_seq_size)
        return tokenized_seq

    def tokenize(self, str_, lower=True):
        if lower:
            str_ = str_.lower()
        str_ = str_.replace(';', ' ;').replace('?', ' ?')
        return str_.split()

    def pad_sequence(self, sequence, max_sequence_size):
        pads_to_use = max_sequence_size - len(sequence)
        pads_to_use = max(pads_to_use, 0)

        padded_sequence = sequence + pads_to_use*[self.to_token_idx(self.pad_token)]
        padded_sequence = padded_sequence[:max_sequence_size] # if question is longer than should be

        return padded_sequence

    def scene_to_txt(self, scene, rels_to_sample=None, shuffle=True):
        objs_strs = []
        for obj_idx, obj in enumerate(scene['objects']):
            objs_strs.append(self.object_to_text(obj, obj_idx=obj_idx))

        relations_strs = self.relations_to_text(scene['relationships'])
        if rels_to_sample and (rels_to_sample < len(relations_strs)):
            relations_strs = random.sample(relations_strs, rels_to_sample)
        if shuffle:
            random.shuffle(relations_strs)

        return ' [SEP] '.join(objs_strs + relations_strs)

    def object_to_text(self, obj, obj_idx):
        properties = ['size', 'color', 'material', 'shape']
        obj_str = [f'[O{obj_idx}]'] + [obj[prop] for prop in properties]
        return ' '.join(obj_str)

    def relations_to_text(self, relations, remove_symmetric=True):
        relation_types = ['front', 'right'] if remove_symmetric else ['behind', 'front', 'left', 'right']

        relations_str = []
        for relation_type in relation_types:
            relation = relations[relation_type]
            for obj, subjs in enumerate(relation):
                relation_str = [f'[O{obj}] {relation_type} [O{subj}]' for subj in subjs]
                relations_str.extend(relation_str)

        return relations_str

    def get_obj_properties(self, start, program):
        filter_fns = {'filter_material', 'filter_color', 'filter_shape', 'filter_size'}

        next_ =  start['inputs'][0]
        properties = {}

        while program[next_]['function'] in filter_fns:
            next_fn = program[next_]
            fn_type = next_fn['function']
            property_type = fn_type.replace('filter_', '')
            properties[property_type] = next_fn['value_inputs'][0]
            next_ = next_fn['inputs'][0]

        return properties

    def object_satisfy(self, object_, properties):
        for k, v in properties.items():
            if object_[k] != v:
                return False
        return True

    def is_object_relevant(self, object_, all_properties):
        return any(self.object_satisfy(object_, properties) for properties in all_properties)

    def get_objects_to_filter(self, question, scene):
        program = question['program']
        objects = scene['objects']

        start_nodes = [fn for fn in program if fn['function'] in {'count', 'exist', 'unique', 'union'}]

        relevant_properties = []
        for obj_to_filter in start_nodes:
            relevant_properties.append(self.get_obj_properties(obj_to_filter, program))

        objects_to_filter = [
            idx for idx, o in enumerate(objects) if not self.is_object_relevant(o, relevant_properties)]

        return objects_to_filter


class CLEVRTextSplit:
    def __init__(self, questions_path, scenes_path, processor=None):
        self.questions_path = questions_path
        self.scenes_path = scenes_path
        self.processor = processor

        with open(scenes_path, 'r') as fp:
            self.scenes = json.load(fp)['scenes']

        self.indexed_scenes = {scene['image_index']: scene for scene in self.scenes}

        with open(questions_path, 'r') as fp:
            self.questions = json.load(fp)['questions']


    def __getitem__(self, idx):
        question = self.questions[idx]
        image_idx = question['image_index']
        scene = self.indexed_scenes[image_idx]

        # question_str = question['question']
        # answer_str = question['answer']

        scene_str, question_str, answer_str = self.processor(scene, question)

        return scene_str, question_str, answer_str

    @property
    def pad_idx(self):
        return self.processor.vocabulary[self.processor.pad_token]

    def iter_qa(self):
        for question in self.questions:
            yield question['question'], question['answer']

    def __len__(self):
        return len(self.questions)

    @classmethod
    def build_splits(cls, config):
        train_split = 'trainA'
        val_split = 'valA'
        test_split = 'valB'
        processor = None

        for split in [train_split, val_split, test_split]:
            questions_path = f'{config.base_path}/questions/CLEVR_{split}_questions.json'
            scenes_path = f'{config.base_path}/scenes/CLEVR_{split}_scenes.json'

            if processor:
                yield cls(questions_path, scenes_path, processor=processor)
            else:
                dataset = cls(questions_path, scenes_path)
                processor = CLEVRTextProcessor(dataset, config)
                dataset.processor = processor
                yield dataset


class CLEVRTextProcessor:
    def __init__(self,
                 dataset,
                 config,
                 pad_questions=True,
                 pad_scenes=True,
                 remove_unneeded_relations=False
                ):

        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.vocabulary, self.inv_vocabulary = self.load_or_build_vocabulary(config, dataset)
        self.answers_index, self.inv_answers_index = self.build_answers_index(dataset)
        self.pad_questions = pad_questions
        self.pad_scenes = pad_scenes
        self.max_question_size = config.max_question_size
        self.max_scene_size = config.max_scene_size
        self.rels_to_sample = config.rels_to_sample
        self.only_front_right_relations = config.only_front_right_relations
        self.filter_symmetric_relations = config.filter_symmetric_relations
        self.display_object_properties = config.display_object_properties
        self.shuffle_object_identities = config.shuffle_object_identities
        self.remove_unneeded_relations = remove_unneeded_relations

    def load_or_build_vocabulary(self, config, dataset):
        if os.path.exists(config.vocabulary_path):
            return self.load_vocabulary(config.vocabulary_path)
        else:
            return self.build_vocabulary(dataset)

    def load_vocabulary(self, vocabulary_path):
        with open(vocabulary_path) as fp:
            vocabulary = list(map(str.strip, fp.readlines()))

        vocabulary = dict(zip(vocabulary, range(len(vocabulary))))
        inv_vocabulary = dict(enumerate(vocabulary))
        return vocabulary, inv_vocabulary


    def build_vocabulary(self, dataset):
        vocabulary = set()

        print('Building vocabulary')
        for question_str, _ in tqdm(dataset.iter_qa(), total=len(dataset)):
            vocabulary.update(self.tokenize(question_str))

        for scene in dataset.scenes:
            scene_text = self.scene_to_txt(scene)
            vocabulary.update(self.tokenize(scene_text, lower=False))

        vocabulary = sorted(list(set(vocabulary)))
        vocabulary = [self.cls_token, self.pad_token] + vocabulary

        vocabulary = dict(zip(vocabulary, range(len(vocabulary))))
        inv_vocabulary = dict(enumerate(vocabulary))
        return vocabulary, inv_vocabulary

    def build_answers_index(self, dataset):
        answers_index = []
        print('Building answers index')
        for _, answer_str in tqdm(dataset.iter_qa(), total=len(dataset)):
            answers_index.extend(answer_str.split())

        answers_index = sorted(list(set(answers_index)))
        answers_index = dict(zip(answers_index, range(len(answers_index))))
        inv_answers_index = dict(enumerate(answers_index))
        return answers_index, inv_answers_index

    def __call__(self, scene, question):
        # scene, question_str, answer_str = clevr_sample
        question_str = question['question']
        answer_str = question['answer']

        filter_objects_from_relations = None
        if self.remove_unneeded_relations:
            filter_objects_from_relations = self.get_objects_to_filter(question, scene)

        # scene_str = self.scene_to_txt(scene, rels_to_sample=self.rels_to_sample)
        scene_str = Scene.from_dict(scene,
                                    shuffle_relations=True,
                                    relations_to_sample=self.rels_to_sample,
                                    only_front_right=self.only_front_right_relations,
                                    filter_symmetric=self.filter_symmetric_relations,
                                    always_display_properties=self.display_object_properties,
                                    filter_objects_from_relations=filter_objects_from_relations,
                                    shuffle_object_identities=self.shuffle_object_identities)
        scene_str = str(scene_str)

        tokenized_scene = self.tokenize_sequence(
            scene_str, self.max_scene_size, self.pad_scenes, lower=False)
        tokenized_question = self.tokenize_sequence(
            question_str, self.max_question_size, self.pad_questions)
        # tokenized_scene = [self.vocabulary[w] for w in self.tokenize(scene_str, lower=False)]
        # tokenized_scene = [self.vocabulary[self.cls_token]] + tokenized_scene
        # if self.pad_scenes:
        #     tokenized_scene = self.pad_sequence(tokenized_scene, self.max_scene_size)

        # tokenized_question = [self.vocabulary[w] for w in self.tokenize(question_str)]
        # tokenized_question = [self.vocabulary[self.cls_token]] + tokenized_question
        # if self.pad_questions:
        #     tokenized_question = self.pad_sequence(tokenized_question, self.max_question_size)

        tokenized_scene = torch.tensor(tokenized_scene).long()
        tokenized_question = torch.tensor(tokenized_question).long()

        answer_idx = self.answers_index[answer_str]

        return tokenized_scene, tokenized_question, answer_idx

    def tokenize_sequence(self, str_seq, max_seq_size, pad_seq, lower=True):
        tokenized_seq = [self.vocabulary[w] for w in self.tokenize(str_seq, lower=lower)]
        tokenized_seq = [self.vocabulary[self.cls_token]] + tokenized_seq
        if pad_seq:
            tokenized_seq = self.pad_sequence(tokenized_seq, max_seq_size)
        return tokenized_seq

    def tokenize(self, str_, lower=True):
        if lower:
            str_ = str_.lower()
        str_ = str_.replace(';', ' ;').replace('?', ' ?')
        return str_.split()

    def pad_sequence(self, sequence, max_sequence_size):
        pads_to_use = max_sequence_size - len(sequence)
        pads_to_use = max(pads_to_use, 0)

        padded_sequence = sequence + pads_to_use*[self.vocabulary[self.pad_token]]
        padded_sequence = padded_sequence[:max_sequence_size] # if question is longer than should be

        return padded_sequence

    def scene_to_txt(self, scene, rels_to_sample=None, shuffle=True):
        objs_strs = []
        for obj_idx, obj in enumerate(scene['objects']):
            objs_strs.append(self.object_to_text(obj, obj_idx=obj_idx))

        relations_strs = self.relations_to_text(scene['relationships'])
        if rels_to_sample and (rels_to_sample < len(relations_strs)):
            relations_strs = random.sample(relations_strs, rels_to_sample)
        if shuffle:
            random.shuffle(relations_strs)

        return ' [SEP] '.join(objs_strs + relations_strs)

    def object_to_text(self, obj, obj_idx):
        properties = ['size', 'color', 'material', 'shape']
        obj_str = [f'[O{obj_idx}]'] + [obj[prop] for prop in properties]
        return ' '.join(obj_str)

    def relations_to_text(self, relations, remove_symmetric=True):
        relation_types = ['front', 'right'] if remove_symmetric else ['behind', 'front', 'left', 'right']

        relations_str = []
        for relation_type in relation_types:
            relation = relations[relation_type]
            for obj, subjs in enumerate(relation):
                relation_str = [f'[O{obj}] {relation_type} [O{subj}]' for subj in subjs]
                relations_str.extend(relation_str)

        return relations_str

    def get_obj_properties(self, start, program):
        filter_fns = {'filter_material', 'filter_color', 'filter_shape', 'filter_size'}

        next_ =  start['inputs'][0]
        properties = {}

        while program[next_]['function'] in filter_fns:
            next_fn = program[next_]
            fn_type = next_fn['function']
            property_type = fn_type.replace('filter_', '')
            properties[property_type] = next_fn['value_inputs'][0]
            next_ = next_fn['inputs'][0]

        return properties

    def object_satisfy(self, object_, properties):
        for k, v in properties.items():
            if object_[k] != v:
                return False
        return True

    def is_object_relevant(self, object_, all_properties):
        return any(self.object_satisfy(object_, properties) for properties in all_properties)

    def get_objects_to_filter(self, question, scene):
        program = question['program']
        objects = scene['objects']

        start_nodes = [fn for fn in program if fn['function'] in {'count', 'exist', 'unique', 'union'}]

        relevant_properties = []
        for obj_to_filter in start_nodes:
            relevant_properties.append(self.get_obj_properties(obj_to_filter, program))

        objects_to_filter = [
            idx for idx, o in enumerate(objects) if not self.is_object_relevant(o, relevant_properties)]

        return objects_to_filter

class Scene:
    def __init__(self,
                 objects,
                 relations,
                 relations_to_sample=0,
                 only_front_right=True,
                 filter_symmetric=True,
                 shuffle_relations=True,
                 always_display_properties=False,
                 filter_objects_from_relations=None,
                 shuffle_object_identities=False):

        self.relations_to_sample = relations_to_sample
        self.filter_symmetric = filter_symmetric
        self.shuffle_relations = shuffle_relations
        self.always_display_properties = always_display_properties
        self.shuffle_object_identities = shuffle_object_identities
        self.object_idx_to_id = list(range(len(objects)))
        if shuffle_object_identities:
            random.shuffle(self.object_idx_to_id)

        self.relation_types = ['front', 'right'] if only_front_right else ['behind', 'front', 'left', 'right']

        self.objects = self.build_objects(objects)
        self.relations = self.build_relations(relations, filter_objects=filter_objects_from_relations)
        self.id_to_object = {obj.object_id: obj for obj in self.objects}

    def build_objects(self, objects):
        objs_list = []
        for obj_idx, obj in enumerate(objects):
            obj_id = self.object_idx_to_id[obj_idx]
            objs_list.append(SceneObject(obj_id, obj))
        return objs_list

    def build_relations(self, relations, filter_objects=None):
        filter_objects = [] if filter_objects is None else filter_objects
        is_relevant = lambda o_idx: self.object_idx_to_id[o_idx] not in filter_objects

        processed_relations = []
        for relation_type in self.relation_types:
            relation_subjs = relations[relation_type]
            for obj_idx, subjs_idxs in enumerate(relation_subjs):
                obj_id = self.object_idx_to_id[obj_idx]
                subj_ids = [self.object_idx_to_id[subjs_idx] for subjs_idx in subjs_idxs]
                relation_str = [SceneRelation(obj_id, relation_type, subj_id) for subj_id in subj_ids
                                if is_relevant(obj_id) or is_relevant(subj_id)]
                processed_relations.extend(relation_str)

        if self.shuffle_relations:
            random.shuffle(processed_relations)
        if self.filter_symmetric:
            processed_relations = list(set(processed_relations))
        if self.relations_to_sample >=0 and (self.relations_to_sample < len(processed_relations)):
            processed_relations = random.sample(processed_relations, self.relations_to_sample)
        if self.shuffle_relations:
            random.shuffle(processed_relations)

        return processed_relations

    @classmethod
    def from_dict(cls, scene_dict, **kwargs):
        return cls(scene_dict['objects'], scene_dict['relationships'], **kwargs)

    def __str__(self):
        objects_str_list = []
        for object in self.objects:
            objects_str_list.append(f'{object.property_str()}')
            # objects_str_list.append(f'{object.identity_str()} {object.property_str()}')
        if self.shuffle_object_identities:
            random.shuffle(objects_str_list)

        relations_str_list = []
        for relation in self.relations:
            object = self.id_to_object[relation.object_id]
            subject = self.id_to_object[relation.subject_id]
            if self.always_display_properties:
                object_str = f'{object.identity_str()} {object.property_str()}'
                subject_str = f'{subject.identity_str()} {subject.property_str()}'
            else:
                object_str = f'{object.identity_str()}'
                subject_str = f'{subject.identity_str()}'

            relations_str_list.append(f'{object_str} {relation.relation_type} {subject_str}')

        return ' [SEP] '.join(objects_str_list + relations_str_list)

class SceneObject:
    objects_properties = ['size', 'color', 'material', 'shape']

    def __init__(self, object_id, properties):
        self.object_id = object_id
        self.properties = {property: properties[property] for property in self.objects_properties}

    def __hash__(self) -> int:
        return self.object_id

    def identity_str(self):
        return f'[O{self.object_id}]'

    def property_str(self):
        return ' '.join([self.properties[prop] for prop in self.objects_properties])

    def __repr__(self) -> str:
        # property_str = ' '.join([f'{k}: {v}' for k, v in self.properties.items()])
        return f'{self.identity_str()}: {self.property_str()}'

class SceneRelation:
    def __init__(self, object_id, relation_type, subject_id):
        self.object_id = object_id
        self.relation_type = relation_type
        self.subject_id = subject_id

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self) -> int:
        if self.relation_type in ('front', 'right'):
            return hash((self.object_id, self.relation_type, self.subject_id))

        relation_type_for_hash = {
            'behind': 'front', 'left': 'right'
        }[self.relation_type]
        return hash((self.subject_id, relation_type_for_hash, self.object_id))

    def __repr__(self) -> str:
        return f'[O{self.object_id}] {self.relation_type} [O{self.subject_id}]'


ALL_POSSIBLE_COLORS = list({
    'gray', # old
    'alice-blue','antique-white','aqua','aqua-marine','azure','beige','bisque','black',
    'blanched-almond','blue','blue-violet','brown','burly-wood','cadet-blue','chartreuse',
    'chocolate','coral','corn-flower-blue','corn-silk','crimson','cyan','dark-blue','dark-cyan',
    'dark-golden-rod','dark-green','dark-grey','dark-khaki','dark-magenta','dark-olive-green',
    'dark-orange','dark-orchid','dark-red','dark-salmon','dark-sea-green','dark-slate-blue',
    'dark-slate-gray','dark-turquoise','dark-violet','deep-pink','deep-sky-blue','dim-grey',
    'dodger-blue','firebrick','floral-white','forest-green','gainsboro','ghost-white','gold',
    'golden-rod','green','green-yellow','grey','honeydew','hot-pink','indian-red','indigo',
    'ivory','khaki','lavender','lavender-blush','lawn-green','lemon-chiffon','light-blue',
    'light-coral','light-cyan','light-golden-rod-yellow','light-green','light-grey','light-pink',
    'light-salmon','light-sea-green','light-sky-blue','light-slate-gray','light-steel-blue',
    'light-yellow','lime','lime-green','linen','magenta-','maroon','medium-aqua-marine',
    'medium-blue','medium-orchid','medium-purple','medium-sea-green','medium-slate-blue',
    'medium-spring-green','medium-turquoise','medium-violet-red','midnight-blue','mint-cream',
    'misty-rose','moccasin','navajo-white','navy','old-lace','olive','olive-drab','orange',
    'orange-red','orchid','pale-golden-rod','pale-green','pale-turquoise','pale-violet-red',
    'papaya-whip','peach-puff','peru','pink','plum','powder-blue','purple','red','rosy-brown',
    'royal-blue','saddle-brown','salmon','sandy-brown','sea-green','sea-shell','sienna','silver',
    'sky-blue','slate-blue','slate-gray','snow','spring-green','steel-blue','tan','teal','thistle',
    'tomato','turquoise','violet','wheat','white','white-smoke','yellow','yellow-green',


    '#000000', '#00002d', '#000059', '#000086', '#0000b2', '#0000df', '#002d00', '#002d2d', '#002d59',
    '#002d86', '#002db2', '#002ddf', '#005900', '#00592d', '#005959', '#005986', '#0059b2', '#0059df',
    '#008600', '#00862d', '#008659', '#008686', '#0086b2', '#0086df', '#00b200', '#00b22d', '#00b259',
    '#00b286', '#00b2b2', '#00b2df', '#00df00', '#00df2d', '#00df59', '#00df86', '#00dfb2', '#00dfdf',
    '#2d0000', '#2d002d', '#2d0059', '#2d0086', '#2d00b2', '#2d00df', '#2d2d00', '#2d2d2d', '#2d2d59',
    '#2d2d86', '#2d2db2', '#2d2ddf', '#2d5900', '#2d592d', '#2d5959', '#2d5986', '#2d59b2', '#2d59df',
    '#2d8600', '#2d862d', '#2d8659', '#2d8686', '#2d86b2', '#2d86df', '#2db200', '#2db22d', '#2db259',
    '#2db286', '#2db2b2', '#2db2df', '#2ddf00', '#2ddf2d', '#2ddf59', '#2ddf86', '#2ddfb2', '#2ddfdf',
    '#590000', '#59002d', '#590059', '#590086', '#5900b2', '#5900df', '#592d00', '#592d2d', '#592d59',
    '#592d86', '#592db2', '#592ddf', '#595900', '#59592d', '#595959', '#595986', '#5959b2', '#5959df',
    '#598600', '#59862d', '#598659', '#598686', '#5986b2', '#5986df', '#59b200', '#59b22d', '#59b259',
    '#59b286', '#59b2b2', '#59b2df', '#59df00', '#59df2d', '#59df59', '#59df86', '#59dfb2', '#59dfdf',
    '#860000', '#86002d', '#860059', '#860086', '#8600b2', '#8600df', '#862d00', '#862d2d', '#862d59',
    '#862d86', '#862db2', '#862ddf', '#865900', '#86592d', '#865959', '#865986', '#8659b2', '#8659df',
    '#868600', '#86862d', '#868659', '#868686', '#8686b2', '#8686df', '#86b200', '#86b22d', '#86b259',
    '#86b286', '#86b2b2', '#86b2df', '#86df00', '#86df2d', '#86df59', '#86df86', '#86dfb2', '#86dfdf',
    '#b20000', '#b2002d', '#b20059', '#b20086', '#b200b2', '#b200df', '#b22d00', '#b22d2d', '#b22d59',
    '#b22d86', '#b22db2', '#b22ddf', '#b25900', '#b2592d', '#b25959', '#b25986', '#b259b2', '#b259df',
    '#b28600', '#b2862d', '#b28659', '#b28686', '#b286b2', '#b286df', '#b2b200', '#b2b22d', '#b2b259',
    '#b2b286', '#b2b2b2', '#b2b2df', '#b2df00', '#b2df2d', '#b2df59', '#b2df86', '#b2dfb2', '#b2dfdf',
    '#df0000', '#df002d', '#df0059', '#df0086', '#df00b2', '#df00df', '#df2d00', '#df2d2d', '#df2d59',
    '#df2d86', '#df2db2', '#df2ddf', '#df5900', '#df592d', '#df5959', '#df5986', '#df59b2', '#df59df',
    '#df8600', '#df862d', '#df8659', '#df8686', '#df86b2', '#df86df', '#dfb200', '#dfb22d', '#dfb259',
    '#dfb286', '#dfb2b2', '#dfb2df', '#dfdf00', '#dfdf2d', '#dfdf59', '#dfdf86', '#dfdfb2', '#dfdfdf',

    '#33ccff', '#bf00ff', '#bfff00', '#ffbfbf', '#008080', '#ff3300', '#80ff80', '#008000', '#339999',
    '#55aa00', '#666600', '#404040', '#aaaa55', '#339900', '#003300', '#006600', '#5555aa', '#aa5555',
    '#00ffaa', '#333366', '#ff6600', '#99ff33', '#ffff66', '#9900cc', '#00ccff', '#4000ff', '#993366',
    '#66cc99', '#55ff00', '#33cc99', '#ff55aa', '#660000', '#55aaaa', '#ffbfff', '#aaff55', '#400000',
    '#0000bf', '#bf0080', '#bfbfbf', '#804040', '#33ff66', '#40ff00', '#996699', '#003366', '#ffaaaa',
    '#663366', '#400040', '#ff0099', '#5500aa', '#6699cc', '#330033', '#000055', '#ccff33', '#404000',
    '#33cc00', '#0040bf', '#ff4040', '#6633cc', '#ff9900', '#bf8080', '#004000', '#ff0055', '#99ff00',
    '#bfbf00', '#00bfff', '#0000aa', '#0080ff', '#55aaff', '#404080', '#8040ff', '#33ffff', '#9999cc',
    '#33cccc', '#ff5555', '#ff00cc', '#cccccc', '#0040ff', '#40ffff', '#ffbf80', '#55ff55', '#ff9966',
    '#004080', '#00aaaa', '#ffff80', '#55ffff', '#cc3399', '#ff0000', '#999900', '#ffff40', '#009933',
    '#804000', '#bf0040', '#0033ff', '#ff55ff', '#cc0000', '#996600', '#80bf40', '#00ffbf', '#bf40bf',
    '#66ff00', '#339966', '#9900ff', '#00ff55', '#aaffff', '#aaff00', '#800040', '#ff40ff', '#6666cc',
    '#80bfff', '#9933ff', '#660066', '#663300', '#99cccc', '#400080', '#ffffcc', '#33cc66', '#80ff00',
    '#009966', '#00bf00', '#ff00ff', '#996633', '#aaaaaa', '#00ff00', '#cc6600', '#3333cc', '#40bfff',
    '#40bfbf', '#9933cc', '#cc9900', '#80ff40', '#408000', '#0000cc', '#33ffcc', '#ccff00', '#ff40bf',
    '#ccff66', '#ff6666', '#4080ff', '#bf0000', '#55ffaa', '#808000', '#999933', '#000080', '#bf4040',
    '#ffcc99', '#33ff33', '#3300ff', '#00ff66', '#aa0000', '#009999', '#00ff80', '#cc00ff', '#008040',
    '#0099cc', '#000033', '#ff3333', '#550000', '#bfff80', '#bfbf40', '#00ff33', '#cc3300', '#660099',
    '#ff0040', '#800000', '#aa00ff', '#80bfbf', '#6666ff', '#5555ff', '#009900', '#333399', '#00bf80',
    '#bfffbf', '#80bf00', '#8080bf', '#ff80bf', '#bf4080', '#9999ff', '#408080', '#ccff99', '#99ffff',
    '#33ff99', '#ffff55', '#3366cc', '#99cc00', '#4080bf', '#cccc33', '#336633', '#cc3366', '#000000',
    '#006666', '#ffcc66', '#66ff66', '#0055aa', '#3399ff', '#80ffff', '#33cc33', '#993300', '#00ff40',
    '#99cc99', '#aa55ff', '#cc9999', '#ff99ff', '#ff0066', '#ff4080', '#4040ff', '#ff66cc', '#66cc33',
    '#ffbf40', '#8080ff', '#aaffaa', '#66cc00', '#669966', '#996666', '#cc6699', '#66cccc', '#8000bf',
    '#ffff99', '#66ccff', '#003333', '#bf8040', '#663399', '#40ff40', '#bfffff', '#ffff00', '#aa55aa',
    '#333333', '#4000bf', '#006699', '#bf80ff', '#ffffaa', '#0033cc', '#000066', '#bf40ff', '#cc6633',
    '#cc9933', '#ffffff', '#80bf80', '#ff00bf', '#0099ff', '#ffcc00', '#333300', '#99cc33', '#40bf00',
    '#006633', '#5500ff', '#ff66ff', '#cc66cc', '#ff8000', '#666666', '#cc3333', '#990099', '#550055',
    '#005555', '#55aa55', '#ff3399', '#339933', '#00bfbf', '#6600ff', '#8040bf', '#ffaaff', '#33ff00',
    '#330000', '#3366ff', '#ff5500', '#bf8000', '#66ffff', '#66ff99', '#808040', '#669999', '#555555',
    '#808080', '#99ff66', '#00aa55', '#9966cc', '#40bf80', '#993399', '#999966', '#ff4000', '#99ccff',
    '#ffbf00', '#cc99ff', '#ccccff', '#800080', '#00ffff', '#99cc66', '#bf00bf', '#bfbfff', '#0066cc',
    '#666633', '#ccffff', '#336699', '#00cc99', '#00cc33', '#990000', '#cc99cc', '#990066', '#ff6633',
    '#0055ff', '#004040', '#ffcccc', '#993333', '#00cccc', '#666699', '#00ff99', '#555500', '#330099',
    '#cc66ff', '#3300cc', '#00aaff', '#99ff99', '#ff80ff', '#aa00aa', '#cc9966', '#ff9933', '#cc6666',
    '#ffaa55', '#999999', '#00bf40', '#ff6699', '#00ffcc', '#660033', '#669900', '#cc00cc', '#8000ff',
    '#bfff40', '#aa0055', '#cc0066', '#ff3366', '#bfbf80', '#ff8080', '#6600cc', '#ff9999', '#ffffbf',
    '#0066ff', '#005500', '#bf80bf', '#330066', '#66ff33', '#ff33ff', '#0000ff', '#6699ff', '#00aa00',
    '#aaaaff', '#cccc66', '#cc0033', '#66ffcc', '#99ffcc', '#ff33cc', '#aaaa00', '#ff0080', '#cc33cc',
    '#336600', '#ffcc33', '#cc0099', '#00cc66', '#ff99cc', '#4040bf', '#804080', '#990033', '#ffccff',
    '#3399cc', '#669933', '#cccc00', '#40ffbf', '#bf4000', '#ff8040', '#cc33ff', '#40ff80', '#cccc99',
    '#ffff33', '#40bf40', '#66cc66', '#80ffbf', '#ccffcc', '#000040', '#aa5500', '#000099', '#3333ff',
    '#6633ff', '#9966ff', '#003399', '#408040', '#336666', '#ff0033', '#ff00aa', '#ffaa00', '#0080bf',
    '#663333', '#00cc00'})