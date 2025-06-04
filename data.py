import copy
import os
import json
import random
from tqdm.auto import tqdm
from PIL import Image
import torch
from torch.utils.data import default_collate
from torch.utils.data import DataLoader
from torch.utils.data import Subset, ConcatDataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ColorJitter


def build_datasets(config):
    print('Building CLEVRMultimodalSplit')
    return CLEVRMultimodalSplit.build_splits(config)


def build_loader(dataset, config, shuffle=True, collate_fn=None):
    cpus_for_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    num_workers = cpus_for_task
    dlkwargs = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if collate_fn:
        dlkwargs['collate_fn'] = collate_fn
    else:
        dlkwargs['collate_fn'] = CollatorForMaskedLanguageModeling(
            config, dataset.processor, mlm_probability=config.mlm_probability)

    dlkwargs['batch_size'] = config.batch_size
    dlkwargs['shuffle'] = shuffle

    return DataLoader(dataset, **dlkwargs)


def build_detailed_test_dataloaders(dataset, config, type_of_tokens_to_test=None):
    processor = dataset.processor
    vocab = processor.vocabulary
    tokens_to_test = {
        'relation': ['left', 'right', 'behind', 'front'],
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

    def __getitems__(self, indices):
        return [self[idx] for idx in indices]


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


class CLEVRMultimodalSplit:
    def __init__(self, scenes_path, images_dir, processor=None):
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

    @staticmethod
    def build_image_transform(config, train=True):
        resize = Resize((config.image_size,config.image_size))
        image_transform = [ToTensor(), resize]

        if train and config.color_jitter:
            image_transform.append(ColorJitter(
                brightness=config.color_jitter_brightness, hue=config.color_jitter_hue,
                saturation=config.color_jitter_saturation, contrast=config.color_jitter_contrast,
                ))
        if not config.not_normalize_image:
            image_transform.append(Normalize(0.5, 1))

        image_transform = Compose(image_transform)
        return image_transform

    @classmethod
    def build_splits(cls, config):
        train_split = 'trainA'
        val_split = 'valA'
        test_split = 'valB'
        # common_test_split = 'valB'
        common_test_split = 'valC'

        split = train_split

        scenes_path = f'{config.base_path}/scenes/CLEVR_{split}_scenes.json'
        images_dir = f'{config.base_path}/images/{split}'


        dataset = cls(scenes_path, images_dir)
        train_image_transform = cls.build_image_transform(config, train=True)
        train_processor = CLEVRMultimodalProcessor(dataset, config, image_transform=train_image_transform)
        dataset.processor = train_processor

        if config.mixture_path and config.p_mixture > 0:
            mixture_scenes_path = f'{config.mixture_path}/scenes/CLEVR_{split}_scenes.json'
            mixture_images_dir = f'{config.mixture_path}/images/{split}'
            mixture_dataset = cls(mixture_scenes_path, mixture_images_dir, processor=train_processor)

            k = int(config.p_mixture * len(dataset))
            random.seed(config.seed + k)
            indices = sorted(random.sample(list(range(len(dataset))), k=k))
            indices_set = set(indices)
            mixture_dataset = ResponsiveSubset(mixture_dataset, indices)
            complemented_indices = [i for i in range(len(dataset)) if i not in indices_set]
            dataset = ResponsiveSubset(dataset, complemented_indices)
            dataset = ResponsiveConcatDataset([dataset, mixture_dataset])

        if config.trainset_subset < 1.:
            k = int(config.trainset_subset * len(dataset))
            random.seed(config.seed + k)
            indices = sorted(random.sample(list(range(len(dataset))), k=k))
            dataset = ResponsiveSubset(dataset, indices)
            print(f'Creating subset of training set of N={len(dataset)}')

        yield dataset

        for split in [val_split, test_split, common_test_split]:
            scenes_path = f'{config.base_path}/scenes/CLEVR_{split}_scenes.json'
            images_dir = f'{config.base_path}/images/{split}'

            test_processor = copy.deepcopy(train_processor)
            test_processor.image_transform = cls.build_image_transform(config, train=False)
            dataset = cls(scenes_path, images_dir, processor=test_processor)

            yield dataset


class CLEVRMultimodalProcessor:
    def __init__(self,
                 dataset,
                 config,
                 image_transform=None,
                 pad_scenes=True,
                ):
        self.pad_scenes = pad_scenes
        self.max_scene_size = config.max_scene_size
        self.rels_to_sample = config.rels_to_sample
        self.only_front_right_relations = config.only_front_right_relations
        self.filter_symmetric_relations = config.filter_symmetric_relations
        self.display_object_properties = config.display_object_properties
        self.shuffle_object_identities = config.shuffle_object_identities

        self.token_translations = self.load_token_translations(config)

        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        self.special_tokens = ['[CLS]', '[PAD]', '[SEP]', '[MASK]'] + [f'[O{i}]' for i in range(10)]
        self.image_transform = image_transform

        self.vocabulary, self.inv_vocabulary = self.load_or_build_vocabulary(config, dataset)
        self.base_n_tokens = len(self.vocabulary)

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

    def __call__(self, image, scene):
        tokenized_scene = self.process_scene(scene)
        image = self.process_image(image)
        return image, tokenized_scene

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

        return tokenized_scene

    @property
    def n_tokens(self):
        return self.base_n_tokens

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
        padded_sequence = padded_sequence[:max_sequence_size]

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
