import os
import json
import random
from tqdm.auto import tqdm
from PIL import Image
import torch
from torchvision.transforms import Compose, Normalize, ToTensor


class CLEVRSplit:
    def __init__(self, questions_path, images_dir, processor=None):
        self.questions_path = questions_path
        self.images_dir = images_dir

        print('Loading questions')
        with open(questions_path, 'r') as fp:
            self.questions = json.load(fp)['questions']

        self.processor = processor

    def __getitem__(self, idx):
        question = self.questions[idx]

        question_str = question['question']
        answer_str = question['answer']
        image_filename = question['image_filename']

        image_path = f'{self.images_dir}/{image_filename}'
        try:
            image = Image.open(image_path).convert('RGB')
        except OSError:
            print(image_path)

        if self.processor:
            image, question_str, answer_str = self.processor(image, question_str, answer_str)

        return image, question_str, answer_str

    @property
    def pad_idx(self):
        return self.processor.vocabulary[self.processor.pad_token]

    def iter_qa(self):
        for question in self.questions:
            yield question['question'], question['answer']

    def __len__(self):
        return len(self.questions)

    @classmethod
    def build_splits(cls, base_path):
        train_split = 'trainA'
        val_split = 'valA'
        test_split = 'valB'
        processor = None

        for split in [train_split, val_split, test_split]:
            questions_path = f'{base_path}/questions/CLEVR_{split}_questions.json'
            images_dir = f'{base_path}/images/{split}'

            if processor:
                yield cls(questions_path, images_dir, processor=processor)
            else:
                dataset = cls(questions_path, images_dir)

                image_transform = Compose([ToTensor(), Normalize(0.5, 1)])
                processor = CLEVRProcessor(dataset, image_transform=image_transform)
                dataset.processor = processor

                yield dataset


class CLEVRProcessor:
    def __init__(self, dataset, image_transform=None, pad_questions=True, max_question_size=45):
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.vocabulary, self.inv_vocabulary = self.build_vocabulary(dataset)
        self.answers_index, self.inv_answers_index = self.build_answers_index(dataset)
        self.image_transform = image_transform
        self.pad_questions = pad_questions
        self.max_question_size = max_question_size

    def build_vocabulary(self, dataset):
        vocabulary = []
        print('Building vocabulary')
        for question_str, _ in tqdm(dataset.iter_qa(), total=len(dataset)):
            vocabulary.extend(question_str.split())

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

    def __call__(self, *clevr_sample):
        image, question_str, answer_str = clevr_sample

        tokenized_question = [self.vocabulary[w] for w in question_str.split()]
        tokenized_question = [self.vocabulary[self.cls_token]] + tokenized_question
        if self.pad_questions:
            tokenized_question = self.pad_question(tokenized_question)

        tokenized_question = torch.tensor(tokenized_question)

        answer_idx = self.answers_index[answer_str]

        if self.image_transform:
            image = self.image_transform(image)

        return image, tokenized_question, answer_idx


class CLEVRMultimodalSplit:
    def __init__(self, scenes_path, images_dir, processor=None):
        # self.questions_path = questions_path
        self.scenes_path = scenes_path
        self.images_dir = images_dir
        self.processor = processor

        with open(scenes_path, 'r') as fp:
            self.scenes = json.load(fp)['scenes']

        self.indexed_scenes = {scene['image_index']: scene for scene in self.scenes}


    def __getitem__(self, idx):
        scene = self.scenes[idx]

        image_filename = scene['image_filename']

        image_path = f'{self.images_dir}/{image_filename}'
        try:
            image = Image.open(image_path).convert('RGB')
        except OSError:
            print(image_path)

        image, scene_str = self.processor(image, scene)

        return image, scene_str

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
            scenes_path = f'{config.base_path}/scenes/CLEVR_{split}_scenes.json'
            images_dir = f'{config.base_path}/images/{split}'

            if processor:
                yield cls(scenes_path, images_dir, processor=processor)
            else:
                dataset = cls(scenes_path, images_dir)
                image_transform = Compose([ToTensor(), Normalize(0.5, 1)])
                processor = CLEVRMultimodalProcessor(dataset, config, image_transform=image_transform)
                dataset.processor = processor
                yield dataset


class CLEVRMultimodalProcessor:
    def __init__(self,
                 dataset,
                 config,
                 image_transform=None,
                 pad_scenes=True,
                ):

        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.image_transform = image_transform
        self.vocabulary, self.inv_vocabulary = self.load_or_build_vocabulary(config, dataset)
        self.pad_scenes = pad_scenes
        self.max_scene_size = config.max_scene_size
        self.rels_to_sample = config.rels_to_sample
        self.only_front_right_relations = config.only_front_right_relations
        self.filter_symmetric_relations = config.filter_symmetric_relations
        self.display_object_properties = config.display_object_properties
        self.shuffle_object_identities = config.shuffle_object_identities

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

    def __call__(self, image, scene):
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

        tokenized_scene = torch.tensor(tokenized_scene)

        if self.image_transform:
            image = self.image_transform(image)

        return image, tokenized_scene

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

        tokenized_scene = torch.tensor(tokenized_scene)
        tokenized_question = torch.tensor(tokenized_question)

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
        if self.relations_to_sample and (self.relations_to_sample < len(processed_relations)):
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
            objects_str_list.append(f'{object.identity_str()} {object.property_str()}')
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
