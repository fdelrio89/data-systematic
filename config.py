from utils import in_amd_cluster

from argparse import ArgumentParser
import os
from pathlib import Path

import torch

# class Config:
#     def __init__(self):
#         self.base_path = '/workspace1/fidelrio/CLEVR_CoGenT_v1.0'

#         self.n_tokens = 117
#         self.n_outputs = 28
#         self.max_question_size = 45

#         self.d_hidden = 128
#         self.n_layers = 4
#         self.nhead = 4
#         self.patch_height = 32
#         self.patch_width = 48
#         self.num_patches = (320 // self.patch_height) * (480 // self.patch_width)

#         self.batch_size = 256
#         self.max_epochs = 50
#         self.lr = 1e-3


def load_config(experiment_name=""):
    if experiment_name:
        config, parser = read_args(defaults=True)
    else:
        config, parser = read_args()

    should_load_config = config.resume_training or experiment_name
    if not should_load_config:
        return config

    if not experiment_name:
        # experiment_name = os.environ.get("EXP_NAME", experiment_name)
        experiment_name = config.experiment_name

    to_clean_int = lambda str_: ''.join(filter(str.isdigit, str_))
    get_version = lambda p: int(to_clean_int(p.stem)) if to_clean_int(p.stem) else 0

    config.outputs_path = os.environ.get('OUTPUTS_PATH', config.outputs_path)
    checkpoint_paths = Path(f'{config.outputs_path}/{experiment_name}/').glob('last*.ckpt')
    checkpoint_paths = sorted(checkpoint_paths, key=get_version, reverse=True)
    checkpoint_path = str(checkpoint_paths[0])
    # checkpoint_path = f"outputs/{experiment_name}/last.ckpt"

    print(f'Loading {experiment_name} last checkpoint config from {checkpoint_path}')
    checkpoint_config = load_config_from_checkpoint(checkpoint_path)

    # Get args setted for this run
    non_default_args = get_non_default_args(parser, config)
    # override original args with new args
    for k, v in non_default_args.items():
        print(f'Updating arg: {k} = {v}')
        setattr(checkpoint_config, k, v)

    # also set new args
    for k, v in vars(config).items():
        if hasattr(checkpoint_config, k):
            continue
        print(f'Add new arg: {k} = {v}')
        setattr(checkpoint_config, k, v)

    return checkpoint_config


def read_args(defaults=False):
    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default='default')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_path", type=str, default='/workspace1/fidelrio/CLEVR_CoGenT_v1.0')
    parser.add_argument("--mixture_path", type=str, default='/workspace1/fidelrio/CLEVR_CoGenT_v1.0')
    parser.add_argument("--p_mixture", type=float, default=0.)
    parser.add_argument("--outputs_path", type=str, default='./outputs')
    parser.add_argument("--comet_experiment_key", type=str, default=None)
    parser.add_argument("--wandb_experiment_id", type=str, default=None)

    # Data
    parser.add_argument("--vocabulary_path", type=str, default='/workspace1/fidelrio/CLEVR_CoGenT_v1.0/vocab.txt')
    parser.add_argument("--pad_idx", type=int, default=1)
    parser.add_argument("--n_tokens", type=int, default=96)
    parser.add_argument("--max_scene_size", type=int, default=50)
    parser.add_argument('--not_normalize_image', action='store_true', default=False)
    parser.add_argument("--trainset_subset", type=float, default=1.)
    parser.add_argument('--permute_pixels', action='store_true', default=False)
    parser.add_argument('--color_jitter', action='store_true', default=False)
    parser.add_argument("--color_jitter_brightness", type=float, default=0.0)
    parser.add_argument("--color_jitter_hue", type=float, default=0.0)
    parser.add_argument("--color_jitter_saturation", type=float, default=0.0)
    parser.add_argument("--color_jitter_contrast", type=float, default=0.0)

    # Model
    parser.add_argument("--d_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--patch_height", type=int, default=16)
    parser.add_argument("--patch_width", type=int, default=16)

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", action='store_true', default=False)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument('--start_from', type=str, default='')

    # Operational
    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--profile', action='store_true', default=False)

    # Experimental
    parser.add_argument('--image_pretraining', action='store_true', default=False)
    parser.add_argument('--multimodal_pretraining', action='store_true', default=False)
    parser.add_argument('--multimodal_training', action='store_true', default=False)
    parser.add_argument('--aug_zero', type=int, default=1)
    parser.add_argument("--aug_zero_independent", action='store_true', default=False)
    parser.add_argument("--aug_zero_color", action='store_true', default=False)
    parser.add_argument('--token_translation_path', type=str, default='')
    parser.add_argument('--use_curriculum', action='store_true', default=False)
    parser.add_argument('--episodic_training', action='store_true', default=False)

    # Legacy
    parser.add_argument("--n_outputs", type=int, default=28)
    parser.add_argument("--rels_to_sample", type=int, default=0)
    parser.add_argument("--mp_probability", type=float, default=0.75)
    parser.add_argument("--max_question_size", type=int, default=45)
    parser.add_argument('--use_txt_scene', action='store_true', default=False)
    parser.add_argument('--not_only_front_right_relations', dest='only_front_right_relations', action='store_false', default=True)
    parser.add_argument('--dont_filter_symmetric_relations', dest='filter_symmetric_relations', action='store_false', default=True)
    parser.add_argument('--display_object_properties', action='store_true', default=False)
    parser.add_argument('--dont_shuffle_object_identities', dest='shuffle_object_identities', action='store_false', default=True)

    # Pretrained Image Embeddings
    parser.add_argument('--use_vit_embedding', action='store_true', default=False)
    parser.add_argument('--use_vit_embedding_loaded', action='store_true', default=False)
    parser.add_argument('--freeze_vit_embedding', action='store_true', default=False)
    parser.add_argument('--use_embedding_loaded', type=str, default='')
    parser.add_argument('--adapt_embedding_from', type=int, default=0)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    if is_notebook() or defaults:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    if args.use_vit_embedding or args.use_vit_embedding_loaded or args.use_embedding_loaded == 'vit':
        args.patch_height = args.patch_width = 16
        args.n_patches = (224 // args.patch_height) * (224 // args.patch_width) + 1 # Add CLS
        args.not_normalize_image = True
        args.adapt_embedding_from = 768
    if args.use_embedding_loaded == 'shrn50':
        args.patch_height = args.patch_width = 32
        args.n_patches = 49 # Add CLS
        args.not_normalize_image = True
        args.adapt_embedding_from = 2048
    elif args.image_pretraining:
        args.patch_height = args.patch_width = 16
        args.n_patches = (224 // args.patch_height) * (224 // args.patch_width) # + 1 Don't Add CLS
        args.not_normalize_image = True
    else:
        # args.n_patches = (320 // args.patch_height) * (480 // args.patch_width)
        args.patch_height = args.patch_width = 16
        args.n_patches = (224 // args.patch_height) * (224 // args.patch_width)
        args.not_normalize_image = False

    return args, parser


def get_workspace():
    import socket
    return {
        'yodaxico': 'workspace',
        'ahsoka': 'storage-otro',
    }.get(socket.gethostname(), 'workspace1')


def load_config_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['hyper_parameters']['config']
    if not in_amd_cluster():
        adapt_workspace_dir(config)
    return config


def adapt_workspace_dir(config):
    print('Adapting WORKSPACE Dir')
    # Adapt workspace to current host
    current_workspace = get_workspace()
    if 'cenia' in config.base_path:
        path_to_data_amd1 = '/work1/cenia/fidelrio/'
        path_to_data_amd2 = '/work1/cenia/dflorea/../fidelrio/'
        path_to_data_ialab = f'/{current_workspace}/fidelrio/CLEVR_CoGenT_v1.0/'
        config.base_path = config.base_path.replace(path_to_data_amd1, path_to_data_ialab)
        config.base_path = config.base_path.replace(path_to_data_amd2, path_to_data_ialab)
        config.vocabulary_path = config.vocabulary_path.replace(path_to_data_amd1, path_to_data_ialab)
        config.vocabulary_path = config.vocabulary_path.replace(path_to_data_amd2, path_to_data_ialab)
        if hasattr(config, 'mixture_path'):
            config.mixture_path = config.mixture_path.replace(path_to_data_amd1, path_to_data_ialab)
            config.mixture_path = config.mixture_path.replace(path_to_data_amd2, path_to_data_ialab)
        return

    old_workspace = config.base_path.split('/')[1]
    if old_workspace != current_workspace:
        config.base_path = config.base_path.replace(old_workspace, current_workspace)
        config.vocabulary_path = config.vocabulary_path.replace(old_workspace, current_workspace)
        if hasattr(config, 'mixture_path'):
            config.mixture_path = config.mixture_path.replace(old_workspace, current_workspace)


def get_non_default_args(parser, args):
    return {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
