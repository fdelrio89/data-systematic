from argparse import ArgumentParser


class Config:
    def __init__(self):
        self.base_path = '/workspace1/fidelrio/CLEVR_CoGenT_v1.0'

        self.n_tokens = 117
        self.n_outputs = 28
        self.max_question_size = 45

        self.d_hidden = 128
        self.n_layers = 4
        self.nhead = 4
        self.patch_height = 32
        self.patch_width = 48
        self.num_patches = (320 // self.patch_height) * (480 // self.patch_width)

        self.batch_size = 256
        self.max_epochs = 50
        self.lr = 1e-3


def load_config():
    parser = ArgumentParser()

    parser.add_argument("--base_path", type=str, default='/workspace1/fidelrio/CLEVR_CoGenT_v1.0')
    parser.add_argument("--vocabulary_path", type=str, default='/workspace1/fidelrio/CLEVR_CoGenT_v1.0/vocab.txt')
    parser.add_argument("--pad_idx", type=int, default=0)
    parser.add_argument("--n_tokens", type=int, default=95)
    parser.add_argument("--n_outputs", type=int, default=28)
    parser.add_argument("--max_question_size", type=int, default=45)
    parser.add_argument("--max_scene_size", type=int, default=259)
    parser.add_argument("--rels_to_sample", type=int, default=50)
    parser.add_argument("--d_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--patch_height", type=int, default=32)
    parser.add_argument("--patch_width", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", action='store_true', default=False)
    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--use_txt_scene', action='store_true', default=False)
    parser.add_argument('--multimodal_pretraining', action='store_true', default=False)
    parser.add_argument('--not_only_front_right_relations', dest='only_front_right_relations', action='store_false', default=True)
    parser.add_argument('--dont_filter_symmetric_relations', dest='filter_symmetric_relations', action='store_false', default=True)
    parser.add_argument('--display_object_properties', action='store_true', default=False)
    parser.add_argument('--shuffle_object_identities', action='store_true', default=False)
    parser.add_argument('--aug_zero', type=int, default=0)
    parser.add_argument('--profile', action='store_true', default=False)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    if not is_notebook():
        args = parser.parse_args()
    else:
        args = parser.parse_args([])

    args.n_patches = (320 // args.patch_height) * (480 // args.patch_width)

    return args


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
