import os
from pathlib import Path
import torch


def load_checkpoint(exp_name, epoch=None):
    to_clean_int = lambda str_: ''.join(filter(str.isdigit, str_))
    get_version = lambda p: int(to_clean_int(p.stem)) if to_clean_int(p.stem) else 0

    outputs_dir =  os.environ.get('OUTPUTS_PATH', './outputs')
    if epoch:
        resume_from_path = str(next(Path(f'{outputs_dir}/{exp_name}/').glob(f'epoch={epoch}-step=*.ckpt')))
    else:
        checkpoint_paths = sorted(Path(f'{outputs_dir}/{exp_name}/').glob('last*.ckpt'), key=get_version, reverse=True)
        resume_from_path = str(checkpoint_paths[0])

    checkpoint = torch.load(resume_from_path)
    return checkpoint
