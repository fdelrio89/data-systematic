import os
from pathlib import Path
import torch

def load_checkpoint(exp_name, epoch=None):
    to_clean_int = lambda str_: ''.join(filter(str.isdigit, str_))
    get_version = lambda p: int(to_clean_int(p.stem)) if to_clean_int(p.stem) else 0

    if epoch:
        resume_from_path = str(next(Path(f'outputs/{exp_name}/').glob(f'epoch={epoch}-step=*.ckpt')))
    else:
        checkpoint_paths = sorted(Path(f'outputs/{exp_name}/').glob('last*.ckpt'), key=get_version, reverse=True)
        resume_from_path = str(checkpoint_paths[0])
    
    checkpoint = torch.load(resume_from_path)
    return checkpoint

def only_in_amd_cluster(dec):
    in_amd_cluster = lambda: os.environ.get('IS_AMD_CLUSTER')
    def decorator(func):
        if not in_amd_cluster():
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator
