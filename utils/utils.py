import random

import torch
import numpy as np
from collections.abc import MutableMapping


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True) # causes errors, cba...
    torch.backends.cudnn.deterministic = True


def flatten_dict(dictionary: dict, parent_key: str='', separator: str='_') -> dict:
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def prep_cfg_for_db(cfg: dict, to_remove: list[str]) -> dict:
    for key in to_remove:
        del cfg[key]
    return flatten_dict(cfg)
