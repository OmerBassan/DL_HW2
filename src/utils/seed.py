import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Seed python, numpy and torch (CPU + CUDA). Call once at every entry point.

    Not covered: cuDNN autotuned kernels can still be nondeterministic on GPU.
    Evaluation is deterministic regardless because the eval pairs are fixed and
    the model runs in eval mode; GPU float differences are ~1e-6 on probabilities.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
