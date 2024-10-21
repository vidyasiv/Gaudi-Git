import numpy as np
import random
import torch
SEED = 42
  
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    from habana_frameworks.torch.hpu import random as hpu_random
    torch.manual_seed(seed)
    hpu_random.manual_seed_all(seed)
