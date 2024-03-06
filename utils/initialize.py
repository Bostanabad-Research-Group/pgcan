
import torch
import numpy as np
import os

def initialize(seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    dtype_long = torch.cuda.LongTensor
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_default_dtype(torch.float32)
    # seed
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return dtype, device