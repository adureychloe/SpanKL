import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)  # 设置Python内置的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子

    if torch.cuda.is_available():  # 如果使用了GPU，还需要额外设置PyTorch的GPU随机种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

