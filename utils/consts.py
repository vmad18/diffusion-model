from torch import nn
from torch.nn import Module
from torch import Tensor
from torch.fft import fft
from torch.nn import Softmax
import torch
import numpy as np
import torch.nn.functional as F
from scipy import linalg

true, false = True, False
null = None


def expand(x: Tensor, dims: int) -> Tensor:
    for _ in range(dims):
        x = x[..., None]
    return x
