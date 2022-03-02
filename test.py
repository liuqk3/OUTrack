import torch
import math
import random
import bisect



def softmax(x, dim=0, epsilon=0.5, t=None):
    """
    Args:
        x: 2D tensor, performae softmax along the given dim
    """
    if t == None:
        K = x.shape[dim]
        t = math.log(K + 1) / epsilon
    x = torch.exp(t*x)
    s = torch.sum(x, dim=dim, keepdim=True)
    x = x / s

    return x, t

print(None)
# a = [1.0,2]
# a = torch.tensor(a)
# b = softmax(a, dim=-1)
# print(b)
