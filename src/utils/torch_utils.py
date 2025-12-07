import torch
import torch.nn as nn
import torch.nn.functional as F


class _ScaleGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def scale_gradient(x, scale):
    return _ScaleGradient.apply(x, scale)


class _PrintGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x
    
    @staticmethod
    def backward(ctx, grad_output):

        if ctx.name is not None:
            print(f"Gradient of {ctx.name}:")
        else:
            print("Gradient:")
        print(grad_output)
        
        return grad_output, None

def print_gradient(x, name=None):
    return _PrintGradient.apply(x, name)


def unsqueeze_to_batch(x, target):
    while x.dim() < target.dim():
        x = x[None]

    return x


def expand_to_batch(x, target):
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[None]
        num_unsqueeze += 1

    x = x.expand(
        *([target.shape[i] for i in range(num_unsqueeze)] + list(og_shape))
    )

    return x
