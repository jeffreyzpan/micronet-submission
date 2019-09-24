import torch
from torch.autograd import gradcheck, Variable
from quantize import Quantize

input = (
    Variable(torch.randn(20, 20).double()),
    Variable(torch.randn(1).double() + 1, requires_grad=True),
    Variable(torch.randn(1).double() - 1, requires_grad=True),
    'TTQ'
)

test = gradcheck(Quantize.apply, input, eps=1e-6, atol=1e-4)
print(test)
