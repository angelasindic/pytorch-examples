import torch

x = torch.Tensor(3,5)
#print(x)
print(torch.cuda.is_available())
print(torch.version.__version__)
