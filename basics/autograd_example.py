import torch

x = torch.tensor([1.])
x.requires_grad_(True)
print(x)

y = 10 * x * x + 2
print(y)
print(y.grad_fn)


print(x, y)

y.backward()
print(x.grad)