import torch

points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
print(points.shape, points)

# transpose a tensor
arr = torch.ones(1,3,5)
print("array with ondes:\n", arr, "\nof shape:\n", arr.shape)

tp_arr = arr.transpose(0,2)
print("transposed array of shape:\n", tp_arr.shape)

print("transposed tensors are not contiguous: tp is_contiouous():", tp_arr.is_contiguous())
print("original tensors are contiguous: arr is_contiouous():", arr.is_contiguous())

torch.manual_seed(7)
ranarr = torch.randn(2,3,5)
print(ranarr)
"3 dim tensor indexing: channel,depth..., than row, col"
print(ranarr[1,1,1])

a = torch.randn(4, 4)
print("argmax of \n", a, "\n ", torch.argmax(a, 1))
