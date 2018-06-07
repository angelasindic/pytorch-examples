import torch


# Forward pass
def forward(x, w):
    return x * w


# Loss function
def loss(y_pred, y):
    return (y_pred - y) * (y_pred - y)


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]



# initialize w, here with 1.0, randn should also work.
w = torch.tensor([1.0])

# calculate gradients with respect to w
w.requires_grad_(True)

# Training loop
for epoch in range(30):
    for x_val, y_val in zip(x_data, y_data):
        print("\tgrad: ", x_val, y_val, w)

        y_pred = forward(x_val, w)

        l = loss(y_pred, y_val)
        l.backward()

        with torch.no_grad():
            w -= 0.01 * w.grad
            # Manually zero the gradients after updating weights
            w.grad.zero_()


    print("progress:", epoch, l)

# After training
print("predict (after training), value: ",  4, ", predicted value: ", forward(4, w))
