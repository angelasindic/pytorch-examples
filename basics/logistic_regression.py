import torch
import torch.nn.functional as F

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, dim_in):
        """
        In the constructor all layeres are created that are used later on in the forward pass.
        """
        super(LogisticRegressionModel, self).__init__()

        self.linear = torch.nn.Linear(dim_in, 1)


    def forward(self, x):
        """
        :param x: input data
        :return: performs logistic regression of the input data
        """
        return F.sigmoid(self.linear(x))


def train(data, labels, model, epochs):
    len = data.shape[1]
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):

        y_pred = model(data)

        loss = criterion(y_pred, labels)
        print("epoch:" , epoch, "loss: ",loss.item())

        # Before! backward pass, clear all gradients from steps before
        # such that new weight could be learn for this instance/epoch of training
        optimizer.zero_grad()

        # backward pass, compute gradients with repect to the given loss and model parameters
        loss.backward()

        # call step on optimizer to update its parameters
        optimizer.step()

"""
train the model and make an prediction
"""

torch.manual_seed(33)

"""
model = LogisticRegressionModel(1)

print(model(torch.Tensor([[1.0]])))
input = torch.Tensor([[1.] [1.], [2.], [3.], [10.], [20.]])
labels = torch.Tensor([[1], [1], [1], [0], [0]])

trained_model = train(input, labels, model, 100)
print(model(torch.Tensor([[1.0]])))
"""


model = LogisticRegressionModel(2)

print(model)

new_value = torch.Tensor([[40., 1.]])

print("Prediction before training: ", model(new_value).item())

input = torch.Tensor([[1., 1.],
                      [2., 2.],
                      [3., 3.],
                      [1., 10],
                      [1., 2.],
                      [100., 100.],
                      [50., 0.]])

labels = torch.Tensor([[1], [1], [1], [0], [0], [1], [0]])

#input = torch.randn(3, 2)

train(input, labels, model, 1000)

print("Prediction after training (should be 0): ", model(new_value).item())
