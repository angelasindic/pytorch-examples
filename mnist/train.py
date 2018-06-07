import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models import LinearModel




def train(model, data_set, batch_size, epochs):

    train_loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    max_batch_idx = len(train_loader) - 1
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = linMod(data)
            loss = criterion(output, target)

            if batch_idx == max_batch_idx:
                print("epoch:", epoch, "loss: ",loss)

            # Before! backward psss, clear all gradients from steps before
            # such that new weight could be learn for this instance/epoch of training
            optimizer.zero_grad()

            # backward pass, compute gradients with repect to the given loss and model parameters
            loss.backward()

            # call step on optimizer to update its parameters
            optimizer.step()


def eval(model, test_data_set):
    test_loader = torch.utils.data.DataLoader(dataset=test_data_set)
    with torch.no_grad():
        correct = 0
        total = 0

        for image, label in test_loader:
            output = model(image)
            _, pred = torch.max(output, 1)
            if label.item() == pred.item():
                correct += 1

        return correct / len(test_loader) * 100


tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST(root='./data/',
                            train=True,
                            transform=tf,
                            download=True)

test_data = datasets.MNIST(root='./data/',
                           train=False,
                           transform=tf)
torch.manual_seed(33)

linMod = LinearModel()
batch_size = 64
epochs = 5

train(linMod, test_data, batch_size, epochs)

print("percentage correctly classified images: ", eval(linMod, test_data))