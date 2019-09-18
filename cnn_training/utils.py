import torch
from torchvision import datasets, transforms
from torch.distributions import Categorical

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def load_data(batch_size=32, transform=None, train=True):
    
    if not transform:
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=False, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

    
def train(model, criterion, optimizer, trainloader, epochs, print_every):
    model.to(device)
    model.train()
    running_losses = []
    running_loss = .0
    for epoch in range(epochs):

        for mini_batch, (images, labels) in enumerate(trainloader, 1):

            # Flatten images into a batch_size X (width*height) dim. tensor
            #images = images.view(images.shape[0], -1)
            images = images.to(device)
            labels = labels.to(device)

            # clear gradients
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            if mini_batch % print_every == 0:
                avg_loss = running_loss/print_every
                running_losses.append(avg_loss)
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, mini_batch, avg_loss))
                running_loss = .0
    return running_losses

def train_and_eval(model, criterion, optimizer, trainloader, epochs, eval_every):    
    
    model.train()
    
    running_losses = []
    test_losses = []
    accuracies = []
    running_loss = .0
    for epoch in range(epochs):

        for mini_batch, (images, labels) in enumerate(trainloader, 1):
            
            images = images.to(device)
            labels = labels.to(device)

            # Flatten images into a batch_size X (width*height) dim. tensor
            images = images.view(images.shape[0], -1)

            # clear gradients
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            if mini_batch % eval_every == 0:
                avg_loss = running_loss/eval_every
                running_losses.append(avg_loss)
                       
                test_loss, accuracy = eval_model(model, criterion, optimizer, batch_size)
                test_losses.append(test_loss)
                accuracies.append(accuracy)
                print('Epoch: {}, Batch: {}, Avg. Loss: {}, Test Loss: {}, Acc.:{}'
                      .format(epoch + 1, mini_batch, avg_loss, test_loss, accuracy))
                model.train()
                running_loss = .0
    return running_losses, test_losses, accuracies


def eval_model(model, criterion, optimizer, testloader):
    model.eval()
    accuracy = .0
    eval_loss = .0
    with torch.no_grad():

        for mini_batch, (images,labels) in enumerate(testloader, 1):
            images = images.to(device)
            labels = labels.to(device)
            
            #forward pas to get logprobs
            probs = model(images)
            
            eval_loss += criterion(probs, labels)
            pred_probs, pred_labels = torch.max(probs, dim=1)
            equals = pred_labels == labels
            
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        l = eval_loss/len(testloader)
        a = accuracy/len(testloader)
    return l,a
    
