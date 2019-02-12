import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self):

        super(LinearModel, self).__init__()

        self.linear_1 = nn.Linear(784, 350)
        self.linear_2 = nn.Linear(350, 10)


    def forward(self, x):
        # flatten input before the linear layers
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))

        x = F.log_softmax(x, dim=1)

        return x

    
    
"""
============ CNN =========================

convoultion with:
    f x f   filter/kernel size
    c_in    number of input channels
    n x n   input dimension
    c_out   number of output channels (number of filters applied)
    s       stride
    p       padding

convolution of
    n x n x c_in  

with f x f kernel

and c_out output channels 

results in
    (n + 2p -f)/s + 1  x  (n + 2p -f)/s + 1  x  c_out
"""

class DeepCNN(nn.Module):

    def __init__(self):

        super(DeepCNN, self).__init__()
       
        # start with 28 x 28 x 1
        self.conv_11 = nn.Conv2d(1, 32, 3)     # 26 x 26 x 32
        self.conv_12 = nn.Conv2d(32, 32, 3)    # 24 x 24 x 32
        self.maxpool_1 = nn.MaxPool2d(2, 2)   # 12 x 12 x 32

        self.conv_21 = nn.Conv2d(32, 64, 3)    # 10 x 10 x 64
        self.conv_22 = nn.Conv2d(64, 64, 3)    #  8 x  8 x 64
        self.maxpool_2 = nn.MaxPool2d(2, 2)   #  4 x  4 x 64

        self.fc_1 = nn.Linear(16*64, 512)
        self.fc_2 = nn.Linear(512, 10)


    def forward(self, x):
        #x initially has batch_size x c_in x n x n
        # fashion mnists has one channel and input dim. is 28 x 28
        
        x = F.relu(self.maxpool_1(self.conv_12(self.conv_11(x))))
        x = F.relu(self.maxpool_2(self.conv_22(self.conv_21(x))))

        # flatten input before the linear layers
        x = x.view(x.size(0), -1)

        x = self.fc_1(x)

        return F.log_softmax(self.fc_2(x), dim=1)


class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()
       
        # start with 28 x 28 x 1
        self.conv_1 = nn.Conv2d(1, 5, 3)     # 26 x 26 x 5
        self.maxpool_1 = nn.MaxPool2d(2, 2)  # 13 x 13 x 5

        self.conv_2 = nn.Conv2d(5, 10, 3)    # 11 x 11 x 10
        self.maxpool_2 = nn.MaxPool2d(2, 2)  #  5 x  5 x 10

        self.fc_1 = nn.Linear(250, 10)

    def forward(self, x):
        #x initially has batch_size x c_in x n x n
        # fashion mnists has one channel and input dim. is 28 x 28
        
        x = F.relu(self.maxpool_1(self.conv_1(x)))
        x = F.relu(self.maxpool_2(self.conv_2(x)))

        # flatten input before the linear layers
        x = x.view(x.size(0), -1)

        x = self.fc_1(x)

        return F.log_softmax(x, dim=1)


