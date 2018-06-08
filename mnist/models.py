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


class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()
        """ 
        convoultion with:
            f x f   filter/kernel size
            c_in    number of input channels
            n x n   input dimension
            c_out   number of output channels (i.e. number of filters applied)
            s       stride
            p       padding
            
        convolution of
            n x n x c_in * f x f 
        with c_out filters results in
            (n + 2p -f)/s + 1  x  (n + 2p -f)/s + 1  x  c_out
        """
        # start with 28 x 28 x 1
        self.conv_1 = nn.Conv2d(1, 5, 3)     # 26 x 26 x 5
        self.maxpool_1 = nn.MaxPool2d(2, 2)  # 13 x 13 x 5

        self.conv_2 = nn.Conv2d(5, 10, 3)    # 11 x 11 x 10
        self.maxpool_2 = nn.MaxPool2d(2, 2)  #  5 x  5 x 10

        self.fc_1 = nn.Linear(250, 10)

    def forward(self, x):
        x = F.relu(self.maxpool_1(self.conv_1(x)))
        x = F.relu(self.maxpool_2(self.conv_2(x)))

        # flatten input before the linear layers
        x = x.view(x.size(0), -1)

        x = self.fc_1(x)

        # just softmax this time.
        return F.log_softmax(x, dim=1)




