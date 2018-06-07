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