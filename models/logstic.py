import torch.nn as nn
import torch.functional as F

class Logistic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x.view(-1,784))
        return out
    
if __name__ == '__main__':
    model = Logistic(in_channel,output_dim)