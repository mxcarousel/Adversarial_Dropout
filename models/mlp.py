
import torch.nn as nn
import torch.nn.functional as F 

class mlp(nn.Module):
    """Neural Networks"""
    def __init__(self, input_size, hidden_size, output_size):
        super(mlp , self).__init__()
        # layer 1 
        self.layer1 = nn.Linear(input_size, hidden_size, bias=True)
        # layer 2
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True)
    
    # forward pass 
    def forward(self, x):
        x = x.view(x.size(0),-1)
        y_hidden = self.layer1(x)        
        y = self.layer2(F.relu(y_hidden))
        
        return y