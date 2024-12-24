import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder
#import timm # unneccesary shit

class TemperatureModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.flatten = nn.Flatten() #
        self.lstm = nn.LSTM()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    temp_model = TemperatureModel()
    temp_model.to(device)
    
    return 0

if __name__=="__main__":
    main()
