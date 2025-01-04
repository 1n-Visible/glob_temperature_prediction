import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder
#import timm # unnecessary

#import data_loader
from data_loader import *

__all__=[
    "TemperatureModel", "main"
]

class TemperatureModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout=0.0, dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        
        self.num_layers=num_layers
        self.rnn_dropout=dropout
        
        self.rnn = nn.LSTM(
            self.input_size, self.hidden_size, num_layers=self.num_layers,
            dropout=self.rnn_dropout, batch_first=True
        ) ## batch_first=False (default)!
        self.linear=nn.Linear(self.hidden_size, self.output_size)
        
        self.dataset=dataset
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()
    
    def forward(self, x, h0=None, c0=None):
        x=x.to(self.device)
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        output, (h1, c1) = self.rnn(x, (h0, c0))
        #print(f"{output.shape = }")
        output = self.linear(output[-1]) # Selecting the last output of LSTM
        return (output, h1, c1)
    
    def train_epoch(self):
        self.train(mode=True)
        loss=0.0
        for data in self.dataset:
            loss+=self.train_sample(*data)
        return loss/len(self.dataset)
    
    def train_sample(self, train_input, train_output):
        h0, c0 = None, None # Initialize hidden and cell states
        #print(f"{train_input.shape=}")
        self.optimizer.zero_grad()
        # Forward pass
        outputs, h0, c0 = self(train_input.type(torch.float32), h0, c0)
        
        # Compute loss
        loss = self.loss_func(outputs, train_output)
        loss.backward()
        self.optimizer.step()
        
        # Detach (copy) hidden and cell states
        # to prevent backpropagation through the entire sequence
        #h0 = h0.detach()
        #c0 = c0.detach()
        return loss
    
    def test_epoch(self):
        return 0.0


def main():
    # TODO: force the model to run on GPU!!!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    '''temp_model = TemperatureModel(100, 200, 2, 50, dropout=0.005, device=device)
    temp_model.train_epoch()'''
    batch_size = 64
    
    captchas_set=CaptchasDataSet("data/captchas/train")
    datasize=len(captchas_set)
    # torch object to load data in batches:
    #dataloader=DataLoader(captchas_set, batch_size=batch_size)
    
    temp_model=TemperatureModel(
        50*50, 40*40, num_layers=2, output_size=6,
        dataset=captchas_set
    ).to(device)
    
    for epoch in range(1, 101):
        loss=temp_model.train_epoch()
        if epoch%10 == 0:
            print(f"Epoch #{epoch}: loss={loss.item():.3f}")
        #temp_model.test_epoch()
    
    return 0

if __name__=="__main__":
    main()
