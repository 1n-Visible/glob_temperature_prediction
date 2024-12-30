from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import torch

import netCDF4
import numpy as np
import os

__all__=[
    "CaptchasDataSet", "DataLoader"
]

'''
def load_temperature(filename) -> np.array:
    file=netCDF4.Dataset(filename, "rs") # 's' for unbuffered
    print(file.groups, file.dimensions)
    file.close()
    return None
'''

class ClimateDataset(Dataset):
    def __init__(self, folder):
        self.folder=folder
        self.is_iterable=False
    
    def __iter__(self):
        self.is_iterable=True
        return self
    
    def __next__(self):
        if not self.is_iterable:
            raise TypeError(f"{type(self).__name__!r} object is not an iterator")
        
        return NotImplemented


class CaptchasDataSet(Dataset):
    def __init__(self, folder):
        self.folder=folder
        self.raw_image_labels=[
            (read_image(os.path.join(self.folder, filename))[0],
             filename.removesuffix(".jpeg").split("_")[-1])
            for filename in os.listdir(self.folder)
        ]
        
        flatten_func=torch.nn.Flatten(start_dim = 0)
        self.data=[
            (torch.stack(tuple(map(flatten_func, image.split(50, dim=1)))),
             torch.Tensor(list(map(ord, label))))
            for image, label in self.raw_image_labels
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


def main():
    #load_temperature("data/data.nc")
    dataset=CaptchasDataSet("data/captchas/test")
    print(f"{len(dataset)=}")
    img300=dataset[300][0]
    print(img300.shape, img300.split(50, dim=2)[1].shape)
    return

if __name__=="__main__":
    main()
