from jittor.dataset import Dataset
from PIL import Image
import os
import numpy as np

class CelebAHQDataset(Dataset):
    def __init__(self, data_root='./data/celeba_hq_256', train=True):
        super().__init__()
        self.data_root = data_root
        self.train = train
        if self.train:
            image_list = range(27000)
            self.shuffle = True
        else:
            image_list = range(27001, 30000)
            self.shuffle = False
        self.image_list = image_list
        self.total_len = len(image_list)
        
    def __getitem__(self, index):
        path = os.path.join(self.data_root, "{:05d}.jpg".format(self.image_list[index]))
        image = Image.open(path)
        image = np.array(image, dtype=np.float32)
        image = image / 255 * (1 - (-1)) + (-1) 
        assert image.shape == (256, 256, 3)
        return image.transpose(2, 0, 1)
    
    
class Cifar10Dataset(Dataset):
    def __init__(self, data_root='./data/cifar-10', train=True):
        super().__init__()
        self.data_root = data_root
        self.train = train
        if self.train:
            image_list = range(1, 45001)
            self.shuffle = True
        else:
            image_list = range(45001, 50001)
            self.shuffle = False
        self.image_list = image_list
        self.total_len = len(image_list)
        
    def __getitem__(self, index):
        path = os.path.join(self.data_root, "{}.png".format(self.image_list[index]))
        image = Image.open(path)
        image = np.array(image, dtype=np.float32)
        image = image / 255 * (1 - (-1)) + (-1) 
        assert image.shape == (32, 32, 3)
        return image.transpose(2, 0, 1)
        
