from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision

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
        
    def __len__(self):
        return self.total_len
        
    def __getitem__(self, index):
        path = os.path.join(self.data_root, "{:05d}.jpg".format(self.image_list[index]))
        image = Image.open(path)
        image = np.array(image, dtype=np.float32)
        image = ((image / 255) - 0.5) / 0.5 
        assert image.shape == (256, 256, 3)
        return image.transpose(2, 0, 1)
    
    
class Cifar10Dataset(Dataset):
    def __init__(self, data_root='./data/cifar-10', train=True):
        super().__init__()
        self.data_root = data_root
        self.train = train
        if self.train:
            self.data_root = os.path.join(self.data_root, 'train')
            image_list = range(1, 50001)
            self.shuffle = True
        else:
            self.data_root = os.path.join(self.data_root, 'test')
            image_list = range(10000)
            self.shuffle = False
        self.image_list = image_list
        self.total_len = len(image_list)
        
    def __len__(self):
        return self.total_len
        
    def __getitem__(self, index):
        path = os.path.join(self.data_root, "{}.png".format(self.image_list[index]))
        image = Image.open(path)
        image = np.array(image, dtype=np.float32)
        image = ((image / 255) - 0.5) / 0.5 
        assert image.shape == (32, 32, 3)
        return image.transpose(2, 0, 1)
        

class MnistDatasetWrapper(Dataset):
    def __init__(self, data_root, train):
        self.dataset = torchvision.datasets.MNIST(data_root, train=train, download=True, 
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5 ,), (0.5,))
                                                ]))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # only images
        return self.dataset[index][0]