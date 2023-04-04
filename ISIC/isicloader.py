import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader, random_split

class ISICDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]+'.jpg'
        img_path = os.path.join(self.data_path, 'ISBI2016_ISIC_Part3B_'+ self.mode +'_Data',name)
        
        mask_name = name.split('.')[0] + '_Segmentation.png'
        msk_path = os.path.join(self.data_path, 'ISBI2016_ISIC_Part3B_'+ self.mode +'_Data',mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        if self.mode == 'Training':
            label = 0 if self.label_list[index] == 'benign' else 1
        else:
            label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        if self.mode == 'Training':
            return (img, mask)
        else:
            return (img, mask, name)

data_path = "./"
tran_list = [transforms.Resize((256, 256)), transforms.ToTensor(), ]
transform_train = transforms.Compose(tran_list)
ds = ISICDataset(None, data_path, transform_train)

train_loader = DataLoader(ds, batch_size=8, shuffle=True)

for index, (img, label) in enumerate(train_loader):
    print(img.shape)
    print(label.shape)
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow((img[0, :, :, :].moveaxis(0, 2)))
    plt.subplot(222)
    plt.imshow(label[0,:,: ].squeeze(0))

    plt.subplot(223)
    plt.imshow((img[6, :, :, :].moveaxis(0, 2)))
    plt.subplot(224)
    plt.imshow(label[6,:,: ].squeeze(0))
    plt.show()
    if index == 0:
        break

