from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import h5py
import cv2
import os
import ipdb
import scipy.io as io


class listDataset(Dataset):
    def __init__(self, root, shape = None, shuffle = True, transform = None,
                 train = False, seen = 0, batch_size = 1, num_workers = 4):
        if train:
            root = root * 4
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen                    # 0?
        self.batch_size = batch_size
        self.num_workers = num_workers      # 4?

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        img = Image.open(img_path).convert('RGB')       # RGB mode, (w,h)
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace(
                         'images', 'ground_truth').replace(
                         'IMG_', 'GT_IMG_')) 
        dots = np.zeros((img.size[0],img.size[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0,len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                dots[int(gt[i][1]), int(dots[i][0])] = 1
        target = gaussian_filter(dots, 15)
        print(target.shape)
        ipdb.set_trace()
        img = Image.open(img_path).convert('RGB')       # RGB mode, (w,h)
        ## pooling effect
        shape1 = int(target.shape[1] / 8.0)                         # w
        shape0 = int(target.shape[0] / 8.0)                         # h
        target = cv2.resize(target, (shape1, shape0)) * 64    # (h/8, w/8)
        if self.transform is not None:
            img = self.transform(img)               # torch.Size([3,h,w])
        return img,target
