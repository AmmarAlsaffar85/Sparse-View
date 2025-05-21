
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, images, masks):

        self.images = images
        self.masks = masks
        self.n_samples = len(images)

    def __getitem__(self, index):
        """ Reading image """
        
        image=self.images[index]
        image = np.expand_dims(image, axis=0) ## (1, 512, 512)
        image = image.astype(np.float32)
        #print('from data the max is =',np.max(image))
        image = torch.from_numpy(image)

        """ Reading mask """
        mask=self.masks[index]
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        #print('from data the max mask is =',np.max(mask))
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
