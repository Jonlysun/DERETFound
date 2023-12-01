from torchvision import datasets
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import glob
import random

class DS(data.Dataset):
    def __init__(self, root, transform=None):

        self.samples = []
        self.labels = []
                           
        imgs_path_list = [            
            "EyePACS/train-FULL/train/",
            "DDR/DDR-dataset/DR_grading/train/",            
            "DDR/DDR-dataset/lesion_segmentation/train/image",
            "AIROGS",
            "ODIR-5K/odir5k/ODIR-5K/ODIR-5K/Training_Images",
                        
        ]

        for img_path in imgs_path_list:
            sample = self.get_image_labels(root + img_path)
            print(img_path, len(sample))
            self.samples.extend(sample)
        
        image_number = len(self.samples)
        print(f'Real image number: {image_number}')

        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)
        else:
            # print("Val_dataset:", val_data_name)
            print("Samples:", len(self.samples))
            
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def get_image_labels(self, imgs_path):
        samples = []
        for imgs_path, _, fnames in sorted(os.walk(imgs_path)):
            for fname in sorted(fnames):
                if '.jpg' in fname or '.png' in fname or '.jpeg' in fname or '.JPG' in fname:
                    path = os.path.join(imgs_path, fname)
                    samples.append(path)
        return samples
                
    def pad(self, im, padding=64):
        h, w = im.shape[-2:]
        mh = h % padding
        ph = 0 if mh == 0 else padding - mh
        mw = w % padding
        pw = 0 if mw == 0 else padding - mw
        shape = [s for s in im.shape]
        shape[-2] += ph
        shape[-1] += pw
        im_p = np.zeros(shape, dtype=im.dtype)
        im_p[..., :h, :w] = im
        im = im_p
        return im

    def __getitem__(self, index):

        sample_path = self.samples[index]
        sample = Image.open(sample_path).convert('RGB')
        sample_name = sample_path.split('/')[-1]
    
        if self.transform is not None:
            sample = self.transform(sample)

        # return sample
        return sample
        