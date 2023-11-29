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
                #   16w
            # DR grading
            "EyePACS/train-FULL/train/",
            # "aptos2019/train_images/train_images/",
            "DDR/DDR-dataset/DR_grading/train/",
            # "messidor2/IMAGES-FULL/IMAGES/",                            
            # "ROC/ROCtraining/ROCtraining/",
            # "ROC/ROCtestImages/images",
            # "DIARETDB0/diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images",
            # "DIARETDB1/diaretdb1-v21/ddb1_v02_01/images",

            # Glaucoma
            # "REFUGE/REFUGE/train/Images",
            "AIROGS",

            # multi disease
            "ODIR-5K/odir5k/ODIR-5K/ODIR-5K/Training_Images",
            # "RFMiD/Training_Set/Training_Set/Training",
            # "JSIEC39/fundusimage1000/1000images",

            # vascular segmentation
            # "IOSTAR/IOSTAR-Vessel-Segmentation-Dataset-2018/IOSTAR Vessel Segmentation Dataset/image",

            # lesion segmentation
            # "RC-RGB-MA/RC-RGB-MA/RC-RGB-MA/Original",
            "DDR/DDR-dataset/lesion_segmentation/train/image",
            # "IDRiD/Full_TRN_nonoverlap",


            # 12w
            # DR grading
            # "aptos2019/train_images/train_images/",
            # "DDR/DDR-dataset/DR_grading/train/",
            # "messidor2/IMAGES-FULL/IMAGES/",                            
            # "ROC/ROCtraining/ROCtraining/",
            # "ROC/ROCtestImages/images",
            # "DIARETDB0/diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images",
            # "DIARETDB1/diaretdb1-v21/ddb1_v02_01/images",

            # # Glaucoma
            # "REFUGE/REFUGE/train/Images",
            # "AIROGS",

            # # multi disease
            # "ODIR-5K/odir5k/ODIR-5K/ODIR-5K/Training_Images",
            # "RFMiD/Training_Set/Training_Set/Training",
            # "JSIEC39/fundusimage1000/1000images",

            # # vascular segmentation
            # "IOSTAR/IOSTAR-Vessel-Segmentation-Dataset-2018/IOSTAR Vessel Segmentation Dataset/image",

            # # lesion segmentation
            # "RC-RGB-MA/RC-RGB-MA/RC-RGB-MA/Original",
            # "DDR/DDR-dataset/lesion_segmentation/train/image",
            # "IDRiD/Full_TRN_nonoverlap",

            # 8w
            # "AIROGS/0",
            # "AIROGS/1",
            # "AIROGS/2",
            # "AIROGS/3",
            
            # "ODIR-5K/odir5k/ODIR-5K/ODIR-5K/Training_Images",
            # "RFMiD/Training_Set/Training_Set/Training",
            # "JSIEC39/fundusimage1000/1000images",

            # # vascular segmentation
            # "IOSTAR/IOSTAR-Vessel-Segmentation-Dataset-2018/IOSTAR Vessel Segmentation Dataset/image",

            # # lesion segmentation
            # "RC-RGB-MA/RC-RGB-MA/RC-RGB-MA/Original",
            # "DDR/DDR-dataset/lesion_segmentation/train/image",
            # "IDRiD/Full_TRN_nonoverlap",

            # 4w
            # "AIROGS/0",
            # "AIROGS/1",                    
            # "ODIR-5K/odir5k/ODIR-5K/ODIR-5K/Training_Images",
            # "RFMiD/Training_Set/Training_Set/Training",
            # "JSIEC39/fundusimage1000/1000images",
            

        ]

        for img_path in imgs_path_list:
            sample = self.get_image_labels(root + img_path)
            print(img_path, len(sample))
            self.samples.extend(sample)
        
        # self.samples = random.sample(self.samples, 100000)
        image_number = len(self.samples)
        print(f'Real image number: {image_number}')

        # syn_root = '/cpfs01/projects-HDD/neikiuyiliaodamoxing_HDD/sunyuqi/SDGenerateRetinaData/'
        # imgs_path_list = [

        #     # 'SDGenerateData/SDGenerateData_1/no referable glaucoma'
        #     # 'SDGenerateData/A clear DSLR fundus photo of Proliferative Diabetic Retinopathy'
        #     # 'SDGenerateData/a DSLR photo of No Diabetic Retinopathy'

        #     # 120w
        # #    'SDGenerateData',
        # #    'SDGenerateData_1',
        # #    'SDGenerateData_2',
        # #    'SDGenerateData_3',
        # #    'SDGenerateData_4',                       
        # #    'SDGenerateData_5',
        # #    'SDGenerateDataFT',
        # #    'SDGenerateDataFT_1',        
        # #    'SDGenerateDataLoraARMD',
        # #    'SDGenerateDataLoraBRVO',
        # #    'SDGenerateDataLoraDARMD',
        # #    'SDGenerateDataLoraDN',
        # #    'SDGenerateDataLoraERM',
        # #    'SDGenerateDataLoraMH',
        # #    'SDGenerateDataLoraMyopia',
        # #    'SDGenerateDataLoraODC',
        # #    'SDGenerateDataLoraPDR',
        # #    'SDGenerateDataLoraSNPDR',
        # #    'SDGenerateDataLoraWARMD',

        #     # 921321
        #     'SDGenerateData',
        #     'SDGenerateData_1',
        #    'SDGenerateData_2',                
        #    'SDGenerateData_3',                
        #    'SDGenerateDataFT',           
        #    'SDGenerateDataLoraARMD',
        #    'SDGenerateDataLoraBRVO',
        #    'SDGenerateDataLoraDARMD',
        #    'SDGenerateDataLoraDN',
        #    'SDGenerateDataLoraERM',
        #    'SDGenerateDataLoraMH',
        #    'SDGenerateDataLoraMyopia',
        #    'SDGenerateDataLoraODC',
        #    'SDGenerateDataLoraPDR',
        #    'SDGenerateDataLoraSNPDR',
        #    'SDGenerateDataLoraWARMD',
    
        #     ## 621321
        # #    'SDGenerateData',
        # #    'SDGenerateData_1',
        # #    'SDGenerateData_2',
        # #    'SDGenerateData_3',

        #     # 30w
        #     # 'SDGenerateData',
        #     # 'SDGenerateData_2',
        # ]
        # for img_path in imgs_path_list:
        #     sample = self.get_image_labels(syn_root + img_path)
        #     self.samples.extend(sample)
        
        # self.samples = random.sample(self.samples, 100000)
        # image_number = len(self.samples)
        # print(f'Syn image number: {image_number}')

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
        