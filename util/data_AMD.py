import os
import glob
from PIL import Image
import pickle
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from glob import glob
import pickle
import random
# from dataset.Mytransforms import *

# 定制数据集
class AMDDataset(Dataset):
    def __init__(self, data_dir, label_path, data_type, data_ratio=100, opt=None,  use_syn=False, syn_data_dir=None, syn_label_path=None):
        self.trainsize = (224,224)
        self.data_dir = data_dir
        self.data_type = data_type
        self.use_syn = use_syn
        self.train = True if data_type == 'train' else False

        self.data_list = []
        
        if data_ratio == 100:
            with open(label_path, "rb") as f:
                tr_dl = pickle.load(f)                
            self.data_list = tr_dl
            print('Total Real Samples:', len(self.data_list))
        else:
            dataset_name = label_path.split('/')[1]
            sample_data_path = os.path.join('SampleData', dataset_name, f'ratio_{data_ratio}', 'train.pkl')
            with open(sample_data_path, "rb") as f:
                tr_dl = pickle.load(f)                
            self.data_list = tr_dl
            print(f'Total Ratio {data_ratio} Samples: {len(self.data_list)}')
                        
        self.size = len(self.data_list)
        print('Total Samples:', self.size)
        
        self.size = len(self.data_list)
        if self.train:
            self.transform_center = transforms.Compose([
                transforms.Resize(self.trainsize),            
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(),
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        else:
            self.transform_center = transforms.Compose([
                # CropCenterSquare(),
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                
                ])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        basename, ext = data_pac['img_root'].split('.')
        imgname = data_pac['img_root']
        # if ext == 'jpg':
        #     imgname = basename + '.JPG'
        img_path = os.path.join(self.data_dir, imgname)
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        return img_torch, label

    def __len__(self):
        return self.size
    
class ISICDataset(Dataset):
    def __init__(self, data_dir, label_path, data_type, opt=None):
        self.trainsize = (224,224)
        self.data_dir = data_dir
        self.data_type = data_type
        self.train = True if data_type == 'train' else False
        with open(label_path, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        # test subset of dataset
        if data_type != 'train':
            test_image_files = os.listdir(self.data_dir)
            matching_items = []
            for item in self.data_list:
                if item['img_root'] in test_image_files:
                    matching_items.append(item)
            self.data_list = matching_items
        self.size = len(self.data_list)

        if self.train:
            self.transform_center = transforms.Compose([
                # CropCenterSquare(),
                transforms.Resize(self.trainsize),
                # RandomHorizontalFlip(),
                # RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                # CropCenterSquare(),
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = os.path.join(self.data_dir, data_pac['img_root'])
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        return img_torch, label
    
    def __len__(self):
        return self.size

class EyePACSDataset(Dataset):
    def __init__(self, data_dir, label_path, data_type, opt=None):
        self.trainsize = (224,224)
        self.data_dir = data_dir
        self.data_type = data_type
        self.train = True if data_type == 'train' else False
        with open(label_path, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl
        self.size = len(self.data_list)

        if self.train:
            self.transform_center = transforms.Compose([
                transforms.Resize(self.trainsize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(),
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Pad(16, fill=0, padding_mode='constant'),
                ])
        else:
            self.transform_center = transforms.Compose([
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Pad(16, fill=0, padding_mode='constant'),
                ])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = os.path.join(self.data_dir, data_pac['img_root'])
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        return img_torch, label
    
    def __len__(self):
        return self.size
    
class MyDataset(Dataset):
    def __init__(self, data_dir, label_path, transforms, data_type="train", debug=True, opt = None):
        self.data_dir = data_dir
        self.label_path = label_path
        # df = pd.read_csv(self.label_path)
        # self.img_list = df[df.columns[0]].values
        # self.label_list = df[df.columns[1]].values
        self.transforms = transforms
        self.imgs = sorted(glob.glob(os.path.join(self.data_dir, "*.*")))
        
        if debug:
            self.imgs = self.imgs[:10]
            
        self.length = len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img_name = os.path.split(img_path)[-1]
        pil_img = Image.open(img_path).convert("RGB")
        img = self.transforms(pil_img)
        return img, img_name

    def __len__(self):
        return self.length

def get_loaders(opt):
    DatasetClass = eval(opt.datasetM)
    num_train_sets = len(opt.train_sets)
    train_dataset_list = []
    for i in range(num_train_sets):
        train_dataset = DatasetClass(data_dir = opt.TRAIN_DATA_DIR[opt.train_sets[i]],
                            label_path = opt.PATH_TO_TRAIN_LABEL[opt.train_sets[i]],
                            data_type  = 'train',
                            debug      = opt.debug,
                            opt = opt)
        train_dataset_list.append(train_dataset)
    train_dataset = ConcatDataset(train_dataset_list)
    test_dataset = DatasetClass(data_dir = opt.TEST_DATA_DIR[opt.test_sets[0]],
                            label_path = opt.PATH_TO_VAL_LABEL[opt.test_sets[0]],
                            data_type  = 'test',
                            debug      = opt.debug,
                            opt = opt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )
    return train_loader, test_loader

## for five-fold cross-validation on Train&Val, return Train&Val loaders
# def get_loaders(opt):
#     DatasetClass = eval(opt.datasetM)
#     train_dataset = DatasetClass(data_dir = opt.DATA_DIR,
#                                  label_path = opt.PATH_TO_LABEL[opt.train_dataset],
#                                  data_type  = 'train',
#                                  debug      = opt.debug,
#                                  opt = opt)

#     # gain indices for cross-validation
#     whole_folder = []
#     whole_num = len(train_dataset)
#     indices = np.arange(whole_num)
#     random.seed(opt.ds_seed)
#     random.shuffle(indices)

#     # split indices into five-fold
#     num_folder = opt.num_folder
#     each_folder_num = int(whole_num / num_folder)
#     for ii in range(num_folder-1):
#         each_folder = indices[each_folder_num*ii: each_folder_num*(ii+1)]
#         whole_folder.append(each_folder)
#     each_folder = indices[each_folder_num*(num_folder-1):]
#     whole_folder.append(each_folder)
#     assert len(whole_folder) == num_folder
#     assert sum([len(each) for each in whole_folder if 1==1]) == whole_num

#     ## split into train/eval
#     train_eval_idxs = []
#     for ii in range(num_folder):
#         eval_idxs = whole_folder[ii]
#         train_idxs = []
#         for jj in range(num_folder):
#             if jj != ii: train_idxs.extend(whole_folder[jj])
#         train_eval_idxs.append([train_idxs, eval_idxs])

#     ## gain train and eval loaders
#     train_loaders = []
#     eval_loaders = []
#     for ii in range(len(train_eval_idxs)):
#         train_idxs = train_eval_idxs[ii][0]
#         eval_idxs  = train_eval_idxs[ii][1]
#         train_loader = DataLoader(train_dataset,
#                                   batch_size=opt.batch_size,
#                                   sampler=SubsetRandomSampler(train_idxs),
#                                   num_workers=opt.num_workers,
#                                   pin_memory=True)
#         eval_loader = DataLoader(train_dataset,
#                                  batch_size=opt.batch_size,
#                                  sampler=SubsetRandomSampler(eval_idxs),
#                                  num_workers=opt.num_workers,
#                                  pin_memory=True)
#         train_loaders.append(train_loader)
#         eval_loaders.append(eval_loader)

#     return train_loaders, eval_loaders

def get_test_loaders(opt):
    test_loaders = []
    if opt.havetest_sets:
        for test_set in opt.test_sets:
            DatasetClass = eval(test_set)
            test_dataset = DatasetClass(data_dir = opt.TEST_DATA_DIR[test_set],
                                 label_path = opt.PATH_TO_TEST_LABEL[test_set],
                                 data_type  = test_set,
                                 debug      = opt.debug)
            
            test_loader = DataLoader(test_dataset,
                                    batch_size=opt.batch_size,
                                    num_workers=opt.num_workers,
                                    shuffle=False,
                                    pin_memory=False)
            test_loaders.append(test_loader)
    return test_loaders

if __name__ == '__main__':
    # train_dataset = APTOSDataset(data_dir = "/mnt/gzy/DiffMed/APTOS/train_images",
                                #  label_path = "/mnt/gzy/DiffMed/APTOS/aptos_test.pkl",
                                #  data_type  = 'train',
                                #  debug      = False)
    train_dataset = ISICDataset(data_dir = "/mnt/gzy/DiffMed/ISIC/rec_subset",
                                 label_path = "/mnt/gzy/DiffMed/ISIC/isic2018_test.pkl",
                                 data_type  = 'test',
                                 debug      = False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8,
    )
    print(len(train_dataset))
    # for batch in train_loader:
    #     img, label = batch
        # print(img.shape)
        # print(label.shape)