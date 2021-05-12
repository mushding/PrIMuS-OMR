import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import numpy as np
import os
from PIL import Image

class PrIMuS_PredictDataset(Dataset):
    def __init__(self, root, split, type, resize_height, transform):
        
        # set variables
        self.resize_height = resize_height
        self.transform = transform    

        # set predict image file path (all in the package dir)
        self.split_path = os.path.join(root, "package")
        self.split_list = os.listdir(self.split_path)

        # set volcabulary (semantic/agnostic) list
        self.volcabulary_path = os.path.join(root, "vocabulary_" + type + ".txt")
        volcabulary_file = open(self.volcabulary_path, "r")
        self.volcabulary_list = [line.rstrip('\n') for line in volcabulary_file]
        volcabulary_file.close()

        # imgs/label list
        self.imgs_path_list = [os.path.join(root, "package", file_split) for file_split in self.split_list]

        # print data number
        print("{}: Total classfication -> {}".format(split, len(self.volcabulary_list)))
        print("{}: Total data number -> {}".format(split, len(self.imgs_path_list)))
        print("=================")

    def __getitem__(self, index):
        
        # load image
        img = Image.open(self.imgs_path_list[index])

        # get image file name
        name = self.split_list[index]

        # Resize fixed height
        width, height = img.size
        aspect_ratio = width / height
        img = img.resize((int(self.resize_height * aspect_ratio), self.resize_height))

        # transform (add padding & ToTenser)
        if self.transform is not None:
            img = self.transform(img)

        img = torch.FloatTensor(img)

        return img, name
    
    def __len__(self):
        return len(self.imgs_path_list)

    def classfication_num(self):
        return len(self.volcabulary_list)

    def index_to_name(self):
        return {index: name for index, name in enumerate(self.volcabulary_list)}

def PrIMuS_collate_fn(batch):
    imgs, names = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, names

class WidthPad:
    def __call__(self, image):
        # 2003 308
        w, h = image.size
        w_fix = 2003 - w
        padding = (0, 0, w_fix, 0)
        return F.pad(image, padding, 0, 'constant')