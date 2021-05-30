import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import numpy as np
import os
from PIL import Image
from music21 import *

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

        # imgs/xml list
        self.imgs_path_list = [os.path.join(root, "package", file_split, file_split + ".png") for file_split in self.split_list]
        self.xmls_path_list = [os.path.join(root, "package", file_split, file_split + ".xml") for file_split in self.split_list]

        # print data number
        print("{}: Total classfication -> {}".format(split, len(self.volcabulary_list)))
        print("{}: Total data number -> {}".format(split, len(self.imgs_path_list)))
        print("=================")

    def __getitem__(self, index):
        
        # load image
        img = Image.open(self.imgs_path_list[index])

        # load xml path
        xml_path = self.xmls_path_list[index]

        # get image file name
        name = self.split_list[index]

        # get label from xml_path -> turn into index & label_length
        label = self._mxl_to_label(xml_path)
        label = [self.volcabulary_list.index(label_index) for label_index in label]
        label_length = [len(label)]

        # Resize fixed height
        width, height = img.size
        aspect_ratio = width / height
        img = img.resize((int(self.resize_height * aspect_ratio), self.resize_height))

        # transform (add padding & ToTenser)
        if self.transform is not None:
            img = self.transform(img)

        img = torch.FloatTensor(img)
        label = torch.LongTensor(label)
        label_length = torch.LongTensor(label_length)

        return img, label, label_length, name, xml_path
    
    def __len__(self):
        return len(self.imgs_path_list)

    def classfication_num(self):
        return len(self.volcabulary_list)

    def index_to_name(self):
        return {index: name for index, name in enumerate(self.volcabulary_list)}

    def _mxl_to_label(self, xml_path):
        xml = converter.parse(xml_path)
        correct_part = xml.getElementsByClass(stream.Part)

        number_to_duration = {
            8: 'double_whole', 12: 'double_whole.', 4: 'whole', 6: 'whole.', 2: 'half', 3: 'half.', 1: 'quarter',
            1.5: 'quarter.', 1.75: 'quarter..', 0.5: 'eighth', 0.75: 'eighth.', 0.875: 'eighth..', 0.25: 'sixteenth',
            0.375: 'sixteenth.', 0.125: 'thirty_second', 0.1875: 'thirty_second.', 0.0625: 'sixty_fourth',
            0.03125: 'hundred_twenty_eighth'
        }

        circle_of_fifths = {
            0: 'CM', 1: 'GM', 2: 'DM', 3: 'AM', 4: 'EM', 5: 'BM', 6: 'F#M', 7: 'C#M', -1: 'FM', -2: 'BbM', -3: 'EbM', -4: 'AbM',
            -5: 'DbM', -6: 'GbM', -7: 'CbM'
        }

        class_list = []

        for element in correct_part.recurse():
            if isinstance(element, clef.Clef):
                class_list.append("clef-" + element.sign + str(element.line))
            elif isinstance(element, key.KeySignature):
                class_list.append("keySignature-" + circle_of_fifths[element.sharps])
            elif isinstance(element, meter.TimeSignature):
                class_list.append("timeSignature-" + element.ratioString)
            elif isinstance(element, note.Note):
                class_list.append("note-" + str(element.pitch).replace('-', 'b') + "_" + number_to_duration[element.duration.quarterLength])
            elif isinstance(element, note.Rest):
                class_list.append("rest-" + number_to_duration[element.duration.quarterLength])
            elif isinstance(element, stream.Measure):
                if element.number != 1:
                    class_list.append("barline")
        class_list.append("barline")

        return class_list

def PrIMuS_collate_fn(batch):
    img, label, label_length, name, xml_path = zip(*batch)
    img = torch.stack(img, 0)
    label = torch.cat(label, 0)
    label_length = torch.cat(label_length, 0)
    return img, label, label_length, name, xml_path

class WidthPad:
    def __call__(self, image):
        # 2003 308
        w, h = image.size
        w_fix = 2003 - w
        padding = (0, 0, w_fix, 0)
        return F.pad(image, padding, 0, 'constant')