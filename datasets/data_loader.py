import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                img_path))
            pass
    return img


class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        image_path, bag_id, cam_id = self.dataset[item]
        img = read_image(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, bag_id, cam_id

    def __len__(self):
        return len(self.dataset)
