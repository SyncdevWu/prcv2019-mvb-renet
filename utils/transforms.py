# encoding: utf-8
from PIL import Image
from torchvision import transforms as T
from config import cfg
from utils.random_erasing import RandomErasing


class TrainTransformer(object):
    def __init__(self):
        pass

    def __call__(self, x):
        x = T.Resize((cfg.TRAIN.IMG_HEIGHT, cfg.TRAIN.IMG_WIDTH), interpolation=Image.BICUBIC)(x)
        x = T.RandomHorizontalFlip()(x)
        x = T.RandomVerticalFlip()(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        x = RandomErasing(probability=0.5, mean=[0., 0., 0.])(x)
        return x


class TestTransformer(object):
    def __init__(self):
        pass

    def __call__(self, x=None):
        x = T.Resize((cfg.TRAIN.IMG_HEIGHT, cfg.TRAIN.IMG_WIDTH))(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return x
