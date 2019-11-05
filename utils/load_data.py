import logging
from datasets.data_loader import ImageData
from datasets.data_manager import init_dataset
from datasets.samplers import RandomIdentitySampler
from torch.utils.data import DataLoader
from utils.transforms import TrainTransformer, TestTransformer
from config import cfg
logger = logging.getLogger('global')

def build_data_loader():
    logger.info("build train dataset")
    # dataset
    dataset = init_dataset(cfg.TRAIN.DATASET)
    sampler = RandomIdentitySampler(dataset.train, cfg.TRAIN.NUM_IDENTITIES)
    train_loader = DataLoader(ImageData(dataset.train, TrainTransformer()),
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=sampler)

    query_loader = DataLoader(ImageData(dataset.query, TestTransformer()),
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              shuffle=False)

    gallery_loader = DataLoader(ImageData(dataset.gallery, TestTransformer()),
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                num_workers=cfg.TRAIN.NUM_WORKERS,
                                pin_memory=True,
                                shuffle=False)
    return dataset, train_loader, query_loader, gallery_loader