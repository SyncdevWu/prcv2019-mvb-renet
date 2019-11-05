import logging
import re

import pandas as pd
import json
import os
import numpy as np
from os import path as osp

"""Dataset classes"""

logger = logging.getLogger('global')


class DataSet(object):

    def __init__(self, dataset_dir, root='data'):
        self.name = dataset_dir
        self.dataset_dir = osp.join(root, dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'probe')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        train_relabel = True
        train, num_train_images, num_train_bags, num_train_cams = \
            self._process_dir(self.train_dir, relabel=train_relabel)
        query, num_query_images, num_query_bags, num_query_cams = \
            self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_images, num_gallery_bags, num_gallery_cams = \
            self._process_dir(self.gallery_dir, relabel=False)

        num_total_bags = num_train_bags + num_gallery_bags
        num_total_imgs = num_train_images + num_query_images + num_gallery_images

        logger.info("=> {} loaded".format(self.name))
        logger.info("Dataset statistics:")
        logger.info("---------------------------------")
        logger.info("  subset   | # ids | # images")
        logger.info("---------------------------------")
        logger.info("  train    | {:5d} | {:8d}".format(num_train_bags, num_train_images))
        logger.info("  query    | {:5d} | {:8d}".format(num_query_bags, num_query_images))
        logger.info("  gallery  | {:5d} | {:8d}".format(num_gallery_bags, num_gallery_images))
        logger.info("---------------------------------")
        logger.info("  total    | {:5d} | {:8d}".format(num_total_bags, num_total_imgs))
        logger.info("---------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_bags = num_train_bags
        self.num_query_bags = num_query_bags
        self.num_gallery_bags = num_gallery_bags

        self.num_train_cams = num_train_cams
        self.num_query_cams = num_query_cams
        self.num_gallery_cams = num_gallery_cams

    def _check_before_run(self):
        """Check if all files are available before going deeper"""

        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):

        images = os.listdir(dir_path)
        images = sorted(images)
        bag_set = set()
        cam_set = set()
        pattern = re.compile(r'([-\d]+)_([a-z]_[-\d])')
        for image_name in images:
            # train or gallery
            if pattern.search(image_name):
                bag_id, cam_id = pattern.search(image_name).groups()
            # probe
            else:
                continue
            cam_set.add(cam_id)
            bag_set.add(bag_id)

        # sort
        cam_set = sorted(list(cam_set))
        bag_set = sorted(list(bag_set))
        if len(cam_set) != 0 and len(bag_set) != 0:
            camId2label = {cam_id: label for label, cam_id in enumerate(cam_set)}
            bagId2label = {bag_id: label for label, bag_id in enumerate(bag_set)}
        else:
            bagId2label = {}
            camId2label = {}
        dataset = []

        for image_name in images:
            image_path = osp.join(dir_path, image_name)

            # train
            if relabel:
                bag_id, cam_id = pattern.search(image_name).groups()
                bag_id = bagId2label[bag_id]
                cam_id = camId2label[cam_id]
                dataset.append((image_path, bag_id, cam_id))
            # gallery
            elif pattern.search(image_name):
                bag_id, cam_id = pattern.search(image_name).groups()
                dataset.append((image_path, bag_id, cam_id))
            # query
            else:
                img_id = int(image_name.split('.')[0])
                dataset.append((image_path, img_id, ''))

        num_images = len(dataset)
        num_bags = len(bag_set)
        num_cams = len(cam_set)

        return dataset, num_images, num_bags, num_cams


def init_dataset(name):
    if name == 'MVB':
        return DataSet('MVB')
