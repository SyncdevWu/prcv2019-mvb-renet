import logging

import numpy as np
import torch
from config import cfg
from models.network import BagReID
from re_ranking import re_ranking as re_ranking_func
from train import build_data_loader

logger = logging.getLogger('global')

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, queryloader, galleryloader, re_ranking=False):
        self.model.eval()
        qf = []
        imgs_id = [ ]
        for inputs in queryloader:
            img, img_id, _ = self._parse_data(inputs)
            img_hflip = self.flip_horizontal(img)
            img_vflip = self.flip_vertical(img)
            img_hvflip = self.flip_vertical(img_hflip)
            feature = self._forward(img)
            feature_hflip = self._forward(img_hflip)
            feature_vflip = self._forward(img_vflip)
            feature_hvflip = self._forward(img_hvflip)
            qf.append(torch.max(feature,
                                torch.max(feature_vflip, torch.max(feature_hflip, feature_hvflip))))
            imgs_id.extend(img_id)
        qf = torch.cat(qf, 0)

        logger.info("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf = []
        g_bagids = []
        for inputs in galleryloader:
            img, bagid, _ = self._parse_data(inputs)
            img_hflip = self.flip_horizontal(img)
            img_vflip = self.flip_vertical(img)
            img_hvflip = self.flip_vertical(img_hflip)
            feature = self._forward(img)
            feature_hflip = self._forward(img_hflip)
            feature_vflip = self._forward(img_vflip)
            feature_hvflip = self._forward(img_hvflip)
            gf.append(torch.max(feature,
                                torch.max(feature_vflip, torch.max(feature_hflip, feature_hvflip))))
            g_bagids.extend(bagid)
        gf = torch.cat(gf, 0)
        logger.info("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        logger.info("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        if re_ranking:
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                       torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                       torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist, k1=5, k2=5, lambda_value=0.3))
        else:
            distmat = q_g_dist

        g_bagids = torch.Tensor(list(map(int, g_bagids)))
        imgs_id = torch.Tensor(list(map(int, imgs_id)))

        self.to_csv(distmat, imgs_id, g_bagids)


    def _parse_data(self, inputs):
        imgs, bad_ids, camids = inputs
        return imgs.cuda(), bad_ids, camids

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()

    def flip_horizontal(self, image):
        '''flip horizontal'''
        inv_idx = torch.arange(image.size(3) - 1, -1, -1, dtype=torch.int64)  # N x C x H x W
        if cfg.CUDA:
            inv_idx = inv_idx.cuda()
        img_flip = image.index_select(3, inv_idx)
        return img_flip

    def flip_vertical(self, image):
        '''flip vertical'''
        inv_idx = torch.arange(image.size(2) - 1, -1, -1, dtype=torch.int64)  # N x C x H x W
        if cfg.CUDA:
            inv_idx = inv_idx.cuda()
        img_flip = image.index_select(2, inv_idx)
        return img_flip


    def to_csv(self, distmat, imgs_id, g_bagids):
        rank = torch.argsort(distmat, dim=1)
        ret = ''
        with open(cfg.TEST.OUTPUT, 'w') as f:
            for ii, row in enumerate(rank):
                line = ''
                img_id = imgs_id[ii].item()
                img_id = '{:05d}'.format(int(img_id)) + ','
                line += img_id
                bag_set = set()
                for jj, col in enumerate(row):
                    bagid = int(g_bagids[col])
                    if bagid not in bag_set:
                        score = distmat[ii, col.long()].item()
                        line += '{:04d}'.format(bagid) + ',' + '{:.8f}'.format(score) + ','
                        bag_set.add(bagid)
                line = line[:-1] + '\n'
                ret += line
            f.write(ret)


if __name__ == '__main__':
    dataset, _, query_loader, gallery_loader = build_data_loader()
    model = BagReID(dataset.num_train_bags)
    model.cuda()
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, re_ranking=True)