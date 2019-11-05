from __future__ import absolute_import
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np
import torch
import random


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, bagid, _, ) in enumerate(data_source):
            self.index_dic[bagid].append(index)
        self.bagids = list(self.index_dic.keys())
        self.num_identities = len(self.bagids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            bagid = self.bagids[i]
            t = self.index_dic[bagid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
