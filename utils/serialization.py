# encoding: utf-8
import errno
import os
import shutil
import sys

import os.path as osp
import torch


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, is_best, save_dir, filename):
    fpath = osp.join(save_dir, filename)
    mkdir_if_missing(save_dir)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(save_dir, 'model_best.pth.tar'))
