import os
import time
import torch
import logging
import shutil

from utils.load_data import build_data_loader
from utils.lr_scheduler import WarmupMultiStepLR
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from config import cfg
from loss.loss import TripletLoss, CrossEntropyLabelSmooth, CenterLoss
from models.network import BagReID_SE_RESNEXT
from utils.log_helper import init_log, add_file_handler
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint

logger = logging.getLogger('global')

xent_criterion = None
triplet_criterion = None
center_criterion = None


def criterion(logits, features, ids):
    global xent_criterion, triplet_criterion, center_criterion
    xcent_loss = sum([xent_criterion(output, ids) for output in logits])
    triplet_loss = sum([triplet_criterion(output, ids)[0] for output in features])
    center_loss = center_criterion(torch.cat(features, dim=1), ids)
    loss = xcent_loss + triplet_loss + cfg.TRAIN.CENTER_LOSS_WEIGHT * center_loss
    return loss


def train(epoch, train_loader, model, criterion, optimizers, summary_writer):
    global center_criterion
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR):
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    # start training
    model.train()
    start = time.time()
    for ii, datas in enumerate(train_loader):
        data_time.update(time.time() - start)
        img, bag_id, cam_id = datas
        if cfg.CUDA:
            img = img.cuda()
            bag_id = bag_id.cuda()

        triplet_features, softmax_features = model(img)

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss = criterion(softmax_features, triplet_features, bag_id)
        loss.backward()

        for param in center_criterion.parameters():
            param.grad.data *= (1. / cfg.TRAIN.CENTER_LOSS_WEIGHT)

        for optimizer in optimizers:
            optimizer.step()

        batch_time.update(time.time() - start)
        losses.update(loss.item())
        # tensorboard
        if summary_writer:
            global_step = epoch * len(train_loader) + ii
            summary_writer.add_scalar('loss', loss.item(), global_step)

        start = time.time()

        if (ii + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            logger.info('Epoch: [{}][{}/{}]\t'
                        'Batch Time {:.3f} ({:.3f})\t'
                        'Data Time {:.3f} ({:.3f})\t'
                        'Loss {:.3f} ({:.3f}) \t'
                        .format(epoch + 1, ii + 1, len(train_loader),
                                batch_time.val, batch_time.mean,
                                data_time.val, data_time.mean,
                                losses.val, losses.mean))
    adam_param_groups = optimizers[0].param_groups
    logger.info('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
                'Adam Lr {:.2e} \t '
                .format(epoch + 1, batch_time.sum, losses.mean,
                        adam_param_groups[0]['lr']))


def build_lr_schedulers(optimizers):
    schedulers = []
    for optimizer in optimizers:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS,
                                      cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS,
                                      cfg.SOLVER.WARMUP_METHOD)
        schedulers.append(scheduler)
    return schedulers


def main():
    global xent_criterion, triplet_criterion
    global center_criterion

    logger.info("init done")

    if os.path.exists(cfg.TRAIN.LOG_DIR):
        shutil.rmtree(cfg.TRAIN.LOG_DIR)
    os.makedirs(cfg.TRAIN.LOG_DIR)
    init_log('global', logging.INFO)
    if cfg.TRAIN.LOG_DIR:
        add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'), logging.INFO)

    dataset, train_loader, _, _ = build_data_loader()
    model = BagReID_SE_RESNEXT(dataset.num_train_bags)
    xent_criterion = CrossEntropyLabelSmooth(dataset.num_train_bags)
    triplet_criterion = TripletLoss(margin=cfg.TRAIN.MARGIN)
    center_criterion = CenterLoss(dataset.num_train_bags,
                                  cfg.MODEL.GLOBAL_FEATS + cfg.MODEL.PART_FEATS)
    if cfg.TRAIN.OPTIM == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.SOLVER.LEARNING_RATE,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.SOLVER.LEARNING_RATE,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    center_optimizer = torch.optim.SGD(center_criterion.parameters(),
                                       lr=cfg.SOLVER.LEARNING_RATE_CENTER)

    optimizers = [optimizer]
    schedulers = build_lr_schedulers(optimizers)

    optimizers += [center_optimizer]

    if cfg.CUDA:
        model.cuda()
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)

    if cfg.TRAIN.LOG_DIR:
        summary_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        summary_writer = None

    logger.info("model prepare done")
    start_epoch = cfg.TRAIN.START_EPOCH
    # start training
    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        train(epoch, train_loader, model, criterion, optimizers, summary_writer)
        for scheduler in schedulers:
            scheduler.step()
        # skip if not save model
        if cfg.TRAIN.EVAL_STEP > 0 and (epoch + 1) % cfg.TRAIN.EVAL_STEP == 0 \
                or (epoch + 1) == cfg.TRAIN.NUM_EPOCHS:

            if cfg.CUDA and torch.cuda.device_count() > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1},
                            is_best=False, save_dir=cfg.TRAIN.SNAPSHOT_DIR,
                            filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')


if __name__ == '__main__':
    main()
