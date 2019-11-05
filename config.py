from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "BagReID"

__C.CUDA = True
__C.DEVICES = [0,1]
# ------------------------------------------------------------------------ #
# Model params
# ------------------------------------------------------------------------ #
__C.MODEL = CN()
__C.MODEL.GLOBAL_FEATS = 2048
__C.MODEL.PART_FEATS = 2048
# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN() 
__C.TRAIN.DATASET = 'MVB'  
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.NUM_WORKERS = 4
__C.TRAIN.NUM_IDENTITIES = 4
__C.TRAIN.NUM_EPOCHS = 150
__C.TRAIN.START_EPOCH = 0
__C.TRAIN.SNAPSHOT_DIR = './snapshot'
__C.TRAIN.IMG_HEIGHT = 300
__C.TRAIN.IMG_WIDTH = 300
__C.TRAIN.MARGIN = 2
__C.TRAIN.EVAL_STEP = 10
__C.TRAIN.PRINT_FREQ = 40
__C.TRAIN.LOG_DIR = './logs'
__C.TRAIN.OPTIM = 'adam'
__C.TRAIN.CENTER_LOSS_WEIGHT = 0.00025
# ------------------------------------------------------------------------ #
# Solver options
# ------------------------------------------------------------------------ #
__C.SOLVER = CN()
__C.SOLVER.STEPS = [40, 70]
__C.SOLVER.GAMMA = 0.1
__C.SOLVER.WARMUP_FACTOR = 0.01
__C.SOLVER.WARMUP_ITERS = 10
__C.SOLVER.WARMUP_METHOD = 'linear'
__C.SOLVER.LEARNING_RATE = 0.00035
__C.SOLVER.WEIGHT_DECAY = 5e-4
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.LEARNING_RATE_CENTER = 0.5
# ------------------------------------------------------------------------ #
# Testing options
# ------------------------------------------------------------------------ #
__C.TEST = CN()
__C.TEST.OUTPUT = '051_bag_result.csv'