import math
import os

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.engine import default_setup


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.RESUME = args.resume
    cfg.OUTPUT_DIR = os.path.join(
        cfg.OUTPUT_DIR, os.path.split(args.config_file)[-1].replace(".yaml", "")
    )

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_inference_iters_id(checkpoint):
    if checkpoint != "final":
        checkpoint = math.ceil(int(checkpoint) / 1000)
        checkpoint = f"{checkpoint}k"
    return checkpoint


def add_config(cfg):

    # only for older parq models
    cfg.MODEL.ENCODER_TYPE = "UNET"
    cfg.MODEL.USE_VESSEL_INFO = "no"
    cfg.MODEL.D_MODEL = 384
    cfg.MODEL.EVAL_VIZ_MODE = False

    cfg.MODEL.CONV_MODEL = CN()
    cfg.MODEL.CONV_MODEL.BACKBONE_TYPE = "CNN"  # for newer deform models
    cfg.MODEL.CONV_MODEL.N_BLOCKS = [2, 3, 3, 3]
    cfg.MODEL.CONV_MODEL.N_FILTERS = [64, 96, 128, 256]
    cfg.MODEL.CONV_MODEL.STEM_FILTERS = 32
    cfg.MODEL.CONV_MODEL.NORM = "BN"
    cfg.MODEL.CONV_MODEL.HEAD_NORM = "BN"
    cfg.MODEL.CONV_MODEL.ACT = "ReLU"
    cfg.MODEL.CONV_MODEL.SE = False
    cfg.MODEL.CONV_MODEL.FIRST_STRIDE = (2, 2, 2)

    cfg.MODEL.DEFORMABLE = CN()
    cfg.MODEL.DEFORMABLE.N_LEVELS = 4
    cfg.MODEL.DEFORMABLE.N_HEADS = 4
    cfg.MODEL.DEFORMABLE.N_ENC_LAYERS = 2
    cfg.MODEL.DEFORMABLE.N_DEC_LAYERS = 2

    cfg.MODEL.DEFORMABLE.FFN_DIM = 512
    cfg.MODEL.DEFORMABLE.DROPOUT = 0.1
    cfg.MODEL.DEFORMABLE.ACTIVATION = "RELU"
    cfg.MODEL.DEFORMABLE.N_ENC_POINTS = 4
    cfg.MODEL.DEFORMABLE.N_DEC_POINTS = 32
    cfg.MODEL.DEFORMABLE.DECODER_ONLY = False
    cfg.MODEL.DEFORMABLE.WITH_RECURRENCE = False
    cfg.MODEL.DEFORMABLE.WITH_STEPWISE_LOSS = False
    cfg.MODEL.DEFORMABLE.SHARED_CENTER_HEAD = True
    cfg.MODEL.DEFORMABLE.HEAD_DROPOUT = 0.1
    cfg.MODEL.DEFORMABLE.SHARED_DECODER_LAYER_WEIGHTS = False
    cfg.MODEL.DEFORMABLE.OFFSET_INIT = "strict"
    cfg.MODEL.DEFORMABLE.USE_GLOBAL_PE = False
    cfg.MODEL.DEFORMABLE.USE_VESSEL_INFO = False
    cfg.MODEL.DEFORMABLE.FIXED_ATTENTION = False

    # the ratio of the original image size to the parq volume size
    # this value must be adjusted when the stride and conv filter size change
    cfg.MODEL.CONV_MODEL.POST_UNET_SCALE_RATIO = [1 / 4, 1 / 4, 1 / 4]
    cfg.MODEL.CONV_MODEL.USE_PRETRAINED_UNET_ENCODER = False
    cfg.MODEL.CONV_MODEL.PRETRAINED_UNET_ENCODER_PATH = ""
    cfg.MODEL.CONV_MODEL.FROZEN_PRETRAINED_ENCODER = False
    cfg.MODEL.CONV_MODEL.RESIZE_PATCH_PROJECTION = False
    cfg.MODEL.CONV_MODEL.RESIZE_PATCH_FACTOR = 2
    # TODO: add config for whether to finetune these

    # parq parameters
    cfg.MODEL.PARQ_MODEL = CN()
    cfg.MODEL.PARQ_MODEL.EMBED_DIM = 1024
    cfg.MODEL.PARQ_MODEL.PATCH_SIZE_3D = (2, 2, 2)
    cfg.MODEL.PARQ_MODEL.NUM_QUERIES = 8

    cfg.MODEL.PARQ_MODEL.DECODER = CN()
    cfg.MODEL.PARQ_MODEL.DECODER.DEC_DIM = 1024
    cfg.MODEL.PARQ_MODEL.DECODER.DEC_HEADS = 4
    cfg.MODEL.PARQ_MODEL.DECODER.DEC_FFN_DIM = 768
    cfg.MODEL.PARQ_MODEL.DECODER.DEC_LAYERS = 2
    cfg.MODEL.PARQ_MODEL.DECODER.DROPOUT_RATE = 0.1
    cfg.MODEL.PARQ_MODEL.DECODER.QUERIES_DIM = 1024
    cfg.MODEL.PARQ_MODEL.DECODER.SHARE_WEIGHTS = False
    cfg.MODEL.PARQ_MODEL.DECODER.USE_POSITIONAL_ENCODING = "use_pe"

    cfg.MODEL.PARQ_MODEL.HEADS = CN()
    cfg.MODEL.PARQ_MODEL.HEADS.SHARED = False
    cfg.MODEL.PARQ_MODEL.HEADS.MEAN_SIZE_PATH = None

    cfg.MODEL.PARQ_MODEL.PARQ_LOSS = CN()
    cfg.MODEL.PARQ_MODEL.PARQ_LOSS.CLS_W = 4.0
    cfg.MODEL.PARQ_MODEL.PARQ_LOSS.SHAPE_W = 0.1
    cfg.MODEL.PARQ_MODEL.PARQ_LOSS.OFFSET_W = 1.0
    cfg.MODEL.PARQ_MODEL.PARQ_LOSS.IOU_W = 1.0

    cfg.MODEL.PARQ_MODEL.PARQ_LOSS.DO_CLF_FOCAL = False
    cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_GAMMA = 2.0
    cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_ALPHA = 0.75

    cfg.MODEL.CONV_MODEL.DET_LOSS = CN()
    # number of matching anchors for each annotations
    cfg.MODEL.CONV_MODEL.DET_LOSS.TOPK = 7
    # number of ignored anchors is the next topk*ignore_ration anchors
    cfg.MODEL.CONV_MODEL.DET_LOSS.IGNORE_RATIO = 26
    cfg.MODEL.CONV_MODEL.DET_LOSS.FOCAL_GAMMA = 2.0
    cfg.MODEL.CONV_MODEL.DET_LOSS.FOCAL_ALPHA = 0.75

    # number of neg anchors to samples
    cfg.MODEL.CONV_MODEL.DET_LOSS.NUM_NEG = 10000
    # top hard neg anchor when there is no positive anchor
    cfg.MODEL.CONV_MODEL.DET_LOSS.NUM_HARD_NEG = 100
    # neg:pos ratio when there are positive anchors
    cfg.MODEL.CONV_MODEL.DET_LOSS.NEG_RATIO = 100
    # NOTE: num_hard_neg and neg ratio should be equal and the loss is scaled up by this number
    cfg.MODEL.CONV_MODEL.DET_LOSS.CLS_W = 1.0
    cfg.MODEL.CONV_MODEL.DET_LOSS.SHAPE_W = 5.0
    cfg.MODEL.CONV_MODEL.DET_LOSS.OFFSET_W = 5.0
    cfg.MODEL.CONV_MODEL.DET_LOSS.IOU_W = 1.0

    cfg.MODEL.CONV_MODEL.DET_POSTPROCESS = CN()
    # topk candidates for each patch
    cfg.MODEL.CONV_MODEL.DET_POSTPROCESS.TOPK = 60
    # score threshold for candidates
    cfg.MODEL.CONV_MODEL.DET_POSTPROCESS.SCORE_THRESHOLD = 0.15
    # IOU threshold for nms
    cfg.MODEL.CONV_MODEL.DET_POSTPROCESS.NMS_THRESHOLD = 0.05
    # topk after applying nms
    cfg.MODEL.CONV_MODEL.DET_POSTPROCESS.NMS_TOPK = 20

    cfg.MODEL.SEMI_SPARSE = CN()
    cfg.MODEL.SEMI_SPARSE.TOPK = 20  # choosing top k from mask prediction

    cfg.DATA = CN()
    cfg.DATA.N_CHANNELS = 1
    cfg.DATA.PATCH_SIZE = (96, 96, 96)
    cfg.DATA.OVERLAP = (48, 48, 48)
    cfg.DATA.SPACING = (0.4, 0.4, 0.4)
    cfg.DATA.WINDOW = [0.0, 800.0]

    cfg.DATA.DIR = CN()
    cfg.DATA.DIR.TRAIN = CN()
    cfg.DATA.DIR.TRAIN.SCAN_DIR = ""
    cfg.DATA.DIR.TRAIN.ANNOTATION_FILE = ""
    cfg.DATA.DIR.TRAIN.VESSEL_DIR = ""
    cfg.DATA.DIR.VAL = CN()
    cfg.DATA.DIR.VAL.SCAN_DIR = ""
    cfg.DATA.DIR.VAL.ANNOTATION_FILE = ""

    cfg.DATA.DIR.VAL.VESSEL_DIR = ""

    cfg.DATA.CROPPING_AUG = CN()
    cfg.DATA.CROPPING_AUG.TRANSLATION = (20.0, 20.0, 20.0)
    cfg.DATA.CROPPING_AUG.ROTATION = (20.0, 0.0, 0.0)
    cfg.DATA.CROPPING_AUG.SPACING = [0.9, 1.2]
    cfg.DATA.CROPPING_AUG.BLANK_SIDE = 0
    cfg.DATA.CROPPING_AUG.TP_RATIO = 0.5

    cfg.CUSTOM = CN()
    cfg.CUSTOM.DATASET_FUNCTION = "CTADatasetFunction"
    cfg.CUSTOM.DATASET_MAPPER = "CTADatasetMapper"
    cfg.CUSTOM.TRACKING_GRADIENT_NORM = False
    cfg.CUSTOM.DEBUG = False
    cfg.CUSTOM.DUMMY_SIZE = 100
    cfg.CUSTOM.FULL_MASK = False

    # ocntrols the number of scans to use during debug
    cfg.CUSTOM.DEBUG_DATASET_SIZE = 1000
    cfg.CUSTOM.USE_SINGLE_BATCH = False
    cfg.CUSTOM.DEFAULT_INIT = False
    cfg.CUSTOM.CACHE = False
    cfg.CUSTOM.CLEAR_CUDA_CACHE_PERIOD = 1000

    cfg.SOLVER.SCANS_PER_BATCH = 8
    cfg.SOLVER.SAMPLES_PER_SCAN = 8
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MIN_LR = 0.00001
    cfg.SOLVER.SCHED_CYCLE = 2000
    cfg.SOLVER.FLAT_ITER = 2000
    cfg.SOLVER.AMP.ENABLED = False

    cfg.SOLVER.GRAD_ACCUM = CN()
    cfg.SOLVER.GRAD_ACCUM.STEPS = 1
    cfg.SOLVER.GRAD_ACCUM.ENABLED = False

    cfg.TEST.PATCHES_PER_ITER = 8
    cfg.TEST.NMS_TOPK = 20

    cfg.POSTPROCESS = CN()
    cfg.POSTPROCESS.ANNOTATION_DIR = ""
    cfg.POSTPROCESS.CHECKPOINT = "final"
    cfg.POSTPROCESS.THRESHOLD = 0.5
