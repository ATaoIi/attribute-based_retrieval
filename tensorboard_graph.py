import distutils.version
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import os
import pickle

from dataset.augmentation import get_transform
from metrics.pedestrian_metrics import get_pedestrian_metrics, get_map_metrics
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default import _C, update_config
from dataset.pedes_attr.pedes import PedesAttr
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str

from losses import bceloss
from models.backbone import swin_transformer,DeiT,vit
set_seed(605)

from tools.function import compare_dicts
def create_tensorboard_graph(cfg,args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)
    train_tsfm, valid_tsfm = get_transform(cfg)

    train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
                          target_transform=cfg.DATASET.TARGETTRANSFORM)
    # Same setup as before
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)
    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=train_set.attr_num,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale=cfg.CLASSIFIER.SCALE
    )
    model = FeatClassifier(backbone, classifier)

    # Load the model
    model = get_reload_weight(model_dir, model,
                              pth=r'C:\Users\licha\PycharmProjects\attribute-based_retrieval\exp_result\PA100k\deit_base_distilled_patch16_224.base.adam\img_model\ckpt_max_2023-05-17_02-14-18.pth')

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create a TensorBoard writer
    writer = SummaryWriter(log_dir="runs/deit_base_distilled_patch16_224_structure")

    # Add the model graph to the writer
    writer.add_graph(model, input_to_model=torch.zeros((1, 3, 224, 224)).cuda())

    # Close the writer
    writer.close()


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(_C, args)

    create_tensorboard_graph(_C, args)

    # python tensorboard_graph.py --cfg configs/pedes_baseline/pa100k.yaml
    # python tensorboard_graph.py --cfg configs/pedes_baseline/peta.yaml
    #tensorboard --logdir=runs
    #tensorboard --logdir=tensorboard_logs
