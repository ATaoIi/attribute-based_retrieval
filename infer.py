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


def main(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    train_tsfm, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)

    train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
                          target_transform=cfg.DATASET.TARGETTRANSFORM)
    valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                          target_transform=cfg.DATASET.TARGETTRANSFORM)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{cfg.DATASET.TRAIN_SPLIT} set: {len(train_loader.dataset)}, '
          f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=train_set.attr_num,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale=cfg.CLASSIFIER.SCALE
    )
    print(train_set.attr_num)
    print(valid_set.attr_num)
    model = FeatClassifier(backbone, classifier)

    loaded_model_state_dict = get_reload_weight(model_dir, model,
                                                pth=r'C:\Users\licha\PycharmProjects\attribute-based_retrieval\exp_result\PETA\resnet50.base.adam\img_model\ckpt_max_2023-04-22_02-15-39.pth')

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # print("Keys in original model state_dict but not in loaded model state_dict:")
    # missing_in_original, missing_in_loaded = compare_dicts(model_original_state_dict, loaded_model_state_dict)
    # print(missing_in_loaded)
    #
    # print("Keys in loaded model state_dict but not in original model state_dict:")
    # print(missing_in_original)

    model = loaded_model_state_dict

    model.eval()
    preds_probs = []
    gt_list = []
    path_list = []

    attn_list = []

    # 这里使用cpu()将张量从GPU转移到CPU，是因为在将张量与numpy()一起使用之前，需要将张量从GPU内存转移到CPU内存。numpy()不支持GPU内存，所以在使用numpy()之前，必须将张量转移到CPU上。
    #
    # 这里的代码将特征和标签分别添加到feature_list和gt_list中。这些列表用于在计算过程结束时组合所有批次的特征和标签。因为这些列表最终会被转换为NumPy数组，所以需要将张量转移到CPU并将其转换为NumPy数组，这样在计算结束时可以轻松地连接它们。
    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            valid_logits, attns = model(imgs, gt_label)

            valid_probs = torch.sigmoid(valid_logits[0])

            path_list.extend(imgname)
            gt_list.append(gt_label.cpu().numpy())
            preds_probs.append(valid_probs.cpu().numpy())

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    if cfg.METRIC.TYPE == 'pedestrian':
        valid_result = get_pedestrian_metrics(gt_label, preds_probs)
        valid_map, _ = get_map_metrics(gt_label, preds_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  map: {:.4f}, label_f1: {:4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, valid_map, np.mean(valid_result.label_f1), np.mean(valid_result.label_pos_recall),
                  np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1)
              )

        with open(os.path.join(model_dir, 'results_test_feat_best.pkl'), 'wb+') as f:
            pickle.dump([valid_result, gt_label, preds_probs, attn_list, path_list], f, protocol=4)

        print(f'{time_str()}')
        print('-' * 60)


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

    main(_C, args)

# python infer.py --cfg configs/pedes_baseline/pa100k.yaml
# python infer.py --cfg configs/pedes_baseline/peta.yaml