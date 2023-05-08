import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from dataset.augmentation import get_transform
from dataset.pedes_attr.pedes import PedesAttr
from models.base_block import FeatClassifier
from models.model_factory import build_backbone, build_classifier
from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from configs.default import _C, update_config

set_seed(605)

def main(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    _, valid_tsfm = get_transform(cfg)

    train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
                          target_transform=cfg.DATASET.TARGETTRANSFORM)
    valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                              target_transform=cfg.DATASET.TARGETTRANSFORM)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=valid_set.attr_num,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale=1  # Disable feature scaling by setting the scale to 1
    )
    print(valid_set.attr_num)
    print(train_set.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(os.path.join(model_dir, 'ckpt_max_2023-04-21_04-05-23.pth'))
    model.load_state_dict(checkpoint['state_dicts'])

    model.eval()
    feature_list = []
    gt_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, _) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            logits, features = model(imgs)
            features = features.detach().cpu().numpy().reshape(features.shape[0], -1)
            feature_list.append(features)
            gt_list.append(gt_label.cpu().numpy())

    features = np.vstack(feature_list)
    gt_labels = np.concatenate(gt_list, axis=0)

    similarity_matrix = cosine_similarity(features)

    retrieval_results = []
    for i in range(similarity_matrix.shape[0]):
        retrieval_rankings = np.argsort(similarity_matrix[i])[::-1]
        retrieval_results.append(retrieval_rankings)

    mAP, rank1 = calculate_metrics(retrieval_results, gt_labels)

    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print(f"Rank-1 Accuracy (R-1): {rank1:.4f}")

    with open(os.path.join(model_dir, 'attribute_retrieval_results.pkl'), 'wb+') as f:
        pickle.dump([retrieval_results, gt_labels], f, protocol=4)

    print(f'{time_str()}')
    print('-' * 60)


def calculate_metrics(rankings, gt_labels):
    print('okay1')
    ap_scores = []
    rank1_scores = 0

    gt_labels_int = np.argmax(gt_labels, axis=1)
    similarity_threshold = 0.999  # You can adjust this value to change the similarity threshold

    for i, ranking in enumerate(rankings):
        # Remove the query image itself from the ranking
        ranking = np.delete(ranking, np.where(ranking == i))
        relevant_indices = np.where(gt_labels_int == gt_labels_int[i])[0]
        relevant_indices = np.delete(relevant_indices, np.where(relevant_indices == i))

        num_relevant = 0
        rank_positions = []
        relevant_rankings = np.intersect1d(ranking, relevant_indices, assume_unique=True)
        rank_positions = (np.arange(1, len(relevant_rankings) + 1) / (np.searchsorted(ranking, relevant_rankings) + 1))

        if len(rank_positions) > 0:
            ap_scores.append(np.sum(rank_positions) / len(relevant_indices))  # Normalize AP by the number of relevant images
        else:
            ap_scores.append(0)

        # Check if the top-ranked image is relevant
        top_ranked_similarity = np.sum(gt_labels[ranking[0]] == gt_labels[i]) / gt_labels.shape[1]
        if ranking[0] in relevant_indices and top_ranked_similarity > similarity_threshold:
            rank1_scores += 1

    print('okay2')
    return np.mean(ap_scores), rank1_scores / len(rankings)


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute-based retrieval",
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
# python attribute_retrieval.py --cfg configs/pedes_baseline/pa100k.yaml
# python attribute_retrieval.py --cfg configs/pedes_baseline/peta.yaml