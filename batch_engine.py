import math
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.distributed import reduce_tensor
from tools.utils import AverageMeter, to_scalar, time_str


def logits4pred(criterion, logits_list):
    if criterion.__class__.__name__.lower() in ['bceloss']:
        logits = logits_list[0]
        probs = logits.sigmoid()
    else:
        assert False, f"{criterion.__class__.__name__.lower()} not exits"

    return probs, logits


def batch_trainer(cfg, args, epoch, model, model_ema, train_loader, criterion, optimizer, loss_w=[1, ], scheduler=None):
    model.train()
    epoch_time = time.time()

    loss_meter = AverageMeter()
    subloss_meters = [AverageMeter() for i in range(len(loss_w))]

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    preds_logits = []
    imgname_list = []
    loss_mtr_list = []

    lr = optimizer.param_groups[1]['lr']

    for step, (imgs, gt_label, imgname) in enumerate(train_loader):
        iter_num = epoch * len(train_loader) + step

        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        train_logits, feat = model(imgs, gt_label)


        loss_list, loss_mtr = criterion(train_logits, gt_label)

        train_loss = 0

        for i, l in enumerate(loss_w):
            train_loss += loss_list[i] * l

        optimizer.zero_grad()
        train_loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print("NO " + name)
        #     else:
        #         print("YES " + name)

        if cfg.TRAIN.CLIP_GRAD:
            clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works

        optimizer.step()

        if cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine' and scheduler is not None:
            scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        if len(loss_list) > 1:
            for i, meter in enumerate(subloss_meters):
                meter.update(
                    to_scalar(reduce_tensor(loss_list[i], args.world_size)
                              if args.distributed else loss_list[i]))
        loss_meter.update(to_scalar(reduce_tensor(train_loss, args.world_size) if args.distributed else train_loss))

        train_probs, train_logits = logits4pred(criterion, train_logits)

        gt_list.append(gt_label.cpu().numpy())
        preds_probs.append(train_probs.detach().cpu().numpy())
        preds_logits.append(train_logits.detach().cpu().numpy())

        imgname_list.append(imgname)

        log_interval = 100

        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            if args.local_rank == 0:
                print(f'{time_str()}, '
                      f'Step {step}/{batch_num} in Ep {epoch}, '
                      f'LR: [{optimizer.param_groups[0]["lr"]:.1e}, {optimizer.param_groups[1]["lr"]:.1e}] '
                      f'Time: {time.time() - batch_time:.2f}s , '
                      f'train_loss: {loss_meter.avg:.4f}, ')

                print([f'{meter.avg:.4f}' for meter in subloss_meters])

            # break

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    if args.local_rank == 0:
        print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list


def valid_trainer(cfg, args, epoch, model, valid_loader, criterion, loss_w=[1, ]):
    model.eval()
    loss_meter = AverageMeter()
    subloss_meters = [AverageMeter() for i in range(len(loss_w))]

    preds_probs = []
    preds_logits = []
    gt_list = []
    imgname_list = []
    loss_mtr_list = []

    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits, feat = model(imgs, gt_label)


            loss_list, loss_mtr = criterion(valid_logits, gt_label)
            valid_loss = 0
            for i, l in enumerate(loss_list):
                valid_loss += loss_w[i] * l

            valid_probs, valid_logits = logits4pred(criterion, valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            preds_logits.append(valid_logits.cpu().numpy())

            if len(loss_list) > 1:
                for i, meter in enumerate(subloss_meters):
                    meter.update(
                        to_scalar(reduce_tensor(loss_list[i], args.world_size) if args.distributed else loss_list[i]))
            loss_meter.update(to_scalar(reduce_tensor(valid_loss, args.world_size) if args.distributed else valid_loss))

            torch.cuda.synchronize()

            imgname_list.append(imgname)

    valid_loss = loss_meter.avg

    if args.local_rank == 0:
        print([f'{meter.avg:.4f}' for meter in subloss_meters])

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    preds_logits = np.concatenate(preds_logits, axis=0)

    return valid_loss, gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list
'''infer.py 和 batch_engine.py 中的 valid_trainer 函数确实有一些相似之处，
但它们的目的和功能略有不同。以下是它们的相同点和不同点：

相同点：

都会对模型在验证集上的表现进行评估。
都会将模型设置为评估模式。
都会遍历验证数据加载器，并将图片和标签转移到GPU上。
都会计算模型输出，并将logits转换为概率值。
都会将预测结果、真实标签和图片路径添加到相应的列表中。
不同点：

infer.py 是一个独立的脚本，主要用于加载预训练的模型并对其在验证集上的性能进行评估，
而 valid_trainer 函数是 batch_engine.py 中的一个组件，通常在训练过程中周期性地评估模型性能。
infer.py 主要关注评估指标的计算和结果的展示，
而 valid_trainer 函数更关注计算验证损失以及各种评估指标，以便在训练过程中监控模型性能。
infer.py 会将评估结果保存到pickle文件中，
而 valid_trainer 通常将评估结果返回给调用者，以便在训练过程中进行进一步处理。
总之，虽然 infer.py 和 batch_engine.py 中的 valid_trainer 函数在实现上有相似之处，
但它们的目的和使用场景略有不同。infer.py 主要用于在训练完成后加载预训练的模型并评估其性能，而 valid_trainer 函数则在训练过程中周期性地评估模型性能。'''