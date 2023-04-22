import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight

# 自定义BCELoss类添加了对样本权重的支持，
# 这可以用于处理不平衡类别。
# 这种权重是根据目标标签中正负样本的比例计算的。
# 这是一个额外的功能，用于在训练过程中根据不平衡的类别分布调整损失。
#
# 自定义BCELoss类中还添加了可选的标签平滑功能。
# 标签平滑可以增加模型的泛化能力，减少过拟合。
# 通过将目标值调整为接近0和1之间的值（如0.9和0.1），而不是绝对的0和1，可以鼓励模型更加稳定地学习。
@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

    def forward(self, logits, targets):
        logits = logits[0]

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)

            loss_m = (loss_m * sample_weight.cuda())

        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        return [loss], [loss_m]