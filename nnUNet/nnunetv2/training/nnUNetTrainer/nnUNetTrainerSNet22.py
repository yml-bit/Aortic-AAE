from nnunetv2.nets.Mednext.MedNextV1 import MedNeXt#原始
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import autocast, nn
import torch

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from nnunetv2.nets.AortaNet.AortaV1 import Aorta2#增加了输出层
from monai.losses import DiceCELoss,DiceFocalLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss,DC_and_CE_losss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
import numpy as np
import torch.nn.functional as F

class nnUNetTrainerSNet22(nnUNetTrainer):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4 #nnunetv2:3e-4 mednext:0.001
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33 # 0.33
        self.num_epochs = 450##1000  400x250=10 0000
        self.save_every = 1

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # MedNeXt
        label_manager = plans_manager.get_label_manager(dataset_json)
        model = Aorta2(in_channels=num_input_channels,n_channels=32,
            n_classes=label_manager.num_segmentation_heads,exp_r=[2,3,4,4,4,4,4,3,2],#[2,3,4,4,4,4,4,3,2] [1,2,3,3,3,3,3,2,1]
            kernel_size=3,deep_supervision=True,do_res=True,
            do_res_up_down=True,block_counts=[3,4,4,4,4,4,4,4,3],#True False [3,4,4,4,4,4,4,4,3]
            checkpoint_style='outside_block')
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),lr=self.initial_lr,amsgrad=True)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")
        return optimizer, scheduler

    def _build_loss(self):
        # 计算类别权重（假设你的类别分布为 lumen:1, calc:2, non-calc:3）
        #统计训练集后获得的逆频率权重（需根据实际数据调整）
        # class_weights = torch.tensor([1.0,1.0, 3.0, 5.0])  # 管腔:1x，钙化:3x，非钙化:5x
        class_weights = torch.tensor([1.0,1.0, 3.0, 5.0], device=self.device)
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_losss(
                soft_dice_kwargs={
                    'batch_dice': self.configuration_manager.batch_dice,
                    'smooth': 1e-5,
                    'do_bg': False,
                    'class_weights': [0.2,0.2, 0.3, 0.5]  # Dice 权重
                },
                ce_kwargs={'weight': class_weights},  # 交叉熵权重
                weight_ce=1.5,
                weight_dice=1.0,
                ignore_label=self.label_manager.ignore_label,
                dice_class=self.WeightedSoftDiceLoss  # 使用自定义加权 Dice
            )

        # 深度监督逻辑（确保无 per_layer_args 参数）
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)  # 移除 per_layer_args

        return loss

    class WeightedSoftDiceLoss(nn.Module):
        def __init__(self, apply_nonlin=None, batch_dice=True, smooth=1e-5,
                     do_bg=False, class_weights=None):
            super().__init__()
            self.apply_nonlin = apply_nonlin
            self.batch_dice = batch_dice
            self.smooth = smooth
            self.do_bg = do_bg
            self.class_weights = class_weights  # 保存类别权重

        def forward(self, x, y, loss_mask=None):
            # 应用激活函数（如 softmax）
            if self.apply_nonlin is not None:
                x = self.apply_nonlin(x)

            # 计算 Dice 系数
            axes = [0] + list(range(2, len(x.shape)))
            intersect = torch.sum(x * y, dim=axes)
            sum_pred = torch.sum(x, dim=axes)
            sum_true = torch.sum(y, dim=axes)
            dice = (2.0 * intersect + self.smooth) / (sum_pred + sum_true + self.smooth)

            # 应用类别权重
            if self.class_weights is not None:
                device = x.device
                weights = torch.tensor(self.class_weights).to(device)
                dice = dice * weights

            # 平均或加权求和
            dice_loss = 1 - dice.mean()  # 可根据需求调整加权方式
            return dice_loss

    # def _build_loss(self):
    #     if self.label_manager.has_regions:
    #         loss = DC_and_BCE_loss({},
    #                                {'batch_dice': self.configuration_manager.batch_dice,
    #                                 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
    #                                use_ignore_label=self.label_manager.ignore_label is not None,
    #                                dice_class=MemoryEfficientSoftDiceLoss)
    #     else:
    #         loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
    #                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
    #                               ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
    #         # loss = DiceFocalLoss(to_onehot_y=True, softmax=True,reduction="none")#效果不如原始
    #     # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    #     # this gives higher resolution outputs more weight in the loss
    #
    #     if self.enable_deep_supervision:
    #         deep_supervision_scales = self._get_deep_supervision_scales()
    #         weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
    #         weights[-1] = 0
    #
    #         # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    #         weights = weights / weights.sum()
    #         # now wrap the loss
    #         loss = DeepSupervisionWrapper(loss, weights)
    #     return loss