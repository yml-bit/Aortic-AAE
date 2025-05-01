from nnunetv2.nets.Mednext.MedNextV1 import MedNeXt#原始
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import \
#     nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import autocast, nn
import torch

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from nnunetv2.nets.AortaNet.AortaV1 import Aorta2b

class nnUNetTrainerSNet2b(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # MedNeXt
        label_manager = plans_manager.get_label_manager(dataset_json)
        model = Aorta2b(in_channels=num_input_channels,n_channels=32,
            n_classes=label_manager.num_segmentation_heads,exp_r=[2,3,4,4,4,4,4,3,2],#[2,3,4,4,4,4,4,3,2] [1,2,3,3,3,3,3,2,1]
            kernel_size=3,deep_supervision=True,do_res=True,
            do_res_up_down=True,block_counts=[3,4,4,4,4,4,4,4,3],#True False [3,4,4,4,4,4,4,4,3]
            checkpoint_style='outside_block')

        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    # def configure_optimizers(self):
    #     optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
    #     scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
    #     self.print_to_log_file(f"Using optimizer {optimizer}")
    #     self.print_to_log_file(f"Using scheduler {scheduler}")
    #     return optimizer, scheduler

    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.0)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")
        return optimizer, scheduler