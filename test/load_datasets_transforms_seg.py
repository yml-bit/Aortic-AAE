from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from monai.data.meta_tensor import MetaTensor
import os
from tqdm import tqdm
import numpy as np
# from mypath import Path##这里是作者自己创建的一个文件，用来生成路径的

from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    # RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    Activationsd,

    RandAffine,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandZoomd,
)
from networks.DeMT_3D.transforms import RandCropByPosNegLabeld  # control ratio
from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def data_loader(args):
    dataset = args.dataset
    out_classes = args.num_classes

    if args.mode == 'train':
        f = open(args.train_list)  # test train
        train_lista = []
        train_listb = []
        train_listc = []
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            train_lista.append(line)
            train_listb.append(line.replace("0.nii.gz", "1.nii.gz"))
            train_listc.append(line.replace("0.nii.gz", "2.nii.gz"))
        f.close()  # 关

        train_samples = {}
        valid_samples = {}
        train_samples['images'] = train_lista
        # train_samples['targets'] = train_listb
        train_samples['labels'] = train_listc

        # val
        f = open(args.val_list)  # test train
        val_lista = []
        val_listb = []
        val_listc = []
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            val_lista.append(line)
            val_listb.append(line.replace("0.nii.gz", "1.nii.gz"))
            val_listc.append(line.replace("0.nii.gz", "2.nii.gz"))
        f.close()  # 关

        valid_samples['images'] = val_lista
        # valid_samples['targets'] = val_listb
        valid_samples['labels'] = val_listc

        ## Input training data
        # train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
        # train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
        # train_samples['images'] = train_img
        # train_samples['labels'] = train_label
        #
        # ## Input validation data
        # valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesVal', '*.nii.gz')))
        # valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
        # valid_samples['images'] = valid_img
        # valid_samples['labels'] = valid_label

        print('Finished loading all training samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))

        return train_samples, valid_samples, out_classes

    elif args.mode == 'val':
        valid_samples = {}
        f = open(args.val_list)  # test train
        val_lista = []
        val_listb = []
        val_listc = []
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            val_lista.append(line)
            val_listb.append(line.replace("0.nii.gz", "1.nii.gz"))
            val_listc.append(line.replace("0.nii.gz", "2.nii.gz"))
        f.close()  # 关

        valid_samples['images'] = val_lista
        valid_samples['targets'] = val_listb
        valid_samples['labels'] = val_listc

        print('Finished loading all inference samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))
        return valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}
        f = open(args.test_list)  # test train
        test_lista = []
        test_listb = []
        test_listc = []
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            test_lista.append(line)
            test_listb.append(line.replace("0.nii.gz", "1.nii.gz"))
            test_listc.append(line.replace("0.nii.gz", "2.nii.gz"))
        f.close()  # 关
        test_samples['images'] = test_lista
        test_samples['targets'] = test_listb
        test_samples['labels'] = test_listc

        # ## Input inference data
        # test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
        # test_samples['images'] = test_img
        print('Finished loading all inference samples from dataset: {}!'.format(dataset))
        return test_samples, out_classes


def preprocess_label_only(data):
    label = data["label"]
    data["labels"] = []
    for scale in [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]:
        spatial_size = (label.shape[1] * scale, label.shape[2] * scale, label.shape[3] * scale)
        Resize = ResizeWithPadOrCropd(keys=["label"], spatial_size=spatial_size)({"label": label})
        data["labels"].append(Resize["label"])
    return data


def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None
    if dataset == '301':  # (96, 96, 96)  (96, 96, 48)  (1, 512, 462, 332)
        # patch_size=(128, 128, 48)#(96, 96, 64)#(96, 96, 48)
        patch_size = np.array(args.patch, dtype=int)  # (96,96,48)
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
                Spacingd(keys=["image", "label"], pixdim=(
                    1, 1, 1), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-500, a_max=500,  # -500
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),  # 去掉空白留白,保证输入的是数值图，
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=patch_size,
                    pos=1,
                    neg=2,  # 阴性阳性比
                    num_samples=crop_samples,
                    image_key="image",
                    # allow_smaller=True,#允许裁减尺寸大于输入图像尺寸
                ),
                # RandZoomd(keys=["image"], prob=0.2, keep_size=True),  # 随机缩放
                # RandGaussianNoised(keys=["image"], mean=0.0, std=0.1),  # 添加高斯噪声
                # RandGaussianSmoothd(keys=["image"], prob=0.2),  # 高斯模糊
                # RandAdjustContrastd(keys=["image"], prob=0.2),  # 对比度调节
                # RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2)),
                # RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.2),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=patch_size,
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
                preprocess_label_only,
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
                Spacingd(keys=["image", "label"], pixdim=(
                    1, 1, 1), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-500, a_max=500,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                # CropForegroundd(keys=["image","target", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),

            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
                Spacingd(keys=["image", "label"], pixdim=(
                    1, 1, 1), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-500, a_max=500,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                # CropForegroundd(keys=["image","target", "label"], source_key="image"),
                # ToTensord(keys=["image","label"]),
            ]
        )
        overal_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     1.25, 1.25, 1.25), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ToTensord(keys=["image"]),  # original
            ]
        )

    if args.mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'val':
        print('Performed transformations for all samples!')
        return val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms
    elif args.mode == "overal_test":
        return overal_transforms

# DeepLab v3+ model
def calculate_weigths_labels(dataset, dataloader, num_classes, path):
    # Create an instance from the data loader
    path = path.split("train")[0]
    num_classes = num_classes
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']  ##这里是作者创建的一个dataloader，这里的sample['label']返回的是标签图像的lable mask
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < (num_classes + 1))
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)  ##统计每幅图像中不同类别像素的个数
        z += count_l
    tqdm_batch.close()
    z = z[1:]  # [25208301.  1889810.    36304.    77890.   214511.]
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))  ##这里是计算每个类别像素的权重
        # class_weight = (1/(num_classes-1))/frequency / total_frequency
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    ret = ret / np.sum(ret[0])  # 权重归一化
    classes_weights_path = os.path.join(path, 'classes_weights.txt')  ##生成权重文件
    np.savetxt(classes_weights_path, ret)  ##把各类别像素权重保存到一个文件中
    print("catagery weight:", ret)
    return ret


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1,) * len(args[0])
        else:
            weights = self.weight_factors
        taotal_loss = sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
        return taotal_loss


def infer_post_transforms(output, test_transforms, out_classes):
    print("notice:comment the line 558 of monai/transforms/utility/dictionary.py")
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),

        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        ## If monai version <= 0.6.0:
        # AsDiscreted(keys="pred", threshold=0.5),
        AsDiscreted(keys="pred", argmax=True, n_classes=out_classes),
        ## If moani version > 0.6.0:
        # AsDiscreted(keys="pred", argmax=True)
        # KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3]),
        # SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output,
        #            output_postfix="seg", output_ext=".nii.gz", separate_folder=False,resample=True),
    ])

    return post_transforms


# 已测试
def remove_regions(mask):
    mask1 = mask
    mask1 = np.where(mask1 > 0, 1, 0)
    # mask2 = mask
    # mask2=np.where(mask2>1,1,0)
    segmentation_sitk = sitk.GetImageFromArray(mask1)
    # 计算标签图像的连通分量
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = connected_components_filter.Execute(segmentation_sitk)

    # 获取每个连通分量的大小
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labeled_image)

    # 找到最大的连通分量ID
    max_size = 0
    largest_label = 0
    for i in range(1, label_shape_filter.GetNumberOfLabels() + 1):  # Label index starts from 1
        if label_shape_filter.GetNumberOfPixels(i) > max_size:
            max_size = label_shape_filter.GetNumberOfPixels(i)
            largest_label = i

    # 仅保留最大连通分量
    binary_mask = sitk.Equal(labeled_image, largest_label)
    cleaned_segmentation = sitk.Cast(binary_mask, segmentation_sitk.GetPixelID())
    cleaned_segmentation = sitk.GetArrayFromImage(cleaned_segmentation)
    cleaned_segmentation = cleaned_segmentation * mask
    # print(cleaned_segmentation.max())
    return cleaned_segmentation