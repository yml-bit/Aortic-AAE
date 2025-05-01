import torch
torch.multiprocessing.set_sharing_strategy('file_system')##训练python脚本中import torch后，加上下面这句。
from load_datasets_transforms_seg import data_loader, data_transforms, infer_post_transforms,remove_regions
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete,SaveImaged
from monai.metrics import DiceMetric,MeanIoU,ConfusionMatrixMetric,SurfaceDistanceMetric
from monai.data.meta_tensor import MetaTensor


import os
import argparse
import yaml
import random
import numpy as np
import SimpleITK as sitk

def config():
    parser = argparse.ArgumentParser(description='overal test')
    ## Input data hyperparameters
    # parser.add_argument('--root', type=str, default='', required=True, help='Root folder of all your images and labels')
    parser.add_argument('--dataset', type=str, default='301', help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
    parser.add_argument('--patch', type=int, default=(96,96,96), help='Batch size for subject input')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')#4 2
    parser.add_argument('--mode', type=str, default='overal_test', help='Training or testing mode')
    parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
    parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
    parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
    parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

    ## Efficiency hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
    parser.add_argument('--cache_rate', type=float, default=0.2, help='Cache rate to cache your dataset into GPUs')#0.1
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers')

    args = parser.parse_args()
    return args

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True / False

def dis_dimention(path,target_path):
    # gd0 = sitk.ReadImage(path.replace("2.nii.gz","0.nii.gz"), sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    # gd_array0 = sitk.GetArrayFromImage(gd0)
    #
    # gd1 = sitk.ReadImage(path.replace("2.nii.gz","1.nii.gz"), sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    # gd_array1 = sitk.GetArrayFromImage(gd1)
    flag=0
    gd = sitk.ReadImage(path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    gd_array = sitk.GetArrayFromImage(gd)
    target = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    target_array = sitk.GetArrayFromImage(target)
    if gd_array.shape != target_array.shape:
        print(path)
        flag=1
    return flag

def check_data(path,target_path):
    gd = sitk.ReadImage(path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    gd_array0 = sitk.GetArrayFromImage(gd)
    target = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    target_array0 = sitk.GetArrayFromImage(target)
    flag=0
    if np.all(gd_array0 == 0) or np.all(target_array0 == 0):
        print("all is 0!")
        print(path)
        flag = 1
    if np.all(gd_array0 == 1) or np.all(target_array0 == 1):
        print("all is 1!")
        print(path)
        flag = 1
    return flag

def split_dataset_test():
    ii = 0
    # models = ["p2res", "SwinUNETR", "unet", "p2ux", "p2nnf", "MedNeXt","Aortic_index_v1"]  #
    models = ["p2res"]
    labels = ["hnnk", "lz", "cq"]
    # labels = ["cq"]
    out = "./data/segement_metricx/"
    os.makedirs(out, exist_ok=True)
    for model in models:
        # model="nnUNetTrainerMaCNN"
        # output_file = out + model + "_segement_total.txt"
        output_file = out + model + "_segement.txt"
        if os.path.exists(output_file):
            os.remove(output_file)
        # output_file = out + model + "_segement.txt"
        i = 0
        for label in labels:
            test_list = []
            labelsTs_list = []
            out_test_list = []
            labelsTs = "/media/bit301/data/yml/data/p2_nii/external/" + label
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "2.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        test_list.append(path)
            # for root, dirs, files in os.walk(labelsTs, topdown=False):
            #     for k in range(len(files)):
            #         path = os.path.join(root, files[k])
            for path in test_list:
                ppp = path.split("external/")[1]  # test
                pp=ppp.replace("2.nii.gz","0.nii.gz")
                if pp in dis_list:
                    ii = ii + 1
                    continue
                if "2.nii.gz" in path:
                    # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                    labelsTs_list.append(path)
                    # 3DUXNET SwinUNETR MedNeXt unet MedNeXtl MedNeXtx1c MedNeXtx1cc nSwinUNETR
                    # p2res p2nnf p2ux
                    # p2res SwinUNETR unet 3DUXNET p2ux p2nnf MedNeXt Aortic_index（MedNeXtx2(未做钙化处理)）
                    if model == "Aortic_index_v1":
                        target_path = path.replace("external", "statisticians/" + model).replace("2.nii.gz", "22.nii.gz")
                    else:
                        # Aortic_index_v1:模型分割结果   Aortic_index：后处理结果，用于计算几何参数
                        target_path = path.replace("external", "test/" + model)  # test/MedNeXtx2
                    out_test_list.append(target_path)

            val_files = [{"image": image_name, "label": label_name}
                         for image_name, label_name in
                         zip(labelsTs_list, out_test_list)]
            val_transforms = data_transforms(args)

            ## Inference Pytorch Data Loader and Caching
            val_ds = CacheDataset(
                data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")
            patch = np.array(args.patch, dtype=int)  # (96,96,48)
            out_classes = args.num_classes
            post_label = AsDiscrete(to_onehot=out_classes)
            post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
            # dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)  # 去除背景项目
            IoU_metric = MeanIoU(include_background=False, reduction="none", get_not_nans=False)
            # conf_matrix_metric=ConfusionMatrixMetric(include_background=True, reduction="none", get_not_nans=False)#percentile=95,
            conf_matrix_metric = ConfusionMatrixMetric(include_background=False, metric_name="precision", reduction="none", get_not_nans=False)

            # assd_Matrix=SurfaceDistanceMetric(include_background=False, reduction="none", symmetric=False,get_not_nans=False)#ASSD应该设置 symmetric=True  结果存在inf指
            dice_vals = list()
            for i, val_data in enumerate(val_loader):  # 读取数据不对可能会导致数据加载报错
                roi_size = patch  # roi_size=(96, 96, 96)
                a = val_data["image"]
                a[a>2]=3
                # a[a > 0] = 1
                # a=np.where(a>1,2,a)
                b = val_data["label"]
                b[b > 2] = 3
                # b[b > 0] = 1
                del val_data
                val_labels, val_outputs = (a.to(device), b.to(device))  # 512x512x370
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_output_list = decollate_batch(val_outputs)
                val_output_convert = [post_label(val_output_tensor) for val_output_tensor in val_output_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                # dice = dice_metric.aggregate().item()
                # dice_vals.append(dice)
                dice = dice_metric.aggregate().cpu().detach().numpy()
                IoU_metric(y_pred=val_output_convert, y=val_labels_convert)
                iou = IoU_metric.aggregate().cpu().detach().numpy()
                conf_matrix_metric(y_pred=val_output_convert, y=val_labels_convert)
                ppv = conf_matrix_metric.aggregate()[0].cpu().detach().numpy()
                # assd_Matrix(y_pred=val_output_convert, y=val_labels_convert)
                # assd = assd_Matrix.aggregate().cpu().detach().numpy()
                # a=1

            # mean_dice_val = np.mean(dice_vals)
            # print("dice:",mean_dice_val)
            sub_mean_dice = np.nanmean(dice, axis=0)  # 列平均
            sub_std_dice = np.nanstd(dice, axis=0)

            sub_mean_ppv = np.nanmean(ppv, axis=0)  # 列平均
            sub_std_ppv = np.nanstd(ppv, axis=0)  # 列平均

            sub_mean_iou = np.nanmean(iou, axis=0)  # 列平均
            sub_std_iou = np.nanstd(iou, axis=0)  # 列平均
            # print("sub_mean_dice:", sub_mean_dice)
            # print("sub_mean_ppv:", sub_mean_ppv)
            # print("sub_mean_iou:", sub_mean_iou)

            # total_mean_dice = np.mean(sub_mean_dice)  # 列平均
            # print("total_mean_dice:", total_mean_dice)
            # 输出混淆矩阵
            outname = model + "    " + label
            with open(output_file, 'a') as file:
                file.write(outname + "\n")
                file.write(f"{sub_mean_dice}\n")
                file.write(f"{sub_std_dice}\n")
                file.write("\n")
                file.write(f"{sub_mean_ppv}\n")
                file.write(f"{sub_std_ppv}\n")
                file.write("\n")
                file.write(f"{sub_mean_iou}\n")
                file.write(f"{sub_std_iou}\n")
                file.write("\n")
        print("finished " + model)

def total_test():
    ii = 0
    # models=["Aortic_index_v1"]#未经过处理
    # models = ["p2res", "SwinUNETR", "unet", "p2ux", "p2nnf", "MedNeXt","Aortic_index_v1"]  #
    models = ["unet", "p2ux", "p2nnf", "MedNeXt", "Aortic_index_v1"]  #
    out = "./data/segement_metricx1/"
    os.makedirs(out, exist_ok=True)
    for model in models:
        # model="nnUNetTrainerMaCNN"
        # output_file = out + model + "_segement_total_1dataset.txt"
        output_file = out + model + "_segement_1dataset.txt"
        if os.path.exists(output_file):
            os.remove(output_file)
        # output_file = out + model + "_segement.txt"
        i = 0

        test_list = []
        labelsTs_list = []
        out_test_list = []
        labelsTs = "/media/bit301/data/yml/data/p2_nii/external/"
        for root, dirs, files in os.walk(labelsTs, topdown=False):
            for k in range(len(files)):
                path = os.path.join(root, files[k])
                if "2.nii.gz" in path:
                    # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                    test_list.append(path)
        for path in test_list:
            ppp = path.split("external/")[1]  # test
            pp = ppp.replace("2.nii.gz", "0.nii.gz")
            if pp in dis_list:
                ii = ii + 1
                continue
            if "2.nii.gz" in path:
                # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                labelsTs_list.append(path)
                # 3DUXNET SwinUNETR MedNeXt unet MedNeXtl MedNeXtx1c MedNeXtx1cc nSwinUNETR
                # p2res p2nnf p2ux
                # p2res SwinUNETR unet p2ux p2nnf MedNeXt Aortic_index（MedNeXtx2(未做钙化处理)）
                if model == "Aortic_index_v1":
                    target_path = path.replace("external", "statisticians/" + model).replace("2.nii.gz", "22.nii.gz")
                else:
                    # Aortic_index_v1:模型分割结果   Aortic_index：后处理结果，用于计算几何参数
                    target_path = path.replace("external", "test/" + model)  # test/MedNeXtx2
                out_test_list.append(target_path)

        val_files = [{"image": image_name, "label": label_name}
                     for image_name, label_name in
                     zip(labelsTs_list, out_test_list)]
        val_transforms = data_transforms(args)

        ## Inference Pytorch Data Loader and Caching
        val_ds = CacheDataset(
            data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        patch = np.array(args.patch, dtype=int)  # (96,96,48)
        out_classes = args.num_classes
        post_label = AsDiscrete(to_onehot=out_classes)
        post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
        # dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)  # 去除背景项目
        IoU_metric = MeanIoU(include_background=False, reduction="none", get_not_nans=False)
        # conf_matrix_metric=ConfusionMatrixMetric(include_background=True, reduction="none", get_not_nans=False)#percentile=95,
        conf_matrix_metric = ConfusionMatrixMetric(include_background=False, metric_name="precision", reduction="none", get_not_nans=False)

        # assd_Matrix=SurfaceDistanceMetric(include_background=False, reduction="none", symmetric=False,get_not_nans=False)#ASSD应该设置 symmetric=True  结果存在inf指
        dice_vals = list()
        for i, val_data in enumerate(val_loader):  # 读取数据不对可能会导致数据加载报错
            roi_size = patch  # roi_size=(96, 96, 96)
            a = val_data["image"]
            a[a>2]=3
            # a[a > 0] = 1
            # a=np.where(a>1,2,a)
            b = val_data["label"]
            b[b > 2] = 3
            # b[b > 0] = 1
            del val_data
            val_labels, val_outputs = (a.to(device), b.to(device))  # 512x512x370
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_output_list = decollate_batch(val_outputs)
            val_output_convert = [post_label(val_output_tensor) for val_output_tensor in val_output_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            # dice = dice_metric.aggregate().item()
            # dice_vals.append(dice)
            dice = dice_metric.aggregate().cpu().detach().numpy()
            IoU_metric(y_pred=val_output_convert, y=val_labels_convert)
            iou = IoU_metric.aggregate().cpu().detach().numpy()
            conf_matrix_metric(y_pred=val_output_convert, y=val_labels_convert)
            ppv = conf_matrix_metric.aggregate()[0].cpu().detach().numpy()
            # assd_Matrix(y_pred=val_output_convert, y=val_labels_convert)
            # assd = assd_Matrix.aggregate().cpu().detach().numpy()
            # a=1

        # mean_dice_val = np.mean(dice_vals)
        # print("dice:",mean_dice_val)
        sub_mean_dice = np.nanmean(dice, axis=0)  # 列平均
        sub_std_dice = np.nanstd(dice, axis=0)

        sub_mean_ppv = np.nanmean(ppv, axis=0)  # 列平均
        sub_std_ppv = np.nanstd(ppv, axis=0)  # 列平均

        sub_mean_iou = np.nanmean(iou, axis=0)  # 列平均
        sub_std_iou = np.nanstd(iou, axis=0)  # 列平均
        # print("sub_mean_dice:", sub_mean_dice)
        # print("sub_mean_ppv:", sub_mean_ppv)
        # print("sub_mean_iou:", sub_mean_iou)

        # total_mean_dice = np.mean(sub_mean_dice)  # 列平均
        # print("total_mean_dice:", total_mean_dice)
        # 输出混淆矩阵
        outname = model
        with open(output_file, 'a') as file:
            file.write(outname + "\n")
            file.write(f"{sub_mean_dice}\n")
            file.write(f"{sub_std_dice}\n")
            file.write("\n")
            file.write(f"{sub_mean_ppv}\n")
            file.write(f"{sub_std_ppv}\n")
            file.write("\n")
            file.write(f"{sub_mean_iou}\n")
            file.write(f"{sub_std_iou}\n")
            file.write("\n")
    print("finished " + model)

if __name__ == '__main__':
    #对比模型等对比实验测试
    args=config() #the first running should excute this code
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #注意修改num_classes=4 or total num_classes=2
    f = open("/media/bit301/data/yml/project/python39/p2/process/list/discard.txt")  # dml jc
    dis_list = []
    for line in f.readlines():  # tile_step_size=0.75较好处理官腔错位问题
        path=line.split('\n')[0].split("external/")[1]
        dis_list.append(path)
    split_dataset_test()#dis_list 可以传递到下属函数中，
    # total_test()#one dataset