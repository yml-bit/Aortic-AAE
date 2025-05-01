import SimpleITK as sitk
import numpy as np
from skimage import measure, morphology
import h5py
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation
import copy
import cv2
from skimage.morphology import binary_closing
from scipy.stats import pearsonr,binned_statistic
from sklearn.metrics import mean_absolute_error,r2_score,mean_absolute_percentage_error
from natsort import natsorted
import openpyxl
from sklearn.metrics import confusion_matrix
import pandas as pd
from scipy.ndimage import label, generate_binary_structure
from batchgenerators.utilities.file_and_folder_operations import *
import os
import shutil
from scipy.signal import butter,lfilter,savgol_filter
from scipy.signal import find_peaks
from collections import OrderedDict
import logging
logging.basicConfig(level=logging.DEBUG)
from collections import Counter
import math

# 设置打印选项，使得所有数组都以小数形式输出，且设置小数点后保留的位数
np.set_printoptions(suppress=True, precision=8)  # suppress=True 禁用科学记数法，precision设置小数点后的位数

##remain the  Connected region whichs more than 1000 voxel
def remove_small_volums(mask):
    mask1 = np.where(mask > 0, 1, 0)
    segmentation_sitk = sitk.GetImageFromArray(mask1)
    # 计算标签图像的连通分量
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = connected_components_filter.Execute(segmentation_sitk)
    # 获取每个连通分量的大小
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labeled_image)
    # 初始化一个空的数组来存储处理后的mask
    cleaned_segmentation = np.zeros_like(mask1)
    # 遍历每个连通分量，保留体积大于min_volume的连通区域
    for i in range(1, label_shape_filter.GetNumberOfLabels() + 1):
        if label_shape_filter.GetNumberOfPixels(i) >= 14000:
            binary_mask = sitk.Equal(labeled_image, i)
            binary_mask_array = sitk.GetArrayFromImage(binary_mask)
            cleaned_segmentation[binary_mask_array == 1] = 1
    # 返回处理后的mask
    cleaned_segmentation = cleaned_segmentation * mask
    return cleaned_segmentation.astype(np.int16)

def to_windowdata(image, WC, WW):
    # image = (image + 1) * 0.5 * 4095
    # image[image == 0] = -2000
    # image=image-1024
    center = WC  # 40 400//60 300
    width = WW  # 200
    try:
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    except:
        # print(WC[0])
        # print(WW[0])
        center = WC[0]  # 40 400//60 300
        width = WW[0]  # 200
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (win_max - win_min)
    image = image - win_min
    image = np.trunc(image * dFactor)
    image[image > 255] = 255
    image[image < 0] = 0
    image = image / 255  # np.uint8(image)
    # image = (image - 0.5)/0.5
    return image

def caw(mask):
    """
    使用矢量化操作重新赋值3D掩模中的像素值。

    :param mask: 一个三维numpy数组,代表3D掩模。
    :return: 重新赋值后的3D掩模。
    """
    reassigned_mask = np.zeros_like(mask)
    # 数值区间及其对应的赋值
    reassigned_mask[(mask >= 130) & (mask <= 199)] = 1
    reassigned_mask[(mask >= 200) & (mask <= 299)] = 2
    reassigned_mask[(mask >= 300) & (mask <= 399)] = 3
    reassigned_mask[mask >= 400] = 4
    return reassigned_mask

def calcium_Severity(mask_arrayy):
    mask_array=copy.deepcopy(mask_arrayy)
    calciuum_array=np.where(mask_array==2,1,0)
    ca_mea = np.sum(calciuum_array)*0.67*0.67*1.25/3#0.67*0.67*1.25/3=0.187
    if ca_mea > 400:#400 101
        cal_mea = 5
    else:
        cal_mea = 1
    return cal_mea

def remove_common_elements_except_1(list1, list2):

    # 将两个列表转换为集合，并找到交集（不包括 1）
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2) - {1}

    # 使用列表推导式过滤掉相同的元素（除了 1）
    new_list1 = [x for x in list1 if x not in common_elements]
    new_list2 = [x for x in list2 if x not in common_elements]

    return new_list1, new_list2

#statis3：多病变判别
def compute_confusion_matrix_mutil(mask2r,mask2p,ref_mask, pred_mask, ):
    # threshold1 = [4000,4000,4000,4000]  # nor jc dml xz
    # threshold2 = [4000,4000,4000,4000]#0116
    threshold1 = [4000,4000,4000,4000]  # nor jc dml xz
    threshold2 = [4000,500,8000,4000]#0116
    # threshold2 = [14000,48000,64000,6000]#0117
    confusion_matrix = np.zeros((5, 5), dtype=int)
    re_ca=calcium_Severity(mask2r)
    pred_ca=calcium_Severity(mask2p)
    confusion_matrix[re_ca-1,pred_ca-1]=1
    unique_classes_ref = np.unique(ref_mask)
    unique_classes_ref = unique_classes_ref[unique_classes_ref > 0]  # 去除背景类
    ref=[]
    for ref_class in unique_classes_ref:
        region_ref = ref_mask == ref_class
        ss=np.sum(region_ref)
        if ss >= threshold1[ref_class-1]:
            ref.append(ref_class)

    unique_classes_pred = np.unique(pred_mask)
    unique_classes_pred = unique_classes_pred[unique_classes_pred > 0]  # 去除背景类
    pred = []
    for pred_class in unique_classes_pred:
        region_pred = pred_mask == pred_class
        ss=np.sum(region_pred)
        if ss >= threshold2[pred_class - 1]:
            pred.append(pred_class)
            if pred_class in ref and pred_class > 1:#不统计正常的，正常的最后统计
                confusion_matrix[pred_class - 1, pred_class - 1]= 1

    nref,npred=remove_common_elements_except_1(ref,pred)#剔除除1以外的相同类别，剔除1不好统计误诊和漏诊
    classes_setr = set(nref)
    new_refmask = np.where(np.isin(ref_mask, list(classes_setr)), ref_mask, 0)

    classes_setp = set(npred)
    new_predmask = np.where(np.isin(pred_mask, list(classes_setp)), pred_mask, 0)

    #统计漏诊
    for class_id in nref:
        region_mask1 = new_refmask == class_id
        mask2_in_region = new_predmask[region_mask1]
        filtered_mask2 = mask2_in_region[mask2_in_region != 0]
        unique_classes_mask2_in_region, counts = np.unique(filtered_mask2, return_counts=True)
        if len(counts)==0:
            continue

        max_overlap_class = unique_classes_mask2_in_region[np.argmax(counts)]# 找到最大重叠的类别
        # max_overlap_count = np.max(counts)
        confusion_matrix[class_id-1, max_overlap_class-1]= 1
        npred=[x for x in npred if x != max_overlap_class]#

    #统计误诊
    for pred_id in npred:
        region_mask2 = new_predmask == pred_id
        mask1_in_region = new_refmask[region_mask2]
        filtered_mask1 = mask1_in_region[mask1_in_region != 0]
        unique_classes_mask1_in_region, counts = np.unique(filtered_mask1, return_counts=True)
        if len(counts)==0:
            continue

        max_overlap_class = unique_classes_mask1_in_region[np.argmax(counts)]# 找到最大重叠的类别
        # max_overlap_count = np.max(counts)
        confusion_matrix[max_overlap_class-1,pred_id-1]= 1  # 误诊

    if re_ca>1:#严重钙化
        non_zero_positions1 = []
        rows, cols = confusion_matrix.shape
        for i in range(1):## 遍历矩阵，提取非对角且非零元素的位置坐标
            for j in range(cols-1):
                if i != j and confusion_matrix[i, j] != 0:# 检查是否是非对角且非零元素
                    non_zero_positions1.append((i, j))
        confusion_matrix[0,1:4]=0
        for _, j in non_zero_positions1:
            confusion_matrix1 = np.zeros((5, 5), dtype=int)
            confusion_matrix1[4,j]=1
            confusion_matrix[4,4]=0 #需要保证真实该严重钙化数量
            confusion_matrix+=confusion_matrix1

    if pred_ca>1:#严重钙化
        non_zero_positions2 = []
        rows, cols = confusion_matrix.shape
        for i in range(1):##列
            for j in range(rows-1):
                if j != i and confusion_matrix[j, i] != 0:# 检查是否是非对角且非零元素
                    non_zero_positions2.append((j, i))
        confusion_matrix[1:4,0]=0
        for j,_ in non_zero_positions2:
            confusion_matrix1 = np.zeros((5, 5), dtype=int)
            confusion_matrix1[j,4]=1
            confusion_matrix+=confusion_matrix1

    aa = copy.deepcopy(confusion_matrix)
    aa[0, 0] = 0  # 判断其它位置是否有0
    if aa.sum() != 0:
        confusion_matrix[0, 0] = 0  # 保证有病变就不会是健康
    else:
        confusion_matrix[0, 0] = 1
    return confusion_matrix

# statis1，直接对分割结果统计
def confusion_matrix_dis1():
    models=["nnUNetTrainerMednext","nnUNetTrainerUNETRPP","nnUNetTrainerSwinUNETR",
            "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot"]
    # models=["nnUNetTrainer","nnUNetTrainerUMambaBot",
    #         "nnUNetTrainerNnformer", "nnUNetTrainerUNETR", "nnUNetTrainerSegResNet"]
    labels = ["hnnk", "lz", "cq"]
    out = "./p4_confusion/"
    os.makedirs(out, exist_ok=True)
    for model in models:
        pp = "p4testb/"+model
        output_file = out + model + "_p4confusion.txt"
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p4/external/"+label  # hnnk lz cq
            # labelsTs = "/media/bit301/data/yml/data/p3/external/cq/dis/dml/PA248"
            # out_test=labelsTs.replace("hnnk","unet/hnnk")
            confusion_matrix = np.zeros((5, 5), dtype=int)#nor+dml+jc+xz
            labelsTs_list = []
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)
            for path in labelsTs_list:
                read = sitk.ReadImage(path.replace("3.nii.gz","2.nii.gz"), sitk.sitkInt16)
                mask2r = sitk.GetArrayFromImage(read)  # real

                read = sitk.ReadImage(path, sitk.sitkInt16)
                mask3r = sitk.GetArrayFromImage(read)#real
                # mask1=copy.deepcopy(mask11)

                pred3_path = path.replace("p4", pp)  #
                read = sitk.ReadImage(pred3_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask3p = sitk.GetArrayFromImage(read)#predict

                pred2_path=pred3_path.replace("3.nii.gz","2.nii.gz").replace("p4testb","p4testa")
                read = sitk.ReadImage(pred2_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask2p = sitk.GetArrayFromImage(read)#predict

                confusion_matrix += compute_confusion_matrix_mutil(mask2r, mask2p, mask3r,mask3p)  # >400 重度钙化
                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
            # 输出混淆矩阵
            outname = model + "    " + label
            with open(output_file, 'a') as file:
                file.write(outname + "\n")
                file.write(f"{confusion_matrix}\n")
                file.write("\n")
        print("finished "+model)

def confusion_matrix_dis2():
    # models=["nnUNetTrainerMednext1","nnUNetTrainerMednext4b",nnUNetTrainerMednext3b]
    models=["nnUNetTrainerMednext3"]#
    labels = ["hnnk", "lz", "cq"]
    out = "./p4_confusion/"
    os.makedirs(out, exist_ok=True)
    for model in models:
        # pp = "p4testb/"+model
        pp = "p4test/b/" + model
        output_file = out + model + "_p4confusion.txt"
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p4/external/"+label  # hnnk lz cq
            # labelsTs = "/media/bit301/data/yml/data/p3/external/cq/dis/dml/PA248"
            # out_test=labelsTs.replace("hnnk","unet/hnnk")
            confusion_matrix = np.zeros((5, 5), dtype=int)#nor+dml+jc+xz
            labelsTs_list = []
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)
            for path in labelsTs_list:
                read = sitk.ReadImage(path.replace("3.nii.gz","2.nii.gz"), sitk.sitkInt16)
                mask2r = sitk.GetArrayFromImage(read)  # real

                read = sitk.ReadImage(path, sitk.sitkInt16)
                mask3r = sitk.GetArrayFromImage(read)#real
                # mask1=copy.deepcopy(mask11)

                pred3_path = path.replace("p4", pp)  #
                read = sitk.ReadImage(pred3_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask3p = sitk.GetArrayFromImage(read)#predict

                pred2_path=pred3_path.replace("3.nii.gz","2.nii.gz").replace("p4test/b","p4test/a")
                # pred2_path=pred2_path.replace("nnUNetTrainerMednext3b","nnUNetTrainerMednext0")
                read = sitk.ReadImage(pred2_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask2p = sitk.GetArrayFromImage(read)#predict

                confusion_matrix += compute_confusion_matrix_mutil(mask2r, mask2p, mask3r,mask3p)  # >400 重度钙化
                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
            # 输出混淆矩阵
            outname = model + "    " + label
            with open(output_file, 'a') as file:
                file.write(outname + "\n")
                file.write(f"{confusion_matrix}\n")
                file.write("\n")
        print("finished "+model)

def crop_mask(preseg):
    output_size = [1000, 300, 300]  # 统计mask范围得到my=262 mx=217
    # output_size = [1000, 360, 320]
    index = np.nonzero(preseg)
    # 检查是否有非零值
    if index[0].size == 0:
        print("No non-zero values in the mask.")
    else:
        # 计算非零值的范围
        # z_min = np.min(index[0])
        # z_max = np.max(index[0])
        # 统计mask范围得到my=262 mx=217
        y_min = np.min(index[1])
        y_max = np.max(index[1])
        x_min = np.min(index[2])
        x_max = np.max(index[2])
        # z_middle = int((z_min + z_max) / 2)
        y_middle = int((y_min + y_max) / 2)
        x_middle = int((x_min + x_max) / 2)
        crop_y_down = y_middle - int(output_size[1] / 2)
        crop_y_up = y_middle + int(output_size[1] / 2)
        if crop_y_down < 0:
            crop_y_down = 64
            crop_y_up = crop_y_down + output_size[1]
        elif crop_y_up > 512:
            crop_y_up = 448
            crop_y_down = crop_y_up - output_size[1]

        crop_x_down = x_middle - int(output_size[2] / 2)
        crop_x_up = x_middle + int(output_size[2] / 2)
        if crop_x_down < 0:
            crop_x_down = 64
            crop_x_up = crop_x_down + output_size[2]
        elif crop_x_up > 512:
            crop_x_up = 448
            crop_x_down = crop_x_up - output_size[2]
        # if crop_y_down > y_min or crop_y_up < y_max or crop_x_down > x_min or crop_x_up < x_max:

    preseg= preseg[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    return preseg

################ 数据分布统计+依据混淆计算评测结果##############
def dataset_statis():
    labels = ["hnnk", "lz", "cq"]
    # labels = ["cq"]
    for label in labels:
        labelsTs = "/media/bit301/data/yml/data/p4/external/" + label  # hnnk lz cq
        # labelsTs = "/media/bit301/data/yml/data/p3/internal/"  # hnnk lz cq
        labelsTs_list = []
        for root, dirs, files in os.walk(labelsTs, topdown=False):
            for k in range(len(files)):
                path = os.path.join(root, files[k])
                if "3.nii.gz" in path:# and "hnnk" not in path:
                    # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                    labelsTs_list.append(path)
        tref = []
        for path in labelsTs_list:
            read0 = sitk.ReadImage(path.replace("3.nii.gz", "2.nii.gz"), sitk.sitkInt16)
            img_array = sitk.GetArrayFromImage(read0)  # real
            read = sitk.ReadImage(path, sitk.sitkInt16)
            ref_mask = sitk.GetArrayFromImage(read)  # real
            threshold1 = [4000, 4000, 4000, 4000]
            ref = []
            re_ca = calcium_Severity(img_array)
            unique_classes_ref = np.unique(ref_mask)
            unique_classes_ref = unique_classes_ref[unique_classes_ref > 1]  # 去除背景类
            for ref_class in unique_classes_ref:
                region_ref = ref_mask == ref_class
                ss = np.sum(region_ref)
                if ss >= threshold1[ref_class - 1]:
                    ref.append(ref_class)
            if re_ca == 5:
                ref.append(re_ca)  # 钙化
            if len(ref) == 0:  # 没有病变，则为正常
                ref.append(1)  # 如果没有病变，或者没有1
            tref += ref
        count_dict = dict(Counter(tref))
        for key in sorted(count_dict):
            print(f"{key}: {count_dict[key]}")

def parse_confusion_matrix(file_path, ss):
    """
    从 txt 文件中读取多个混淆矩阵，并返回一个字典，键为标签，值为混淆矩阵。
    参数:
    file_path (str): 包含混淆矩阵的 txt 文件路径
    返回:
    dict: 键为标签，值为混淆矩阵的字典
    """
    confusion_matrices = {}
    current_label = None
    current_matrix = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 如果是空行或注释，跳过
        if not line or line.startswith('#'):
            i += 1
            continue
        # 如果遇到新的标签行
        if line.startswith(ss):
            # 如果有未处理的矩阵，先保存
            if current_label is not None and current_matrix:
                confusion_matrices[current_label] = np.array(current_matrix, dtype=int)
                current_matrix = []
            # 提取标签
            parts = line.split()
            current_label = parts[1]
            i += 1
            continue
        # 如果遇到矩阵行
        if line.startswith('[') and line.endswith(']'):
            # 去掉方括号并分割成数字
            row = line.strip('[]').split()
            row = [int(x) for x in row]
            current_matrix.append(row)
            # 如果当前矩阵已经完整（5x5），保存并重置
            if len(current_matrix) == 5:
                confusion_matrices[current_label] = np.array(current_matrix, dtype=int)
                current_matrix = []
                current_label = None
        i += 1
    # 如果文件末尾还有未处理的矩阵，保存
    if current_label is not None and current_matrix:
        confusion_matrices[current_label] = np.array(current_matrix, dtype=int)

    return confusion_matrices

def conf_index(confusion_matrix):
    # 2-TP/TN/FP/FN的计算
    weight = confusion_matrix.sum(axis=0) / confusion_matrix.sum()  ## 求出每列元素的和
    FN = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)  # 所有对的 TP.sum=TP+TN
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    epsilon = 1e-40  # 非常小的偏移量
    TPR = np.where(TP + FN > 0, TP / (TP + FN + epsilon), 0)  # Sensitivity/ hit rate/ recall/ true positive rate
    TNR = np.where(TN + FP > 0, TN / (TN + FP + epsilon), 0)  # Specificity/ true negative rate
    PPV = np.where(TP + FP > 0, TP / (TP + FP + epsilon), 0)  # Precision/ positive predictive value
    NPV = np.where(TN + FN > 0, TN / (TN + FN + epsilon), 0)  # Negative predictive value
    FPR = np.where(TN + FP > 0, FP / (TN + FP + epsilon), 0)  # Fall out/ false positive rate
    FNR = np.where(TP + FN > 0, FN / (TP + FN + epsilon), 0)  # False negative rate
    FDR = np.where(TP + FP > 0, FP / (TP + FP + epsilon), 0)  # False discovery rate
    sub_ACC = np.where(TP+ TN+FP + FN > 0, (TP + TN) / (TP+ TN+FP + FN + epsilon), 0)  # accuracy of each class
    IOU=np.where(TP + FP+ FN > 0, TP / (TP + FP+ FN + epsilon), 0)  # False discovery rate

    average_acc = TP.sum() / (TP.sum() + FN.sum())
    # F1_Score = 2 * TPR * PPV / (PPV + TPR)
    F1_Score = np.where(PPV + TPR > 0, 2 * TPR * PPV / (PPV + TPR + epsilon), 0)
    Macro_F1 = F1_Score.mean()
    weight_F1 = (F1_Score * weight).sum()  # 应该把不同类别给与相同权重,不应该按照数量进行加权把？
    # print('acc:',average_acc)
    # print('Sensitivity:', TPR.mean())#Macro-average方法
    # print('Specificity:', TNR.mean())
    # print('Precision:', PPV.mean())
    # print('Macro_F1:',Macro_F1)
    # 创建一个字典来存储每个类别的评价指标
    metrics = {
        'average_acc': average_acc,
        'Macro_F1': Macro_F1,
        'Sensitivity':TPR.mean(),#模型预测为正类的样本中，实际为正类的比例。
        'Specificity': TNR.mean(),#模型预测为正类的样本中，实际为负类的比例。
        'Precision': PPV.mean(), #预测为正类的样本，模型预测正确的比例
        'IOU': IOU.mean(),
    }

    # 为每个类别创建一个字典，存储其具体的评价指标
    class_metrics = {}
    for i in range(len(TP)):
        class_metrics[f'Class_{i}'] = {
            'sub_ACC': sub_ACC[i],
            'F1_Score': F1_Score[i],
            'Sensitivity': TPR[i],
            'Specificity': TNR[i],
            'Precision': PPV[i],
            'IOU': IOU[i],
        }

    # return average_acc, TPR.mean(), TNR.mean(), PPV.mean(), Macro_F1
    return metrics, class_metrics

def confuse_plot(cm, save_path):
    save_path += ".tif"
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap='Blues')#Blues Oranges

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel('Value', rotation=-90, va="bottom", fontsize=11)  # 设置颜色条标题的字号

    # 调整颜色条的刻度标签字体大小
    cbar.ax.tick_params(labelsize=11)

    # 显示数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > (cm.max() / 2.) else "black",
                    fontsize=11)  # 设置数值的字号

    # 设置坐标轴标签
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(['Normal', 'Aneurysm', 'Dissection', 'Stenosis', 'Calcification'], fontsize=11)
    ax.set_yticklabels(['Normal', 'Aneurysm', 'Dissection', 'Stenosis', 'Calcification'], fontsize=11)
    # ax.set_xticklabels(['Type 1', 'Type 2', 'Type 3','Type 4','Type 5'], fontsize=11)  # 设置X轴标签字号
    # ax.set_yticklabels(['Type 1', 'Type 2', 'Type 3','Type 4','Type 5'], fontsize=11)  # 设置Y轴标签字号

    # 旋转顶部的标签,避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=11)  # 设置X轴刻度字号

    # 设定底部和右侧的边框不可见
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 设定底部和左侧的边框线宽
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 调整子图布局,防止坐标标签被截断
    plt.tight_layout()
    plt.savefig(save_path,dpi=300, format='tif')
    # plt.savefig(save_path, dpi=600, format='tif')
    # plt.show()
    plt.close(fig)

def confusion_matrics():
    # models=["nnUNetTrainerMednext1","nnUNetTrainerMednext4b",nnUNetTrainerMednext3b]
    models=["nnUNetTrainerMednext3"]#
    path = "./p4_confusion/"
    out_put = path +"Confuse_disp"
    if not os.path.isdir(out_put):
        os.makedirs(out_put)
    for model in models:
        confusion_path = path + model + "_p4confusion.txt"
        file_path = path + "Metrics/"+ model + "metrics.txt"
        folder_path = os.path.dirname(file_path)
        os.makedirs(folder_path, exist_ok=True)
        matrices = parse_confusion_matrix(confusion_path, model)
        for label, matrix in matrices.items():
            save_path = os.path.join(out_put, model +label)  # mm="matrix1b_m"
            confuse_plot(matrix, save_path)
            metrics, class_metrics = conf_index(matrix)
            with open(file_path, 'a') as file:
                file.write(label + "\n")
                file.write("Overall Metrics:\n")
                for key, value in metrics.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")

                # 打印每个类别的指标
                file.write("\nClass Metrics:\n")
                for class_name, class_metric in class_metrics.items():
                    file.write(f"{class_name}:")
                    for key, value in class_metric.items():
                        file.write(f"  {key}: {value}\n")
                    file.write("\n")
        print("finished " + model)

#Bland-Altman
def dot_plot(data,save_path):
    gd="True "+save_path.split("_")[-1]
    mea="Predict "+save_path.split("_")[-1]+" from NCCT"
    save_path=save_path+".tif"
    data=np.array(data)
    # true_values = data[:, 0]  # 请替换为实际真实值数组
    # predicted_values = data[:, 1]
    # errors = np.abs(predicted_values - true_values)
    true_values = data[:, 0]  # 请替换为实际真实值数组
    predicted_values = data[:, 1]
    errors = data[:, 2]
    fig, ax = plt.subplots()
    scatter = ax.scatter(true_values, predicted_values, c=errors, cmap='viridis', s=20, alpha=0.8)
    cbar = fig.colorbar(scatter, ax=ax, label='Mean Absolute Percentage Error (%)')
    plt.plot([np.nanmin(true_values), np.nanmax(true_values)],
             [np.nanmin(true_values), np.nanmax(true_values)],
             'r--', label='Perfect reconstruction line')
    ax.set_xlabel(gd, fontsize=10)
    ax.set_ylabel(mea, fontsize=10)
    plt.legend()
    ax.set_title('Predict vs True with Mean Absolute Percentage Error', fontsize=10)
    # plt.savefig(save_path)  # 矢量图
    plt.savefig(save_path, dpi=600)
    # plt.show()
    plt.close(fig)# 显式关闭当前figure

if __name__ == '__main__':
    # dataset_statis()
    # confusion_matrix_dis1()
    confusion_matrix_dis2()
    confusion_matrics()
    # disp()

