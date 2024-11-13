import os
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import datetime
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
# import voxelmorph as vxm  # nopep8, from packages instead of source code
# from voxelmorph.torch.layers import SpatialTransformer
import sys
sys.path.append(r"/media/bit301/data/yml/project/python39/p2/nnUNet/nnunetv2/training/nnUNetTrainer/Reg")

import losses as src_loss
from losses import combined_loss
import networks as networks
import utils as utils
from TransMorph import CONFIGS as CONFIGS_TM
import TransMorph as TransMorph


def parse_arguments():
    parser = argparse.ArgumentParser()
    # network architecture parameters
    parser.add_argument('--enc', type=int, default=[16, 32, 32, 32], nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, default=[32, 32, 32, 32, 32, 16, 16], nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=1,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
    parser.add_argument('--model', type=str, default='vm-feat', help='Choose a model to train (vm, vm-feat, mm, mm-feat)')

    # 创建一个命名空间对象，使用parser中定义的默认值
    args = parser.parse_args([])
    return args

def parse_arguments11():
    # 创建一个命名空间对象来保存参数
    args = argparse.Namespace()
    # network architecture parameters
    args.enc = [16, 32, 32, 32]
    args.dec = [32, 32, 32, 32, 32, 16, 16]
    args.int_steps = 7
    args.int_downsize = 2
    args.bidir = False  # 默认为False，如果需要可以改为True
    args.model = 'vm-feat'
    # 返回填充好的命名空间对象
    return args

def create_model(inshape,args):
    bidir = args.bidir
    # inshape = (96, 160, 160)
    # unet architecture
    enc_nf = args.enc if args.enc else [16] * 4
    dec_nf = args.dec if args.dec else [16] * 6

    # Define a model
    if args.model == 'vm': #VoxelMorph
        model = networks.VxmDense(
                inshape=inshape,
                nb_unet_features=[enc_nf, dec_nf],
                bidir=bidir,
                int_steps=args.int_steps,
                int_downsize=args.int_downsize,
            )
    elif args.model == 'vm-feat':
        model = networks.VxmFeat(
                inshape=inshape,
                nb_feat_extractor=[[16] * 2, [16] * 4],
                nb_unet_features=[enc_nf, dec_nf],
                bidir=bidir,
                int_steps=args.int_steps,
                int_downsize=args.int_downsize,
            )
    elif args.model == 'tm':
        config = CONFIGS_TM['TransMorph']
        model = TransMorph.TransMorph(config)
    elif args.model == 'tm-feat':
        config = CONFIGS_TM['TransMorph']
        model = TransMorph.TransMorphFeat(config)
    elif args.model == 'mm':
        config = CONFIGS_TM['MambaMorph']
        model = TransMorph.MambaMorph(config)
    elif args.model == 'mm-feat':
        config = CONFIGS_TM['MambaMorph']
        model = TransMorph.MambaMorphFeat(config)
    elif args.model == 'vimm':
        config = CONFIGS_TM['VMambaMorph']
        model = TransMorph.VMambaMorph(config)
    elif args.model == 'vimm-feat':
        config = CONFIGS_TM['VMambaMorph']
        model = TransMorph.VMambaMorphFeat(config)
    elif args.model == 'rvm':
        config = CONFIGS_TM['VMambaMorph']
        model = TransMorph.RecVMambaMorphFeat(config)
    return model

def Reg3D(inshape):
    args = parse_arguments()
    model = create_model(inshape,args)
    return model
