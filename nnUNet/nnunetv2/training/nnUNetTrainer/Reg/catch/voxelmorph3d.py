import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
use_gpu = torch.cuda.is_available()

class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm3d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm3d(out_channels),
                    torch.nn.ReLU()
                )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm3d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm3d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm3d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)
        return  final_layer

class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu==True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y):

        im = F.pad(im, (0,0,1,1,1,1,0,0))

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1,0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1,0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1,0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1,0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1,0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1,0)
        wc = ((1-dx) * dy).transpose(1,0)
        wd = ((1-dx) * (1-dy)).transpose(1,0)

        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)

class VoxelMorph3d(nn.Module):
    def __init__(self, in_channels=1, use_gpu=False):
        super(VoxelMorph3d, self).__init__()
        self.unet = UNet(in_channels, 3)
        self.spatial_transform = SpatialTransformation(use_gpu)
        if use_gpu:
            self.unet = self.unet.cuda()
            self.spatial_transform = self.spatial_transform.cuda()

    def forward(self, moving_image, fixed_image):
        x = torch.cat([moving_image, fixed_image], dim=3).permute(0,3,1,2)
        deformation_matrix = self.unet(x).permute(0,2,3,1)
        registered_image = self.spatial_transform(moving_image, deformation_matrix)
        # return registered_image,deformation_matrix
        ret = {'moved_vol': registered_image, 'pos_flow': deformation_matrix}
        return ret


available_activations = {'ReLU': nn.ReLU,
                         'LeakyReLU': nn.LeakyReLU}

def get_activation_function(act):
    """
    Get an activation function
    :param act:
    :return:
    """
    if act in available_activations:
        return available_activations[act]
    else:
        NotImplementedError(
            "Not Implemented activation type {}, only {} are available now".format(act, available_activations.keys()))

class convBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=False, batchnorm=False, act=nn.ReLU, residual=False, ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param act:
        """

        super(convBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinear = get_activation_function(act) if type(act) is str else act
        self.residual = residual

    def forward(self, x):
        x= self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.nonlinear:
            x = self.nonlinear()(x)
        if self.residual:
            x += x

        return x

def get_identity_transform(size, normalize=True):
    """

    :param size: D,H,W size
    :param normalize:
    :return: 3XDxHxW tensor
    """

    if normalize:
        xx, yy, zz = torch.meshgrid([torch.arange(0, size[k]).float() / (size[k] - 1) * 2.0 - 1 for k in [0, 1, 2]])
    else:
        xx, yy, zz = torch.meshgrid([torch.arange(0, size[k]) for k in [0, 1, 2]])
    _identity = torch.stack([zz, yy, xx])
    return _identity

def get_identity_transform_batch(size, normalize=True):
    """
    generate an identity transform for given image size (NxCxDxHxW)
    :param size: Batch, D,H,W size
    :param normalize: normalized index into [-1,1]
    :return: identity transform with size Nx3xDxHxW
    """
    _identity = get_identity_transform(size[2:], normalize)
    return _identity

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return nnf.grid_sample(src, new_locs, mode=self.mode)

class VoxelMorphCVPR2018(nn.Module):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param input_channel: channels of input data (2 for a pair of images)
    :param output_channel: channels of output data (3 for 3D registration)
    :param enc_filters: list of encoder filters. values represent the number of filters of each layer
           e.g. (16, 32, 32, 32, 32)
    :param dec_filters: list of decoder filters.
    """
    def __init__(self, inshape,input_channel=2, output_channel=3, enc_filters=(16, 32, 32, 32, 32),
                 dec_filters=(32, 32, 32, 8, 8)):
        super(VoxelMorphCVPR2018, self).__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.enc_filters = enc_filters
        self.dec_filters = dec_filters

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsampling = nn.Upsample(scale_factor=2, mode="trilinear")

        for i in range(len(enc_filters)):
            if i == 0:
                self.encoders.append(convBlock(input_channel, enc_filters[i], stride=1, bias=True))
            else:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=2, bias=True))

        for i in range(len(dec_filters)):
            if i == 0:
                self.decoders.append(convBlock(enc_filters[-1], dec_filters[i], stride=1, bias=True))
            elif i < 4:
                self.decoders.append(convBlock(dec_filters[i-1] if i == 4 else dec_filters[i - 1] + enc_filters[4-i],
                                            dec_filters[i], stride=1, bias=True))
            else:
                self.decoders.append(convBlock(dec_filters[i-1], dec_filters[i], stride=1, bias=True))

        self.flow = nn.Conv3d(dec_filters[-1] + enc_filters[0], output_channel, kernel_size=3, stride=1, padding=1, bias=True)

        # identity transform for computing displacement
        self.id_transform = None
        self.spatial_transform = SpatialTransformer(inshape)

    def forward(self, source, target):


        x_enc_1 = self.encoders[0](torch.cat((source, target), dim=1))
        # del input
        x_enc_2 = self.encoders[1](x_enc_1)
        x_enc_3 = self.encoders[2](x_enc_2)
        x_enc_4 = self.encoders[3](x_enc_3)
        x_enc_5 = self.encoders[4](x_enc_4)

        x_dec_1 = self.decoders[0](F.interpolate(x_enc_5, size=x_enc_4.shape[2:]))
        del x_enc_5
        x_dec_2 = self.decoders[1](F.interpolate(torch.cat((x_dec_1, x_enc_4), dim=1), size=x_enc_3.shape[2:]))
        del x_dec_1, x_enc_4
        x_dec_3 = self.decoders[2](F.interpolate(torch.cat((x_dec_2, x_enc_3), dim=1), size=x_enc_2.shape[2:]))
        del x_dec_2, x_enc_3
        x_dec_4 = self.decoders[3](torch.cat((x_dec_3, x_enc_2), dim=1))
        del x_dec_3, x_enc_2
        x_dec_5 = self.decoders[4](F.interpolate(x_dec_4, size=x_enc_1.shape[2:]))
        del x_dec_4
        disp_field = self.flow(torch.cat((x_dec_5, x_enc_1), dim=1))
        del x_dec_5, x_enc_1
        warped_source = self.spatial_transform(source, disp_field)
        # if self.id_transform is None:
        #     self.id_transform = get_identity_transform_batch(source.shape).to(disp_field.device)
        #
        # deform_field = disp_field + self.id_transform
        # # transform images
        # warped_source = F.grid_sample(source, grid=deform_field.permute([0,2,3,4,1]), mode='bilinear',
        #                               padding_mode='zeros', align_corners=True)
        ret = {'moved_vol': warped_source, 'pos_flow': disp_field}
        return ret #disp_field, warped_source, deform_field


    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()

def cross_correlation_loss(I, J, n):
    I = I.permute(0, 3, 1, 2)
    J = J.permute(0, 3, 1, 2)
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    if use_gpu:
        sum_filter = sum_filter.cuda()
    I_sum = torch.conv2d(I, sum_filter, padding=1, stride=(1,1))
    J_sum = torch.conv2d(J, sum_filter,  padding=1 ,stride=(1,1))
    I2_sum = torch.conv2d(I2, sum_filter, padding=1, stride=(1,1))
    J2_sum = torch.conv2d(J2, sum_filter, padding=1, stride=(1,1))
    IJ_sum = torch.conv2d(IJ, sum_filter, padding=1, stride=(1,1))
    win_size = n**2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
    cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
    return torch.mean(cc)

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d/2.0

def vox_morph_loss(y, ytrue, n=9, lamda=0.01):
    cc = cross_correlation_loss(y, ytrue, n)
    sm = smooothing_loss(y)
    #print("CC Loss", cc, "Gradient Loss", sm)
    loss = -1.0 * cc + lamda * sm
    return loss

def dice_score(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    top = 2 *  torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    #print("Dice score", dice)
    return dice
