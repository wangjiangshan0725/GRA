from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.ops import build_plugin_layer
from mmdet.utils import get_root_logger
# from ..builder import BACKBONES
from copy import deepcopy
# from .modules import AdaptiveRotatedConv2d, RountingFunction, AdaptiveRotatedConv2d_multichannel,AdaptiveRotatedConv2d_multichannel_baseline,AdaptiveRotatedConv2d_multichannel_fast,AdaptiveRotatedConv2d_multichannel_fast_2,AdaptiveRotatedConv2d_multichannel_fast_twoside, AdaptiveRotatedConv2d_multichannel_fast_20231024,AdaptiveRotatedConv2d_multichannel_fast_lan

from torch.nn import functional as F
import math
import einops
from calflops import calculate_flops

# from .weight_init import trunc_normal_


class LayerNormProxy(nn.Module):
    # copy from https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    

class RountingFunction(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        # trunc_normal_(self.dwc.weight, std=.02)
        # trunc_normal_(self.fc_alpha.weight, std=.02)
        # trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return alphas, angles

    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)

def _get_rotation_matrix(thetas):
    bs, g = thetas.shape
    device = thetas.device
    thetas = thetas.reshape(-1)  # [bs, n] --> [bs x n]
    
    x = torch.cos(thetas)
    y = torch.sin(thetas)
    x = x.unsqueeze(0).unsqueeze(0)  # shape = [1, 1, bs * g]
    y = y.unsqueeze(0).unsqueeze(0)
    a = x - y
    b = x * y
    c = x + y

    rot_mat_positive = torch.cat((
        torch.cat((a, 1-a, torch.zeros(1, 7, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), x-b, b, torch.zeros(1, 1, bs*g, device=device), 1-c+b, y-b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device), 1-a, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((b, y-b, torch.zeros(1,1 , bs*g, device=device), x-b, 1-c+b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-c+b, x-b, torch.zeros(1, 1, bs*g, device=device), y-b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), 1-a, torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), y-b, 1-c+b, torch.zeros(1, 1, bs*g, device=device), b, x-b, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs*g, device=device), 1-a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative = torch.cat((
        torch.cat((c, torch.zeros(1, 2, bs*g, device=device), 1-c, torch.zeros(1, 5, bs*g, device=device)), dim=1),
        torch.cat((-b, x+b, torch.zeros(1, 1, bs*g, device=device), b-y, 1-a-b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), 1-c, c, torch.zeros(1, 6, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), x+b, 1-a-b, torch.zeros(1, 1, bs*g, device=device), -b, b-y, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), b-y, -b, torch.zeros(1, 1, bs*g, device=device), 1-a-b, x+b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs*g, device=device), c, 1-c, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-a-b, b-y, torch.zeros(1, 1, bs*g, device=device), x+b, -b), dim=1),
        torch.cat((torch.zeros(1, 5, bs*g, device=device), 1-c, torch.zeros(1, 2, bs*g, device=device), c), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    mask = (thetas >= 0).unsqueeze(0).unsqueeze(0)
    mask = mask.float()                                                   # shape = [1, 1, bs*g]
    rot_mat = mask * rot_mat_positive + (1 - mask) * rot_mat_negative     # shape = [k*k, k*k, bs*g]
    rot_mat = rot_mat.permute(2, 0, 1)                                    # shape = [bs*g, k*k, k*k]
    rot_mat = rot_mat.reshape(bs, g, rot_mat.shape[1], rot_mat.shape[2])  # shape = [bs, g, k*k, k*k]
    return rot_mat


def batch_rotate_multiweight_fast(weights, lambdas, thetas):
    """
    Let
        batch_size = b
        kernel_number = 1
        kernel_size = 3
        num_group = g
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
    Return:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    assert(thetas.shape == lambdas.shape)

    weights = weights.squeeze(0) #[Cout, Cin, k, k]
    
    batch_size = thetas.shape[0]
    num_groups = thetas.shape[1]
    k = weights.shape[-1]
    
    Cout, Cin, _, _ = weights.shape


    rotation_matrix = _get_rotation_matrix(thetas)#b,num_groups,9,9


    lambdas = lambdas.unsqueeze(2).unsqueeze(3) #[b,num_groups,1,1] 
    
    rotation_matrix = torch.mul(rotation_matrix, lambdas) #b,num_groups,9,9

    # [1, num_groups,Cout//num_groups,Cin, 9, 1]
    # [b, num_groups,     1          , 1,   9, 9]
    B = weights.view( 1, Cout//num_groups,num_groups, Cin, 9, 1).permute(0,2,1,3,4,5)
    A = rotation_matrix.view(batch_size, num_groups,     1          , 1,   9, 9)
    
    a, b, _, _, c, d=A.shape
    _, b, e, f, d, _=B.shape
    A=A.view(a,b,c,d).permute(1,0,2,3).reshape(b, a*c, d)
    B=B.view(b,e,f,d).permute(0,3,1,2).reshape(b, d, e*f)
    result = torch.bmm(A, B).reshape(b,a,c,e,f).permute(1,0,3,4,2)

    weights = result.reshape(batch_size * Cout, Cin, 3, 3)
    
    return weights

def batch_rotate_multiweight(weights, lambdas, thetas):
    """
    Let
        batch_size = b
        kernel_number = n
        kernel_size = 3
        num_group = g
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
    Return:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    assert(thetas.shape == lambdas.shape)
    assert(lambdas.shape[1] == weights.shape[0])

    b = thetas.shape[0]
    n = thetas.shape[1]
    k = weights.shape[-1]
    _, Cout, Cin, _, _ = weights.shape

    # Stage 1:
    # input: thetas: [b, n]
    #        lambdas: [b, n]
    # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

    #       Sub_Stage 1.1:
    #       input: [b, n] kernel
    #       output: [b, n, 9, 9] rotation matrix
    rotation_matrix = _get_rotation_matrix(thetas)

    #       Sub_Stage 1.2:
    #       input: [b, n, 9, 9] rotation matrix
    #              [b, n] lambdas
    #          --> [b, n, 1, 1] lambdas
    #          --> [b, n, 1, 1] lambdas dot [b, n, 9, 9] rotation matrix
    #          --> [b, n, 9, 9] rotation matrix with gate (done)
    #       output: [b, n, 9, 9] rotation matrix with gate
    lambdas = lambdas.unsqueeze(2).unsqueeze(3)
    
    rotation_matrix = torch.mul(rotation_matrix, lambdas) #b,n=1,9,9  n=1,cout,cin,3,3

    #       Sub_Stage 1.3: Reshape
    #       input: [b, n, 9, 9] rotation matrix with gate
    #       output: [b*9, n*9] rotation matrix with gate
    rotation_matrix = rotation_matrix.permute(0, 2, 1, 3)
    rotation_matrix = rotation_matrix.reshape(b*9, n*9)

    # Stage 2: Reshape 
    # input: weights: [n, Cout, Cin, 3, 3]
    #             --> [n, 3, 3, Cout, Cin]
    #             --> [n*9, Cout*Cin] done
    # output: weights: [n*9, Cout*Cin]
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.contiguous().view(n*9, Cout*Cin)


    # Stage 3: torch.mm
    # [b*9, n*9] x [n*9, Cout*Cin]
    # --> [b*9, Cout*Cin]
    weights = torch.mm(rotation_matrix, weights)

    # Stage 4: Reshape Back
    # input: [b*9, Cout*Cin]
    #    --> [b, 3, 3, Cout, Cin]
    #    --> [b, Cout, Cin, 3, 3]
    #    --> [b * Cout, Cin, 3, 3] done
    # output: [b * Cout, Cin, 3, 3]
    weights = weights.contiguous().view(b, 3, 3, Cout, Cin)
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.reshape(b * Cout, Cin, 3, 3)

    return weights

class AdaptiveRotatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.rounting_func = rounting_func
        self.rotate_func = rotate_func

        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        alphas, angles = self.rounting_func(x)

        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        rotated_weight = self.rotate_func(self.weight, alphas, angles)

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        
        # reshape back
        out = out.reshape(bs, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
             ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
             
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
    
class AdaptiveRotatedConv2d_multichannel_fast(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight_fast, num_groups=1):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.rounting_func = rounting_func
        self.rotate_func = rotate_func
        self.num_groups = num_groups

        self.weight = nn.Parameter(
            torch.Tensor(
                1, 
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
            )
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        alphas, angles = self.rounting_func(x)

        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        rotated_weight = self.rotate_func(self.weight, alphas, angles)

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        
        # reshape back
        out = out.reshape(bs, self.out_channels, *out.shape[2:])
        return out
    
class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 n_stage=None,
                 replace=None, 
                 kernel_number=None,
                 num_divide=1,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    n_stage=n_stage,
                    n_block=0,
                    replace=replace,
                    num_divide=num_divide,
                    kernel_number=kernel_number,
                    **kwargs))
            inplanes = planes * block.expansion
            for idx_block in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        n_stage=n_stage,
                        n_block=idx_block,
                        num_divide=num_divide,
                        replace=replace,
                        kernel_number=kernel_number,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for idx_block in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        n_stage=n_stage,
                        n_block=idx_block,
                        num_divide=num_divide,
                        replace=replace,
                        kernel_number=kernel_number,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    n_stage=n_stage,
                    n_block=(num_blocks-1),
                    num_divide=num_divide,
                    replace=replace,
                    kernel_number=kernel_number,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 num_divide=1):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function"""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 n_stage=None,
                 n_block=None,
                 replace=None,
                 kernel_number=1,
                 num_groups=4,
                 num_groups_1=1,
                 num_groups_2=1,
                 num_divide=1):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.n_stage = n_stage
        self.n_block = n_block
        self.replace = replace
        self.kernel_number = kernel_number
        self.num_groups = num_groups
        self.num_groups_1 = num_groups_1
        self.num_groups_2 = num_groups_2
        
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        # construct conv3x3
        if str(self.n_block) not in self.replace:
            # original conv3x3
            fallback_on_stride = False
            if self.with_dcn:
                fallback_on_stride = dcn.pop('fallback_on_stride', False)
            if not self.with_dcn or fallback_on_stride:
                self.conv2 = build_conv_layer(
                    conv_cfg,
                    planes,
                    planes,
                    kernel_size=3,
                    stride=self.conv2_stride,
                    padding=dilation,
                    dilation=dilation,
                    bias=False)
            else:
                assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
                self.conv2 = build_conv_layer(
                    dcn,
                    planes,
                    planes,
                    kernel_size=3,
                    stride=self.conv2_stride,
                    padding=dilation,
                    dilation=dilation,
                    bias=False)
        else:
            # adaptive rotated convolution
            print("!!!!!!!!!!!!!!!!!!!in else num_divide!!!!!!!!!!!!!!!!!!!!!",num_divide)
            if num_divide==0:
                self.conv2 = AdaptiveRotatedConv2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3, 
                    stride=self.conv2_stride,
                    padding=dilation,
                    groups=1,
                    dilation=dilation,
                    rounting_func=RountingFunction(
                        in_channels=planes,
                        kernel_number=self.kernel_number,
                    ),
                    kernel_number=self.kernel_number,
                ) 
            elif num_divide==1000:
                self.conv2 = AdaptiveRotatedConv2d_multichannel(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3, 
                    stride=self.conv2_stride,
                    padding=dilation,
                    groups=1,
                    dilation=dilation,
                    rounting_func=RountingFunction(
                        in_channels=planes,
                        kernel_number=self.num_groups,
                    ),
                    kernel_number=self.kernel_number,
                    num_groups=self.num_groups
                )
            elif num_divide==20231021:
                self.conv2 = AdaptiveRotatedConv2d_multichannel_fast(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3, 
                    stride=self.conv2_stride,
                    padding=dilation,
                    groups=1,
                    dilation=dilation,
                    rounting_func=RountingFunction(
                        in_channels=planes,
                        kernel_number=self.num_groups,
                    ),
                    kernel_number=self.kernel_number,
                    num_groups=self.num_groups
                )
            elif num_divide==20231023:
                self.conv2 = AdaptiveRotatedConv2d_multichannel_fast_twoside(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3, 
                    stride=self.conv2_stride,
                    padding=dilation,
                    groups=1,
                    dilation=dilation,
                    rounting_func=RountingFunction(
                        in_channels=planes,
                        kernel_number=self.num_groups_1 * self.num_groups_2,
                    ),
                    kernel_number=self.kernel_number,
                    num_groups_1=self.num_groups_1,
                    num_groups_2=self.num_groups_2
                )
            elif num_divide==20231024:
                self.conv2 = AdaptiveRotatedConv2d_multichannel_fast_20231024(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3, 
                    stride=self.conv2_stride,
                    padding=dilation,
                    groups=1,
                    dilation=dilation,
                    rounting_func=RountingFunction(
                        in_channels=planes,
                        kernel_number=self.num_groups,
                    ),
                    kernel_number=self.kernel_number,
                    num_groups=self.num_groups
                )
            elif num_divide==10212:
                self.conv2 = AdaptiveRotatedConv2d_multichannel_fast_2(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3, 
                    stride=self.conv2_stride,
                    padding=dilation,
                    groups=1,
                    dilation=dilation,
                    rounting_func=RountingFunction(
                        in_channels=planes,
                        kernel_number=self.num_groups,
                    ),
                    kernel_number=self.kernel_number,
                    num_groups=self.num_groups
                )
            elif num_divide==-1:
                self.conv2 = AdaptiveRotatedConv2d_multichannel_baseline(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3, 
                    stride=self.conv2_stride,
                    padding=dilation,
                    groups=1,
                    dilation=dilation,
                    rounting_func=RountingFunction(
                        in_channels=planes,
                        kernel_number=self.num_groups,
                    ),
                    kernel_number=self.kernel_number,
                    num_groups=self.num_groups
                )
            elif num_divide==-2:
                self.conv2 = build_conv_layer(
                    conv_cfg,
                    planes,
                    planes,
                    kernel_size=3,
                    stride=self.conv2_stride,
                    padding=dilation,
                    dilation=dilation,
                    bias=False)
            else:
                self.conv2 = nn.ModuleList([AdaptiveRotatedConv2d(
                    in_channels=planes,
                    out_channels=planes//num_divide,
                    kernel_size=3, 
                    stride=self.conv2_stride,
                    padding=dilation,
                    groups=1,
                    dilation=dilation,
                    rounting_func=RountingFunction(
                        in_channels=planes,
                        kernel_number=self.kernel_number,
                    ),
                    kernel_number=self.kernel_number,
                ) for _ in range(num_divide)])

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """ make plugins for block

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.

        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function"""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            
            if isinstance(self.conv2, nn.ModuleList):
                out_list=[]
                for single_conv in self.conv2:
                    out_list.append(single_conv(out)) #bs,cout/num,h,w
                out = torch.cat(out_list,dim=1)
            else:
                out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# @BACKBONES.register_module()
class ARCResNet_v2_test(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 replace=None,
                 kernel_number=None,
                 num_divide=1,
                 num_groups=4,
                 num_groups_1=1,
                 num_groups_2=1
                 ):
        super(ARCResNet_v2_test, self).__init__()
        
        self.replace = replace
        self.kernel_number = kernel_number

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                n_stage=i,
                replace=self.replace[i], 
                kernel_number=self.kernel_number,
                num_divide=num_divide,
                num_groups=num_groups,
                num_groups_1=1,
                num_groups_2=1
                )
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """ make plugins for ResNet 'stage_idx'th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.
        An example of plugins format could be:

        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function"""
        if self.deep_stem:
            x = self.stem(x)
        else:
            print("!here!")
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        # print("!!!x!!!",x.shape)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            # print("!!!x!!!",x.shape)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed"""
        super(ARCResNet_v2_test, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


from torchinfo import summary
from ptflops import get_model_complexity_info   


# class Conv1(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.conv1=nn.Conv2d(3, 64, kernel_size=3, bias=False)

#     def forward(self, x):
#         out=self.conv1(x)
#         return out


input_shape = (1, 3, 1024, 1024)
# conv1=Conv1()
# flops, macs, params = calculate_flops(model=conv1, 
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)
# print("conv1 FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))




import torchvision
num_groups=4
print("num_groups:",num_groups)
model = ARCResNet_v2_test(
    depth=50,
    replace = [
            ['x'],
            ['0', '1', '2', '3'],
            ['0', '1', '2', '3', '4', '5'],
            ['0', '1', '2']
        ],
    kernel_number = 1,
    num_groups = num_groups,
    num_divide = 20231021
)

flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(1, 3, 1024, 1024),
                                      output_as_string=True,
                                      output_precision=4)
print("model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# summary(model, input_size=input_shape)
# print(model)
# -2: 171.508 / 85.41
# 1: 172.391 / 85.759
# 2: 


# total = sum([param.nelement() for param in model.parameters()])
# print("parameter:%fM" % (total/1e6)) 
# # print(model)
# flops, macs, params = calculate_flops(model=model.layer1[0].conv2 , 
#                                       input_shape=(1, 64, 56, 56),
#                                       output_as_string=True,
#                                       output_precision=4)
# print("conv1 FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# summary(model, input_size=input_shape)
# print(model)
# flops, macs, params = calculate_flops(model=model.layer1[0].conv2 , 
#                                       input_shape=(1, 64, 56, 56),
#                                       output_as_string=True,
#                                       output_precision=4)
# print("conv2 FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# flops, macs, params = calculate_flops(model=model.layer1[0].conv3 , 
#                                       input_shape=(1, 64, 56, 56),
#                                       output_as_string=True,
#                                       output_precision=4)
# print("conv3 FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# flops, macs, params = calculate_flops(model=model.layer1[0].downsample , 
#                                       input_shape=(1, 64, 56, 56),
#                                       output_as_string=True,
#                                       output_precision=4)
# print("downsample FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

# print(summary(model, input_size=input_shape))

# flops, params = get_model_complexity_info(model, (3, 1024, 1024), as_strings=True, print_per_layer_stat=True)

# 


# model_bl = ARCResNet_v2_test(
#     depth=50,
#     replace = [
#             ['x'],
#             ['0', '1', '2', '3'],
#             ['0', '1', '2', '3', '4', '5'],
#             ['0', '1', '2']
#         ],
#     kernel_number = 4,
#     num_groups = 4,
#     num_divide = 0
# )
# input_shape = (1, 3, 1024, 1024)
# print(model_bl)
# print(summary(model_bl, input_size=input_shape))


# flops, macs, params = calculate_flops(model=model, 
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)
# print("ARCResNet_v2_test FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# flops, macs, params = calculate_flops(model=model_bl, 
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)
# print("ARCResNet_v2_test baseline FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))




# print("ARCResNet_v2_test",model)
# # model_bl = ARCResNet_v2_test(
#     depth=50,
#     replace = [
#             ['x'],
#             ['0', '1', '2', '3'],
#             ['0', '1', '2', '3', '4', '5'],
#             ['0', '1', '2']
#         ],
#     kernel_number = 16,
#     num_groups = 4,
#     num_divide = 0
# )

# tensor = torch.rand(1, 3, 1024, 1024)
# model(tensor)
# # 分析FLOPs
# flops = FlopCountAnalysis(model, tensor)
# print("FLOPs: ", flops.total())

#4:
#87083937792
#87386517504

#16:
#87084023808
#88596922368


###################################################

#4:
#4263163392
#4565743104

#8:
#4263192064
#4969211392

#16:
#4263249408
#5776147968