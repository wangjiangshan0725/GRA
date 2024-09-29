import torch
import torch.nn as nn
from torch.nn import functional as F


__all__ = ['AdaptiveRotatedConv2d','AdaptiveRotatedConv2d_multichannel','AdaptiveRotatedConv2d_multichannel_baseline','AdaptiveRotatedConv2d_multichannel_fast','AdaptiveRotatedConv2d_multichannel_fast_2','AdaptiveRotatedConv2d_multichannel_fast_twoside','AdaptiveRotatedConv2d_multichannel_fast_20231024']


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

def batch_rotate_multiweight_one_channel(weights, lambdas, thetas):
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

    weights = weights.contiguous().view(b, 3, 3, Cout, Cin)
    weights = weights.permute(0, 3, 4, 1, 2)

    return weights

def batch_rotate_multiweight_multi_channel(weights, lambdas, thetas, num_groups):
    """
    Let
        batch_size = b
        kernel_number = n
        kernel_size = 3
        num_group = g
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
    """ 
    # lambdas: [b, n]
    b = thetas.shape[0]
    kernel_number, Cout, Cin, k, k = weights.shape
    weights = weights.reshape(kernel_number, Cout//num_groups, num_groups, Cin, k, k).permute(2, 0, 1, 3, 4, 5)

    final_weight=[]

    
    for idx, weight in enumerate(weights):
        final_weight.append(batch_rotate_multiweight_one_channel(weight, lambdas[:,idx].unsqueeze(1), thetas[:,idx].unsqueeze(1)))
    
    final_weights = torch.cat(final_weight, dim=1)
    final_weights = final_weights.reshape(b * Cout, Cin, 3, 3)
    # print("!!!final_weights!!!!",final_weights.shape)
    return final_weights


def batch_rotate_multiweight_multi_channel_baseline(weights, lambdas, thetas, num_groups):
    """
    Let
        batch_size = b
        kernel_number = n
        kernel_size = 3
        num_group = g
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
    """ 
    # lambdas: [b, n]
    b = thetas.shape[0]
    kernel_number, Cout, Cin, k, k = weights.shape
    weights = weights.reshape(kernel_number, Cout//num_groups, num_groups, Cin, k, k).permute(2, 0, 1, 3, 4, 5)

    final_weight=[]

    ##########################################
    thetas=torch.zeros_like(thetas)
    lambdas=torch.ones_like(lambdas)
    ###########################################
    
    for idx, weight in enumerate(weights):
        final_weight.append(batch_rotate_multiweight_one_channel(weight, lambdas[:,idx].unsqueeze(1), thetas[:,idx].unsqueeze(1)))
    
    final_weights = torch.cat(final_weight, dim=1)
    final_weights = final_weights.reshape(b * Cout, Cin, 3, 3)
    # print("!!!final_weights!!!!",final_weights.shape)
    return final_weights



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
    
    
    

class AdaptiveRotatedConv2d_multichannel(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight_multi_channel, num_groups=1):
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
        rotated_weight = self.rotate_func(self.weight, alphas, angles, self.num_groups)

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
    

class AdaptiveRotatedConv2d_multichannel_baseline(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight_multi_channel_baseline, num_groups=1):
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
        rotated_weight = self.rotate_func(self.weight, alphas, angles, self.num_groups)

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










##########################################################################################################################################################


# def batch_rotate_multiweight_fast(weights, lambdas, thetas):
#     """
#     Let
#         batch_size = b
#         kernel_number = 1
#         kernel_size = 3
#         num_group = g
#     Args:
#         weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
#         thetas: tensor of thetas,  shape = [batch_size, kernel_number]
#     Return:
#         weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
#     """
#     assert(thetas.shape == lambdas.shape)

#     weights = weights.squeeze(0) #[Cout, Cin, k, k]
    
#     b = thetas.shape[0]
#     num_groups = thetas.shape[1]
#     k = weights.shape[-1]
    
#     Cout, Cin, _, _ = weights.shape


#     rotation_matrix = _get_rotation_matrix(thetas)#b,num_groups,9,9


#     lambdas = lambdas.unsqueeze(2).unsqueeze(3) #[b,num_groups,1,1] 
    
#     rotation_matrix = torch.mul(rotation_matrix, lambdas) #b,num_groups,9,9

#     # [1, num_groups,Cout//num_groups,Cin, 9, 1]
#     # [b, num_groups,     1          , 1,   9, 9]
#     weights = weights.view(                1, Cout//num_groups,num_groups, Cin, 9, 1).permute(0,2,1,3,4,5)
#     rotation_matrix = rotation_matrix.view(b, num_groups,     1          , 1,   9, 9)
    
    
#     weights=torch.matmul(rotation_matrix, weights)
#     weights = weights.reshape(b * Cout, Cin, 3, 3)
    
#     return weights

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
    # print("!!!!!num_groups!!!!!",num_groups)
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
    
    
##############################################################################################################################################################################

def batch_rotate_multiweight_fast_2(weights, lambdas, thetas):
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
    
    b = thetas.shape[0]
    num_groups = thetas.shape[1]
    k = weights.shape[-1]
    
    Cout, Cin, _, _ = weights.shape


    rotation_matrix = _get_rotation_matrix(thetas)#b,num_groups,9,9


    lambdas = lambdas.unsqueeze(2).unsqueeze(3) #[b,num_groups,1,1] 
    
    rotation_matrix = torch.mul(rotation_matrix, lambdas) #b,num_groups,9,9


    weights = weights.view(                1, num_groups,Cout//num_groups, Cin, 9, 1)
    rotation_matrix = rotation_matrix.view(b, num_groups,     1          , 1,   9, 9)
    
    weights=torch.matmul(rotation_matrix, weights)
    weights = weights.reshape(b * Cout, Cin, 3, 3)
    
    return weights


class AdaptiveRotatedConv2d_multichannel_fast_2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight_fast_2, num_groups=1):
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
    
    
##########################################################################################################################################################


def batch_rotate_multiweight_fast_twoside(weights, lambdas, thetas, num_groups_1, num_groups_2):
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
    
    b = thetas.shape[0]
    # num_groups = thetas.shape[1]//2
    k = weights.shape[-1]
    
    Cout, Cin, _, _ = weights.shape


    rotation_matrix = _get_rotation_matrix(thetas)#b,num_groups,9,9


    lambdas = lambdas.unsqueeze(2).unsqueeze(3) #[b,num_groups,1,1] 
    
    rotation_matrix = torch.mul(rotation_matrix, lambdas) #b,num_groups,9,9


    weights = weights.view(                1, Cout//num_groups_1,num_groups_1, Cin//num_groups_2,num_groups_2, 9, 1).permute(0,2,1,4,3,5,6)
    rotation_matrix = rotation_matrix.view(b, num_groups_1,     1          , num_groups_2,   9, 9)
    
    weights=torch.matmul(rotation_matrix, weights)
    weights = weights.reshape(b * Cout, Cin, 3, 3)
    
    return weights


class AdaptiveRotatedConv2d_multichannel_fast_twoside(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight_fast_twoside, num_groups_1=1, num_groups_2=1):
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
        self.num_groups_1 = num_groups_1
        self.num_groups_2 = num_groups_2

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
        rotated_weight = self.rotate_func(self.weight, alphas, angles, self.num_groups_1, self.num_groups_2)

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
    
#######################################################################################################
    
def batch_rotate_multiweight_fast_20231024(weights, lambdas, thetas):
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
    
    b = thetas.shape[0]
    num_groups = thetas.shape[1]
    k = weights.shape[-1]
    
    Cout, Cin, _, _ = weights.shape


    rotation_matrix = _get_rotation_matrix(thetas)#b,num_groups,9,9


    lambdas = lambdas.unsqueeze(2).unsqueeze(3) #[b,num_groups,1,1] 
    
    rotation_matrix = torch.mul(rotation_matrix, lambdas) #b,num_groups,9,9

    # weights:         [1, num_groups, Cout//num_groups, Cin,  9,  1] ->[num_groups*9,   Cout//num_groups * Cin]
    # rotation_matrix: [b, num_groups, num_groups      , 1,    9,  9] ->[b*num_groups*9, num_groups*9]
    
    # [b*9, n*9] x [n*9, Cout*Cin]
    #
    rotation_matrix = rotation_matrix.repeat(1, 1, num_groups, 1, 1, 1).view(b*num_groups*9, num_groups*9)
    weights = weights.view(num_groups*9,   Cout//num_groups * Cin)

    weights = torch.mm(rotation_matrix, weights) #[b*num_groups*9, Cout//num_groups * Cin]
    
    weights = weights.reshape(b, num_groups, 9, Cout//num_groups, Cin).permute(0, 3, 1, 4, 2).reshape(b, Cout, Cin, 9)
    
    
    # weights = weights.view(                1, Cout//num_groups,num_groups, Cin, 9, 1).permute(0,2,1,3,4,5)
    # rotation_matrix = rotation_matrix.view(b, num_groups,     1          , 1,   9, 9)
    # weights=torch.matmul(rotation_matrix, weights)
    
    weights = weights.reshape(b * Cout, Cin, 3, 3)
    
    return weights

class AdaptiveRotatedConv2d_multichannel_fast_20231024(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight_fast_20231024, num_groups=1):
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