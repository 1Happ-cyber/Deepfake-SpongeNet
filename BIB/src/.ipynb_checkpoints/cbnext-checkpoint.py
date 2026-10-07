import torch
import torch.nn as nn
from torchinfo import summary
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from .EMA import EMA
from .SKNET import SKAttention
from .CBAM import CBAM
#这里是CNN结合BNN，其中CNN加在stage2和stage3上，主要是引入多尺度注意力

#stage ratio: 1:1:3:1
stage_out_channel_tiny = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

#stage ratio 1:1:3:1
stage_out_channel_small = [48] + [96] + [192] * 2 + [384] * 2 + [768] * 6 + [1536] * 2

#stage ratio 2:2:4:2
stage_out_channel_middle = [48] + [96] + [192] * 4 + [384] * 4 + [768] * 8 + [1536] *4

#stage ratio 2:2:8:2
stage_out_channel_large = [64] + [128] + [256] * 4 + [512] * 4 + [1024] * 16 + [2048] * 4

class BinarizedMLP(nn.Module):
    # ... (代码与上一版相同，无需修改) ...
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.norm_in = nn.GroupNorm(num_groups=1, num_channels=in_features)
        self.sign_in = HardSign()
        self.fc1 = HardBinaryConv(in_features, hidden_features, kernel_size=1, padding=0)
        self.norm_mid = nn.GroupNorm(num_groups=1, num_channels=hidden_features)
        self.activation = nn.PReLU(hidden_features)
        self.sign_mid = HardSign()
        self.fc2 = HardBinaryConv(hidden_features, out_features, kernel_size=1, padding=0)
    def forward(self, x_real):
        x_bin = self.sign_in(self.norm_in(x_real))
        x_real_out1 = self.fc1(x_bin)
        x_real_mid = self.activation(self.norm_mid(x_real_out1))
        x_bin_mid = self.sign_mid(x_real_mid)
        x_real_out2 = self.fc2(x_bin_mid)
        return x_real_out2

# ===================================================================
# BNN_CSI_SAFM 模块的最终修复版
# ===================================================================
class BNN_CSI_SAFM(nn.Module):
    """
    Binarized Cross-Scale Interaction Scale-Aware Fusion Module.
    FINAL ROBUST VERSION: All BatchNorm2d layers are replaced with GroupNorm
    to handle arbitrary feature map sizes, including 1x1.
    """
    def __init__(self, dim, n_levels=4):
        super().__init__()
        assert n_levels == 4, "BNN_CSI_SAFM is currently designed for 4 levels."
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.chunk_dim = chunk_dim

        # 动态选择一个合适的组数
        gn_groups = 16 if chunk_dim % 16 == 0 else 8 if chunk_dim % 8 == 0 else 4 if chunk_dim % 4 == 0 else 2 if chunk_dim % 2 == 0 else 1
        
        # ===> 修复点 1: 将 self.branch_bns 替换为 GroupNorm <===
        self.branch_norms = nn.ModuleList([nn.GroupNorm(gn_groups, chunk_dim) for _ in range(n_levels)])
        
        # 自适应核大小 (保持不变)
        mfr_list = []
        for i in range(n_levels):
            kernel_size = 3 if i == 0 else 1
            mfr_list.append(HardBinaryConv(chunk_dim, chunk_dim, kernel_size=kernel_size, stride=1, groups=chunk_dim))
        self.mfr = nn.ModuleList(mfr_list)
        
        # 全局注意力生成器 (内部已是GroupNorm，安全)
        self.global_att_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            BinarizedMLP(chunk_dim, chunk_dim // 4, chunk_dim),
            nn.Sigmoid() 
        )

        # ===> 修复点 2: 将局部空间注意力的BN也替换为GroupNorm <===
        self.local_att_generator = nn.Sequential(
            nn.GroupNorm(gn_groups, chunk_dim),
            HardSign(), 
            HardBinaryConv(chunk_dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # ===> 修复点 3: 将聚合层的BN也替换为GroupNorm <===
        aggr_gn_groups = 16 if dim % 16 == 0 else 8 if dim % 8 == 0 else 4 if dim % 4 == 0 else 1
        self.aggr_norm = nn.GroupNorm(aggr_gn_groups, dim)
        self.aggr_sign = HardSign()
        self.aggr = HardBinaryConv(dim, dim, kernel_size=1, padding=0)
        
        self.act = nn.GELU()

    def forward(self, x_real_in):
        h, w = x_real_in.size()[-2:]
        xc_real = x_real_in.chunk(self.n_levels, dim=1)
        
        processed_levels_real = []
        for i in range(self.n_levels):
            s_real = xc_real[i]
            if i > 0:
                p_size = (max(1, h // 2**i), max(1, w // 2**i))
                s_real = F.adaptive_max_pool2d(s_real, p_size)
            
            # 使用GroupNorm，不再有尺寸问题
            s_real_norm = self.branch_norms[i](s_real)
            s_bin = torch.sign(s_real_norm)
            s_out_real = self.mfr[i](s_bin)
            processed_levels_real.append(s_out_real)

        p0_real, p1_real, p2_real, p3_real = processed_levels_real
        attn_c = self.global_att_generator(p3_real)
        attn_s = self.local_att_generator(p0_real)
        enhanced_p0 = p0_real * attn_c
        attn_s_downsampled = F.adaptive_max_pool2d(attn_s, p3_real.size()[-2:])
        enhanced_p3 = p3_real * attn_s_downsampled

        out_levels = [enhanced_p0, p1_real, p2_real, enhanced_p3]
        for i in range(1, self.n_levels):
            out_levels[i] = F.interpolate(out_levels[i], size=(h, w), mode='nearest')
        concatenated_features_real = torch.cat(out_levels, dim=1)
        
        final_fusion_norm = self.aggr_norm(concatenated_features_real)
        final_fusion_bin = self.aggr_sign(final_fusion_norm)
        out_real = self.aggr(final_fusion_bin)
        final_output = self.act(out_real) * x_real_in
        
        return final_output
        
def conv3x3(in_planes, out_planes, kernel_size = 3, stride=1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, dilation = dilation, groups = groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class HardSigmoid(nn.Module):
    def __init__(self,):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x+3)/6

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.prelu = nn.PReLU(oup, oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardSign(nn.Module):
    def __init__(self, range = [-1, 1], progressive = False):
        super(HardSign, self).__init__()
        self.range = range
        self.progressive = progressive
        self.register_buffer("temperature", torch.ones(1))
        
    def adjust(self, x, scale = 0.1):
        self.temperature.mul_(scale)

    def forward(self, x):
        replace = x.clamp(self.range[0], self.range[1])
        x = x.div(self.temperature.clamp(min = 1e-8)).clamp(-1, 1)
        if not self.progressive:
            sign = x.sign()
        else:
            sign = x
        return (sign - replace).detach() + replace


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups = 1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups
        self.number_of_weights = in_chn // groups * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn//groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn((self.shape)) * 0.001, requires_grad=True)
        
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        if self.training:
            self.weight.data.clamp_(-1.5, 1.5)
        
        real_weights = self.weight
        
        if self.temperature < 1e-7:
            binary_weights_no_grad = real_weights.sign()
        else:
            binary_weights_no_grad = (real_weights/self.temperature.clamp(min = 1e-8)).clamp(-1, 1)
        cliped_weights = real_weights
        
        if self.training:
            binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            binary_weights =  binary_weights_no_grad
        
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups = self.groups)

        return y


class SqueezeAndExpand(nn.Module):
    def __init__(self, channels, planes, ratio = 8, attention_mode = "hard_sigmoid"):
        super(SqueezeAndExpand, self).__init__()
        self.se = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1,1)) ,
                        nn.Conv2d(channels, channels // ratio, kernel_size = 1, padding = 0),
                        nn.ReLU(channels//ratio),
                        nn.Conv2d(channels//ratio, planes, kernel_size = 1, padding = 0),
                    )

        if attention_mode == "sigmoid":
            self.attention = nn.Sigmoid()
        
        elif attention_mode == "hard_sigmoid":
            self.attention = HardSigmoid()

        else:
            self.attention = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.se(x)
        x = self.attention(x)
        return x


class Attention(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate = 0.1, infor_recoupling=True, groups = 1):
        super(Attention, self).__init__()
        
        self.inplanes = inplanes
        self.planes = planes
        self.infor_recoupling = infor_recoupling

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range = [-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size = 3, stride=stride, groups = groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        self.downsample = downsample
        self.stride = stride
        if stride == 2:
            self.pooling = nn.AvgPool2d(2,2)
        
        if self.infor_recoupling:
            self.se = SqueezeAndExpand(planes, planes, attention_mode = "sigmoid")
            self.scale = nn.Parameter(torch.ones(1, planes, 1, 1)*0.5)
        
    def forward(self, input):
        
        residual = self.activation1(input)
        
        if self.stride == 2:
          residual = self.pooling(residual)
        
        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)
        
        if self.infor_recoupling:
            if self.training:
                self.scale.data.clamp_(0, 1)
            if self.stride == 2:
                input = self.pooling(input)
            mix = self.scale*input + x*(1-self.scale)
            x = self.se(mix)*x
        else:
            pass
        x = x * residual 
        x = self.norm2(x)
        x = x + residual
        
        return x


class FFN_3x3(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, downsample = None, drop_rate = 0.1, infor_recoupling = True, groups = 1):
        super(FFN_3x3, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.infor_recoupling = infor_recoupling

        self.move= LearnableBias(inplanes)
        self.binary_activation = HardSign(range = [-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size = 3, stride = stride, groups = groups)
        
        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)
        
        if stride == 2:
            self.pooling = nn.AvgPool2d(2,2)
        
        if self.infor_recoupling:
            self.se = SqueezeAndExpand(inplanes, planes, attention_mode = "sigmoid")
            self.scale = nn.Parameter(torch.ones(1, planes, 1, 1)*0.5)

    def forward(self, input):
        
        residual = input

        if self.stride == 2:
            residual = self.pooling(residual)
        
        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)
        
        if self.infor_recoupling:
            if self.training:
                self.scale.data.clamp_(0,1)
            if self.stride == 2:
                input = self.pooling(input)
            mix = self.scale*input + (1-self.scale)*x
            x = self.se(mix) * x
            x = self.norm2(x)
        else:
            pass
        
        x = x + residual
        
        return x
        

class FFN_1x1(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, attention = True, drop_rate = 0.1, infor_recoupling = True):
        super(FFN_1x1, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.infor_recoupling = infor_recoupling

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range = [-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size = 1, stride = stride, padding = 0)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        if stride == 2:
            self.pooling = nn.AvgPool2d(2,2)
        
        if self.infor_recoupling:
            self.se = SqueezeAndExpand(inplanes, planes, attention_mode = "sigmoid")
            self.scale = nn.Parameter(torch.ones(1, planes, 1, 1)*0.5)        
         
    def forward(self, input):
        
        residual = input

        if self.stride == 2:
            residual = self.pooling(residual)
        
        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)
        if self.infor_recoupling:
            self.scale.data.clamp_(0, 1)
            mix = self.scale*input + (1-self.scale)*x
            x = self.se(mix)*x
            x = self.norm2(x)
        else:
            pass
        
        x = x + residual
        
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate = 0.1, mode = "scale", add_cnn_branch = "False"):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.add_cnn_branch = add_cnn_branch

        
        if mode == "scale":
            self.Attention = Attention(inplanes, inplanes, stride, None, drop_rate = drop_rate, groups = 1)
        else:
            self.Attention = FFN_3x3(inplanes, inplanes, stride, None, drop_rate = drop_rate, groups = 1)
        
        if inplanes == planes:
          self.FFN = FFN_1x1(inplanes, inplanes, drop_rate = drop_rate)
                        
        else:
          self.FFN_1 = FFN_1x1(inplanes, inplanes, drop_rate = drop_rate)

          self.FFN_2 = FFN_1x1(inplanes, inplanes, drop_rate = drop_rate)

        # --- NEW: 如果标志为True，则创建并行的CNN分支 ---
        if self.add_cnn_branch and inplanes == planes:
            self.cnn_branch = BNN_CSI_SAFM(dim = inplanes)
        # ---------------------------------------------------

    
    def forward(self, input):
        x = self.Attention(input)

        if self.inplanes == self.planes:
            y = self.FFN(x)
            if self.add_cnn_branch:
                y_cnn = self.cnn_branch(input)
                y = y + y_cnn

        else:   
          y_1 = self.FFN_1(x)
          y_2 = self.FFN_2(x)
          y = torch.cat((y_1, y_2), dim = 1)
            

        return y


class CBNext(nn.Module):
    def __init__(self, num_classes=1000, size = "tiny", ELM_Attention = True, Infor_Recoupling = True):
        super(CBNext, self).__init__()
        drop_rate = 0.2 if num_classes == 100 else 0.0
        
        if size == "tiny":
            stage_out_channel = stage_out_channel_tiny
        elif size == "small":
            stage_out_channel = stage_out_channel_small
        elif size == "middle":
            stage_out_channel = stage_out_channel_middle
        elif size == "large":
            stage_out_channel = stage_out_channel_large
        else:
            raise ValueError("The size is not defined!")

        if ELM_Attention and Infor_Recoupling:
            basicblock = BasicBlock
            print("Model with ELM Attention and Infor-Recoupling")
        else:
            basicblock = BasicBlock_No_Extra_Design
            print("Model with no Extra Design")

        
        # 需要添加CNN的索引,当前的只针对小的14
        cnn_stage_indices = [5,11]
        self.feature = nn.ModuleList()
        drop_rates = [x.item() for x in torch.linspace(0, drop_rate, (len(stage_out_channel)))]
        
        for i in range(len(stage_out_channel)):

            add_cnn = True if i in cnn_stage_indices else False
            
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 1 if num_classes != 1000 else 2))
            elif i == 1:
                self.feature.append((basicblock(stage_out_channel[i-1], stage_out_channel[i], 1, drop_rate = drop_rates[i], mode = "bias", add_cnn_branch=add_cnn)))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != stage_out_channel[1]:
                self.feature.append(basicblock(stage_out_channel[i-1], stage_out_channel[i], 2, drop_rate = drop_rates[i], mode = "scale" if i%2 == 0 else "bias",add_cnn_branch=add_cnn))
            else:
                self.feature.append(basicblock(stage_out_channel[i-1], stage_out_channel[i], 1, drop_rate = drop_rates[i], mode = "scale" if i%2 == 0 else "bias",add_cnn_branch=add_cnn))
        
        self.prelu = nn.PReLU(stage_out_channel[-1])
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_out_channel[-1], num_classes)
        
    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)
        x = self.prelu(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




if __name__ == "__main__":
    model = nn.DataParallel(CBNext(num_classes=1000, size="large")).cpu()
    print(model.eval().cuda(0)(torch.randn(1, 3, 224, 224).cuda(0)))
 
