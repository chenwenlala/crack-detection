import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from models.networks import init_net
from functools import partial
from mmcv.cnn import constant_init, kaiming_init
from torch import nn


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


nonlinearity = partial(F.relu, inplace=True)

__all__ = ['Res2Net', 'res2net50', 'res2net101_26w_4s']

model_urls = {
    'res2net50_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net101_26w_4s-02a759a1.pth',
}
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=8):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)#softmax操作
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SideOutput(nn.Module):

    def __init__(self, num_output, kernel_sz=None, stride=None):
        super(SideOutput, self).__init__()
        self.conv = nn.Conv2d(num_output, 1, 1, stride=1, padding=0, bias=True)

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(1, 1, kernel_sz, stride=stride, padding=0, bias=False)
        else:
            self.upsample = False

    def forward(self, res):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)

        return side_output


class SideOutput2(nn.Module):

    def __init__(self, num_output, kernel_sz=None, stride=None):
        super(SideOutput2, self).__init__()
        self.conv = nn.Conv2d(num_output, 1, 1, stride=1, padding=0, bias=True)

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(1, 1, kernel_sz, stride=stride, padding=1, bias=False)
        else:
            self.upsample = False

    def forward(self, res):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)

        return side_output


class Res5Output(nn.Module):

    def __init__(self, num_output=2048, kernel_sz=8, stride=8):
        super(Res5Output, self).__init__()
        self.conv = nn.Conv2d(num_output, 1, 1, stride=1, padding=0)
        self.upsampled = nn.ConvTranspose2d(1, 1, kernel_size=kernel_sz, stride=stride, padding=0)

    def forward(self, res):
        res = self.conv(res)
        res = self.upsampled(res)
        return res


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate6 = nn.Conv2d(channel, channel, kernel_size=3, dilation=6, padding=6)
        self.dilate8 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate9 = nn.Conv2d(channel, channel, kernel_size=3, dilation=9, padding=9)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_1_out = nonlinearity(self.dilate1(x))
        dilate1_2_out = nonlinearity(self.dilate2(dilate1_1_out))
        dilate1_3_out = nonlinearity(self.dilate4(dilate1_2_out))
        dilate1_4_out = nonlinearity(self.dilate8(dilate1_3_out))

        #dilate2_1_out = nonlinearity(self.dilate1(x))
        dilate2_2_out = nonlinearity(self.dilate3(dilate1_1_out))
        dilate2_3_out = nonlinearity(self.dilate6(dilate2_2_out))
        dilate2_4_out = nonlinearity(self.dilate9(dilate2_3_out))

        dilate3_out = nonlinearity(self.dilate5(dilate1_2_out))

        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_1_out + dilate1_2_out + dilate1_4_out + dilate2_4_out + dilate3_out #+ dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res2 = self._make_layer(block, 64, layers[0])
        self.gc2 = ContextBlock2d(64 * 4, 64 * 4)
        self.res3 = self._make_layer(block, 256, layers[1], stride=2)
        self.gc3 = ContextBlock2d(256 * 4, 256 * 4)
        self.res4 = self._make_layer(block, 256, layers[2], stride=2)
        self.gc4 = ContextBlock2d(256 * 4, 256 * 4)
        self.res5 = self._make_layer(block, 512, layers[3], stride=1)
        self.gc5 = ContextBlock2d(512 * 4, 512 * 4)
        self.SideOutput1 = SideOutput(64)
        self.SideOutput2 = SideOutput(256, kernel_sz=4, stride=2)
        self.SideOutput22 = SideOutput2(256, kernel_sz=4, stride=2)
        self.SideOutput3 = SideOutput(1024, kernel_sz=4, stride=4)
        self.Res5Output = Res5Output()
        self.sigmoid = nn.Sigmoid()
        self.dblock = Dblock(2048)
        filters = [64, 256, 1024, 2048]
        self.decoder4 = DecoderBlock(filters[3], filters[2])  # 256
        self.decoder3 = DecoderBlock(filters[2], filters[1])  # 256
        self.decoder2 = DecoderBlock(filters[1], filters[0])  # 64
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 64
        self.fuse_conv = nn.Conv2d(6, 1, kernel_size=1, stride=1, bias=False)
        self.up = nn.ConvTranspose2d(1024, 1024, 2, stride=2, padding=0, bias=False)

        """for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)"""

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        # res1
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)

        # res2
        x2 = self.maxpool(x1)

        x2 = self.res2(x2)
        x2 = self.gc2(x2)
        # res3
        x3 = self.res3(x2)
        x3 = self.gc3(x3)

        # res4
        x4 = self.res4(x3)
        x4 = self.gc4(x4)
        # res5
        x5 = self.res5(x4)
        x5 = self.gc5(x5)
        side_5 = self.Res5Output
        side_5 = side_5(x5)
        # Center
        x5 = self.dblock(x5)
        x5 = self.gc5(x5)
        side_6 = self.Res5Output
        side_6 = side_6(x5)

        x44 = self.up(x4)
        # Decoder
        d5 = self.decoder4(x5) + x44
        s5 = self.SideOutput3(d5)
        d44 = d5 + x3
        s44 = self.SideOutput3(d44)

        d4 = self.decoder3(d44) + x2
        s4 = self.SideOutput22(d4)

        d3 = self.decoder2(d4) + x1
        s3 = self.SideOutput1(d3)

        fused = self.fuse_conv(torch.cat([
            s5,
            s4,
            s44,
            s3,
            side_5,
            side_6
        ], dim=1))

        return s5, s4, s44, s3, side_5, side_6, fused


def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net101_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model


def res2net50_26w_6s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=6, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_6s']))
    return model


def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_8s']))
    return model


def res2net50_48w_2s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=48, scale=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_48w_2s']))
    return model


def res2net50_14w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_14w_8s']))
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net101_26w_4s(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())


def define_res2net(in_nc,
                   num_classes,
                   ngf,
                   norm='batch',
                   init_type='xavier',
                   init_gain=0.02,
                   gpu_ids=[]):
    net = res2net101_26w_4s(pretrained=False)
    return init_net(net, init_type, init_gain, gpu_ids)
