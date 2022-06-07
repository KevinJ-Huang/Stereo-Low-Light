import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import models.archs.arch_util as arch_utils
import math
from torch.nn import init
import os


class PSP(nn.Module):
    def __init__(self, channel):
        super(PSP, self).__init__()

        self.relu=nn.ReLU(0.2)

        self.tanh=nn.Tanh()

        self.refine1= nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv1020 = nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv1030 = nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv1040 = nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1)  # 1mm

        self.refine3= nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_bilinear


    def forward(self, x):
        process = self.relu((self.refine1(x)))
        shape_out = process.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(process, 16)

        x102 = F.avg_pool2d(process, 8)

        x103 = F.avg_pool2d(process, 4)

        x104 = F.avg_pool2d(process, 2)
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

        process = x1010 + x1020 + x1030 + x1040 + process
        process= self.relu(self.refine3(process))

        return process


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        arch_utils.initialize_weights(self.atrous_conv,0.1)

    def forward(self, x):
        x = self.atrous_conv(x)
        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, inplanes):
        super(ASPP, self).__init__()
        self.refine1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            CALayer(inplanes,inplanes//4))
        self.refine2 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            CALayer(inplanes,inplanes//4))

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes, inplanes, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, inplanes, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, inplanes, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, inplanes, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, inplanes, 1, stride=1, bias=False),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(inplanes*9, inplanes, 1, bias=False)
        self.relu = nn.ReLU()

        self.conv1010 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv1020 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv1030 = nn.Conv2d(inplanes,  inplanes, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv1040 = nn.Conv2d(inplanes,  inplanes, kernel_size=3, stride=1, padding=1)  # 1mm

        arch_utils.initialize_weights([self.refine1,self.conv1010,self.conv1020,self.conv1030,self.conv1040,
                                       self.refine2,self.aspp1,self.aspp2,self.aspp3,self.aspp4],0.1)


    def forward(self, x):
        x = self.refine1(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5,size=x4.size()[2:], mode='bilinear', align_corners=True)

        x1_1 =  F.avg_pool2d(x, 16)
        x2_1 = F.avg_pool2d(x, 8)
        x3_1 = F.avg_pool2d(x, 4)
        x4_1 = F.avg_pool2d(x, 2)
        x1_1 = F.upsample(self.relu(self.conv1010(x1_1)),size=x.size()[2:], mode='bilinear', align_corners=True)
        x2_1 = F.upsample(self.relu(self.conv1020(x2_1)), size=x.size()[2:], mode='bilinear', align_corners=True)
        x3_1 = F.upsample(self.relu(self.conv1030(x3_1)), size=x.size()[2:], mode='bilinear', align_corners=True)
        x4_1 = F.upsample(self.relu(self.conv1040(x4_1)), size=x.size()[2:], mode='bilinear', align_corners=True)


        x = torch.cat((x1, x2, x3, x4, x5, x1_1, x2_1, x3_1, x4_1), dim=1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.relu(self.refine2(x))

        return x


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)




class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

        arch_utils.initialize_weights([self.conv_du,self.process],0.1)

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)+self.contrast(y)
        z = self.conv_du(y)
        return z * y + x



class Encoder(nn.Module):
    def __init__(self,nf):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1)
        self.att1 = nn.Sequential(CALayer(2*nf,nf//2),
                                  CALayer(2*nf,nf//2))
        self.activ_1 = nn.LeakyReLU(0.1,inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv2 = nn.Conv2d(in_channels=2*nf, out_channels=4*nf, kernel_size=3, padding=1)
        self.att2 = nn.Sequential(CALayer(4*nf, nf),
                                  CALayer(4*nf,nf))
        self.activ_2 = nn.LeakyReLU(0.1,inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv3 = nn.Conv2d(in_channels=4*nf, out_channels=8*nf, kernel_size=3, padding=1)
        self.att3 = nn.Sequential(CALayer(8*nf, 2*nf),
                                  CALayer(8 * nf, 2 * nf))
        self.activ_3 = nn.LeakyReLU(0.1,inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv4 = nn.Conv2d(in_channels=8*nf, out_channels=16*nf, kernel_size=3, padding=1)
        self.att4 = nn.Sequential(CALayer(16*nf, 4*nf),
                                  CALayer(16*nf, 4*nf))
        self.activ_4 = nn.LeakyReLU(0.1,inplace=True)

        arch_utils.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.att1,
                                       self.att2, self.att3, self.att4], 0.1)


    def forward(self, x):
        out_1 = x
        out = self.conv1(x)
        out = self.activ_1(out)
        out = self.att1(out)
        size1 = out.size()
        out, indices1 = self.pool1(out)

        out_2 = out
        out = self.conv2(out)
        out = self.activ_2(out)
        out = self.att2(out)
        size2 = out.size()
        out, indices2 = self.pool2(out)

        out_3 = out
        out = self.conv3(out)
        out = self.activ_3(out)
        out = self.att3(out)
        size3 = out.size()
        out, indices3 = self.pool3(out)

        out_4 = out
        out = self.conv4(out)
        out = self.activ_4(out)
        out = self.att4(out)

        return out, out_1, out_2, out_3, out_4, size1, size2, size3,  indices1, indices2, indices3



class Enhance_UNet(nn.Module):

    def __init__(self, nf):
        super(Enhance_UNet, self).__init__()

        self.encoder = Encoder(nf=nf)
        self.aspp = ASPP(16*nf)

        self.deconv1 = nn.ConvTranspose2d(in_channels=24*nf, out_channels=8*nf, kernel_size=3, padding=1)  ##
        self.activ_4 = nn.LeakyReLU(0.1,inplace=True)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.channel_att1 = nn.Sequential(CALayer(8*nf,nf*2),
                                          CALayer(8 * nf, nf * 2))

        self.deconv2 = nn.ConvTranspose2d(in_channels=12*nf, out_channels=4*nf, kernel_size=3, padding=1)  ##
        self.activ_5 = nn.LeakyReLU(0.1,inplace=True)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2)
        self.channel_att2 = nn.Sequential(CALayer(4*nf,nf),
                                          CALayer(4*nf,nf))

        self.deconv3 = nn.ConvTranspose2d(in_channels=6*nf, out_channels=2*nf, kernel_size=3, padding=1)
        self.activ_6 = nn.LeakyReLU(0.1,inplace=True)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2)
        self.channel_att3 = nn.Sequential(CALayer(2*nf,nf//2),
                                          CALayer(2 * nf, nf // 2))

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2*nf, out_channels=2*nf, kernel_size=3,  padding=1),
            CALayer(2*nf,nf//2),
            CALayer(2 * nf, nf // 2),

        )

        self.conv_last2 = CALayer(nf, nf//4)
        self.conv_last = nn.Conv2d(in_channels= nf, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.conv_scale1 = nn.Conv2d(8 * nf, 3, 1, 1, 0)
        self.conv_scale2 = nn.Conv2d(4 * nf, 3, 1, 1, 0)
        self.conv_scale3 = nn.Conv2d(2 * nf, 3, 1, 1, 0)

        arch_utils.initialize_weights([self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.conv_last], 0.1)


    def forward(self, x):

        out, out_1, out_2, out_3, out_4, size1, size2, size3, indices1, indices2, indices3 = self.encoder(x)
        out = self.aspp(out)

        out = torch.cat((out, out_4), dim=1)
        out = self.deconv1(out)
        out = self.activ_4(out)
        out = self.unpool1(out, indices3, size3)
        out = self.channel_att1(out)
        out = torch.cat((out, out_3), dim=1)

        out = self.deconv2(out)
        out = self.activ_5(out)
        out = self.unpool2(out, indices2, size2)
        out = self.channel_att2(out)
        out = torch.cat((out, out_2), dim=1)

        out = self.deconv3(out)
        out = self.activ_6(out)
        out = self.unpool3(out, indices1, size1)
        out = self.channel_att3(out)

        out = self.deconv4(out)

        out = out[:,:8,:,:]*x+out[:,8:,:,:]
        out = self.conv_last2(out)
        out = F.tanh(self.conv_last(out))*0.58+0.5

        return out
