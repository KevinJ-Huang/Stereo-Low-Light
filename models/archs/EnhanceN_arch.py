import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from models.archs.arch_PAM import PAM
from models.archs.arch_util import DWT,IWT,space_to_depth,GaussianBlur,GlobalContextBlock

class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out, buffer_cat


class PreEnhance(nn.Module):
    def __init__(self):
        super(PreEnhance,self).__init__()
        G0 = 8
        G = 4
        self.encoder1 = nn.Conv2d(3, G0, 3, 1, 1)
        self.encoder2 = RDB(G0=4*G0, C=1, G=G)
        self.downsample2 = nn.Conv2d(4 * G0, 4 * G0, 3, padding=1, stride=2)
        self.process = RDB(G0=4*G0, C=1, G=2*G)
        self.up2 = nn.ConvTranspose2d(4 * G0, 4 * G0, 4, stride=2, padding=1)
        self.catfuse = nn.Conv2d(8*G0, 4*G0, 1, 1, 0)
        self.decoder2 = RDB(G0=4*G0, C=1, G=G)
        self.decoder1 =  nn.Conv2d(G0, 3, 3, 1, 1)

    def forward(self, x):
        fe1 = self.encoder1(x)
        fe2 = self.encoder2(space_to_depth(fe1,2))
        f = self.process(self.downsample2(fe2))
        fd2 = self.decoder2(self.catfuse(torch.cat([self.up2(f),fe2],1)))
        fd1 = self.decoder1(F.pixel_shuffle(fd2,2))

        return fd1*x,fd1




class Net(nn.Module):
    def __init__(self, upscale_factor=1.0):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        G0 = 4
        G = 2

        self.pre_enhance = PreEnhance()
        self.blur = GaussianBlur().cuda()

        self.encoder0 = nn.Conv2d(6, G0, 3, padding=1, stride=1)
        self.encoder1 = RDB(G0=G0, C=3, G=G)
        self.downsample1 = nn.Conv2d(G0, 2*G0, 3, padding=1, stride=2)
        self.encoder2 = RDG(G0=2*G0, C=4, G=2*G, n_RDB=4)
        self.downsample2 = nn.Conv2d(2*G0, 4*G0, 3, padding=1, stride=2)
        self.encoder3 = RDG(G0=4*G0, C=2, G=4*G, n_RDB = 4)

        self.pam = PAM(4*G0)

        self.fusion3 = Fusion(4*G0,2*G,4,4)


        self.fusion2 = Fusion(2*G0,G,4,2)

        self.fusion1 = Fusion(G0, G, 2,1)


        self.up2 = nn.ConvTranspose2d(4*G0, 2*G0, 4, stride=2, padding=1)
        self.decoder2 = RDG(G0=2*G0, C=4, G=2*G, n_RDB=4)
        self.up1 = nn.ConvTranspose2d(2*G0, G0, 4, stride=2, padding=1)
        self.decoder1 =  RDB(G0=G0, C=3, G=G)

        self.decoder0 =  nn.Sequential(RDB(G0=G0, C=2, G=G),
                                       nn.Conv2d(G0,3,1,1,0))

        self.sc1 = RDB(G0=G0, C=2, G=G)
        self.sc2 = RDB(G0=2 * G0, C=2, G=G)
        self.warp = Warp()
        self.DWT = DWT()
        self.IWT = IWT()

        self.UPNet3 = nn.Sequential(*[
            nn.Conv2d(4*G0, 2*G0, 3, padding=1, stride=1),
            nn.Conv2d(2*G0, 3, 3, padding=1, stride=1)
        ])

        self.UPNet2 = nn.Sequential(*[
            nn.Conv2d(2*G0, G0, 3, padding=1, stride=1),
            nn.Conv2d(G0, 3, 3, padding=1, stride=1)
        ])

        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x_left, x_right, is_training):

        x_left_en, ill_left = self.pre_enhance(x_left)
        x_right_en, ill_right = self.pre_enhance(x_right)
        left_ef1 = self.encoder1(self.encoder0(torch.cat([x_left, x_left_en], 1)))
        right_ef1 = self.encoder1(self.encoder0(torch.cat([x_right, x_right_en], 1)))

        left_ef2,_ = self.encoder2(self.downsample1(left_ef1))
        right_ef2,_ = self.encoder2(self.downsample1(right_ef1))

        left_ef3, catfea_left = self.encoder3(self.downsample2(left_ef2))
        right_ef3, catfea_right = self.encoder3(self.downsample2(right_ef2))


        buffer_leftT3, buffer_rightT3, (M_right_to_left, M_left_to_right), (V_left, V_right)\
            = self.pam(left_ef3, right_ef3, catfea_left, catfea_right, is_training)

        buffer_leftF3 = self.fusion3(left_ef3, buffer_leftT3, ill_left)
        buffer_rightF3 = self.fusion3(right_ef3, buffer_rightT3,ill_right)


        left_ef2 = self.sc2(left_ef2)
        right_ef2 = self.sc2(right_ef2)
        buffer_leftT2, buffer_rightT2 = self.warp(left_ef2, right_ef2, M_right_to_left, M_left_to_right, V_left, V_right)
        buffer_leftF2 = self.fusion2(left_ef2, buffer_leftT2, ill_left)
        buffer_rightF2 = self.fusion2(right_ef2, buffer_rightT2, ill_right)

        left_df2, _ = self.decoder2(self.up2(buffer_leftF3))
        left_df2 += buffer_leftF2
        right_df2, _ = self.decoder2(self.up2(buffer_rightF3))
        right_df2 += buffer_rightF2

        left_ef1 = self.sc1(left_ef1)
        right_ef1 = self.sc1(right_ef1)
        buffer_leftT1, buffer_rightT1 = self.warp(self.DWT(left_ef1), self.DWT(right_ef1), M_right_to_left, M_left_to_right, V_left, V_right)
        buffer_leftF1 = self.fusion1(left_ef1, self.IWT(buffer_leftT1), ill_left)
        buffer_rightF1 = self.fusion1(right_ef1, self.IWT(buffer_rightT1), ill_right)


        left_df1 = self.decoder1(self.up1(left_df2))+buffer_leftF1
        right_df1 = self.decoder1(self.up1(right_df2))+buffer_rightF1


        res3_left = self.UPNet3(buffer_leftF3)
        res3_right = self.UPNet3(buffer_rightF3)

        res2_left = self.UPNet2(left_df2)+self.Img_up(res3_left)
        res2_right = self.UPNet2(right_df2)+self.Img_up(res3_right)

        x_left_en = self.blur(x_left_en)
        x_right_en = self.blur(x_right_en)
        out_left = self.decoder0(left_df1) + self.Img_up(res2_left) + x_left_en
        out_right = self.decoder0(right_df1) + self.Img_up(res2_right) + x_right_en


        if is_training == 1:

            # M_right_to_left = F.interpolate(M_right_to_left,scale_factor=4,align_corners=False,mode='bicubic')
            # M_left_to_right = F.interpolate(M_left_to_right, scale_factor=4, align_corners=False,mode='bicubic')
            # V_left = F.interpolate(V_left,scale_factor=4,align_corners=False,mode='bilinear')
            # V_right = F.interpolate(V_right,scale_factor=4,align_corners=False,mode='bilinear')
            return out_left, out_right, x_left_en, x_right_en, res2_left, res2_right, res3_left, res3_right,\
                   (M_right_to_left, M_left_to_right), (V_left, V_right)
        if is_training == 0:
            return out_left, out_right



class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, max(1,channel//16), 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(max(1,channel//16), channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x


class Fusion(nn.Module):
    def __init__(self,G0, G, N, level):
        super(Fusion,self).__init__()

        self.ill_trans = nn.Sequential(
            nn.Conv2d(3, G0, 1, 1, 0),
            nn.Upsample(scale_factor=1.0/level,mode='bilinear'),
            nn.Sigmoid()
        )

        self.weightfusion = nn.Sequential(
            RDB(G0=2 * G0, C=N, G=G),
            CALayer(2 * G0),
            nn.Conv2d(2 * G0, G0, kernel_size=1, stride=1, padding=0, bias=True))

        self.GCConv = GlobalContextBlock(G0)

        self.Awarefusion = nn.Conv2d(2*G0,G0,1,1,0)


    def forward(self, f, f_buff, ill):
        f_buff = self.GCConv(f_buff)

        ill = ill.detach()
        ill = self.ill_trans(ill)

        f_fuse = self.weightfusion(torch.cat([f,f_buff],dim=1))
        f_out = self.Awarefusion(torch.cat([ill*f_fuse,(1-ill)*f],1))

        return f_out



class Warp(nn.Module):
    def __init__(self):
        super(Warp,self).__init__()
        self.DWT = DWT().cuda()
        self.IWT = IWT().cuda()

    def forward(self, x_left, x_right, M_right_to_left, M_left_to_right, V_left_tanh, V_right_tanh):
        x_left = self.DWT(x_left)
        x_right = self.DWT(x_right)
        b, c0, h0, w0 = x_left.shape

        M_right_to_left = M_right_to_left.detach()
        M_left_to_right = M_left_to_right.detach()
        V_left_tanh = V_left_tanh.detach()
        V_right_tanh = V_right_tanh.detach()

        M_right_to_left = M_right_to_left.contiguous().view(b*h0, w0, w0)
        M_left_to_right = M_left_to_right.contiguous().view(b*h0, w0, w0)


        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)  # B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                             ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)  # B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) + x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        out_left = self.IWT(out_left)
        out_right = self.IWT(out_right)

        return out_left, out_right