import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
from models.loss_new import SSIMLoss,VGGLoss
from models.archs.arch_util import GaussianBlur
import torch.nn.functional as F
from metrics.calculate_PSNR_SSIM import psnr_np
logger = logging.getLogger('base')


class SIEN_Model(BaseModel):
    def __init__(self, opt):
        super(SIEN_Model, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                self.mse = nn.MSELoss().to(self.device)
                self.blur = GaussianBlur().cuda()
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            self.l_pix_w = train_opt['pixel_weight']
            self.l_ssim_w = train_opt['ssim_weight']
            self.l_vgg_w = train_opt['vgg_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['fix_some_part']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        LQleft_IMG = data['LQleft']
        GTleft_IMG = data['GTleft']
        LQright_IMG = data['LQright']
        GTright_IMG = data['GTright']
        self.varleft_L = LQleft_IMG.to(self.device)
        self.varright_L = LQright_IMG.to(self.device)

        if need_GT:
            self.realleft_H = GTleft_IMG.to(self.device)
            self.realright_H = GTright_IMG.to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['fix_some_part'] and step < self.opt['train']['fix_some_part']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()

        HR_left, HR_right, LR_left, LR_right =  self.realleft_H,self.realright_H, self.varleft_L,self.varright_L

        SR_left, SR_right,  SR_left_en, SR_right_en, res2_left, res2_right, res3_left, res3_right,\
        (M_right_to_left, M_left_to_right), (V_left, V_right) = self.netG(LR_left, LR_right, is_training=1)


        loss_SR = self.cri_pix(SR_left, HR_left) + self.cri_pix(SR_right, HR_right)
        loss_pre = self.cri_pix(SR_left_en, self.blur(HR_left)) + self.cri_pix(SR_right_en, self.blur(HR_right))
        loss_level4 =  self.level_loss(LR_left,HR_left,LR_right,HR_right,SR_left,SR_right,SR_left_en,
                                      SR_right_en,M_right_to_left,M_left_to_right,V_left,V_right,4)
        loss_band2 = self.cri_pix(res2_left,F.interpolate(HR_left-SR_left_en,scale_factor=0.5,align_corners=False,mode='bilinear'))+\
                     self.cri_pix(res2_right,F.interpolate(HR_right-SR_right_en,scale_factor=0.5,align_corners=False,mode='bilinear'))
        loss_band3 = self.cri_pix(res3_left, F.interpolate(HR_left - SR_left_en, scale_factor=0.25, align_corners=False,mode='bilinear')) + \
                     self.cri_pix(res3_right, F.interpolate(HR_right - SR_right_en, scale_factor=0.25, align_corners=False,mode='bilinear'))
        # loss_level2 = self.level_loss(LR_left,HR_left,LR_right,HR_right,SR_left,SR_right,M_right_to_left2,M_left_to_right2,V_left2,V_right2,2)
        # loss_level2 = loss_level4

        l_total = loss_SR + 0.4*loss_level4 + loss_band2*0.1 + loss_band3*0.15 +0.4*loss_pre

        l_total.backward()
        self.optimizer_G.step()
        self.fake_H = SR_left
        psnr = psnr_np(self.fake_H.detach(), self.realleft_H.detach())

        # set log
        self.log_dict['psnr'] = psnr.item()
        self.log_dict['l_pre'] = 0.4*loss_pre.item()
        self.log_dict['l_total'] = l_total.item()
        self.log_dict['l_sr'] = loss_SR.item()
        self.log_dict['l_band2'] = 0.1*loss_band2.item()
        self.log_dict['l_band3'] = 0.15*loss_band3.item()
        self.log_dict['l_lv4'] = loss_level4.item()

    def level_loss(self, LR_left, HR_left, LR_right, HR_right, SR_left, SR_right, SR_left_en, SR_right_en,
                   M_right_to_left, M_left_to_right, V_left, V_right, level):
        b, c, h, w = LR_left.shape
        h, w = h // level, w // level
        Res_left = F.interpolate(torch.abs(HR_left - SR_left_en), scale_factor=1.0 / level, align_corners=False,
                                 mode='bicubic')
        Res_right = F.interpolate(torch.abs(HR_right - SR_right_en), scale_factor=1.0 / level, align_corners=False,
                                  mode='bicubic')
        Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                              Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                              ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                               Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                               ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        loss_photo = self.cri_pix(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                     self.cri_pix(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))

        loss_h = self.cri_pix(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                 self.cri_pix(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
        loss_w = self.cri_pix(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                 self.cri_pix(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
        loss_smooth = loss_w + loss_h

        Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                                   Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                   ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                                    Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        loss_cycle = self.cri_pix(Res_left * V_left.repeat(1, 3, 1, 1), Res_left_cycle * V_left.repeat(1, 3, 1, 1)) + \
                     self.cri_pix(Res_right * V_right.repeat(1, 3, 1, 1), Res_right_cycle * V_right.repeat(1, 3, 1, 1))

        SR_left_res = F.interpolate(torch.abs(HR_left - SR_left), scale_factor=1.0 / level, align_corners=False,
                                    mode='bicubic')
        SR_right_res = F.interpolate(torch.abs(HR_right - SR_right), scale_factor=1.0 / level, align_corners=False,
                                     mode='bicubic')
        SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w),
                                 SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                 ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w),
                                  SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        loss_cons = self.cri_pix(SR_left_res * V_left.repeat(1, 3, 1, 1), SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                    self.cri_pix(SR_right_res * V_right.repeat(1, 3, 1, 1), SR_right_resT * V_right.repeat(1, 3, 1, 1))

        return loss_cons + (loss_photo + loss_smooth + loss_cycle)


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            res_left,res_right = self.netG(self.varleft_L,self.varright_L,is_training=0)
            self.fake_H = res_left
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQleft'] = self.varleft_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GTleft'] = self.realleft_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def save_best(self):
        self.save_network(self.netG, 'best', 0)