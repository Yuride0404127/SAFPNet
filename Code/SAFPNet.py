import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

from backbone.mobilenet_v2.mobilenetv2 import mobilenet_v2
from backbone.mobilevit.model import MobileViT_XXS, MobileViT_XS, MobileViT_S
import torchvision.transforms

from backbone.mobilevit.model import MobileViT_S
from backbone.Shunted_Transformer.SSA import shunted_b, shunted_t, shunted_s
from backbone.Dformer.Dformer import DFormer_Small, DFormer_Base, DFormer_Tiny
# from Model_4.Bayesian import BBBConv2d
from Model_4.nn import HypLinear
from torchcrf import CRF
from operator import itemgetter
# from backbone.ConvNext import convnext_base
from einops import rearrange, repeat
from math import exp

stage1_channel_p = 64
stage2_channel_p = 128
stage3_channel_p = 256
stage4_channel_p = 512

stage1_channel_m = 64
stage2_channel_m = 128
stage3_channel_m = 256
stage4_channel_m = 512

stage1_hw = 80
stage2_hw = 40
stage3_hw = 20
stage4_hw = 10


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.contiguous().view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

eps = 1e-12


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                nn.AdaptiveAvgPool1d,
                nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class EmRouting2d(nn.Module):
    def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
        super(EmRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.psize = caps_size
        self.mat_dim = int(caps_size ** 0.5)

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.mat_dim, self.mat_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.beta_u = nn.Parameter(torch.FloatTensor(1, 1, B, 1))
        self.beta_a = nn.Parameter(torch.FloatTensor(1, 1, B))
        nn.init.constant_(self.beta_u, 0)
        nn.init.constant_(self.beta_a, 0)

        self.final_lambda = final_lambda
        self.ln_2pi = math.log(2 * math.pi)
        self.initialize()

    def m_step(self, v, a_in, r):
        # v: [b, l, kkA, B, psize]
        # a_in: [b, l, kkA]
        # r: [b, l, kkA, B, 1]
        b, l, _, _, _ = v.shape

        a = a_in.view(b, l, -1, 1, 1)

        # r: [b, l, kkA, B, 1]
        r = r * a_in.view(b, l, -1, 1, 1)
        # r_sum: [b, l, 1, B, 1]
        r_sum = r.sum(dim=2, keepdim=True)
        # coeff: [b, l, kkA, B, 1]
        coeff = r / (r_sum + eps)

        # mu: [b, l, 1, B, psize]
        mu = torch.sum(coeff * v, dim=2, keepdim=True)
        # sigma_sq: [b, l, 1, B, psize]
        sigma_sq = torch.sum(coeff * (v - mu) ** 2, dim=2, keepdim=True) + eps

        # [b, l, B, 1]
        r_sum = r_sum.squeeze(2)
        # [b, l, B, psize]
        sigma_sq = sigma_sq.squeeze(2)
        # [1, 1, B, 1] + [b, l, B, psize] * [b, l, B, 1]
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        # cost_h = (torch.log(sigma_sq.sqrt())) * r_sum

        # [b, l, B]
        a_out = torch.sigmoid(self.lambda_ * (self.beta_a - cost_h.sum(dim=3)))
        # a_out = torch.sigmoid(self.lambda_*(-cost_h.sum(dim=3)))

        return a_out, mu, sigma_sq

    def e_step(self, v, a_out, mu, sigma_sq):
        b, l, _ = a_out.shape
        # v: [b, l, kkA, B, psize]
        # a_out: [b, l, B]
        # mu: [b, l, 1, B, psize]
        # sigma_sq: [b, l, B, psize]

        # [b, l, 1, B, psize]
        sigma_sq = sigma_sq.unsqueeze(2)

        ln_p_j = -0.5 * torch.sum(torch.log(sigma_sq * self.ln_2pi), dim=-1) \
                 - torch.sum((v - mu) ** 2 / (2 * sigma_sq), dim=-1)

        # [b, l, kkA, B]
        ln_ap = ln_p_j + torch.log(a_out.view(b, l, 1, self.B))
        # [b, l, kkA, B]
        r = torch.softmax(ln_ap, dim=-1)
        # [b, l, kkA, B, 1]
        return r.unsqueeze(-1)

    def forward(self, a_in, pose):
        # pose: [batch_size, A, psize]
        # a: [batch_size, A]
        batch_size = a_in.shape[0]

        # a: [b, A, h, w]
        # pose: [b, A*psize, h, w]
        b, _, h, w = a_in.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(batch_size, l, self.kkA, self.mat_dim, self.mat_dim).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W)

        # [b, l, kkA, B, psize]
        v = pose_out.view(batch_size, l, self.kkA, self.B, -1)

        # [b, kkA, l]
        a_in = F.unfold(a_in, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a_in = a_in.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a_in = a_in.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA]
        a_in = a_in.view(b, l, self.kkA)

        r = a_in.new_ones(batch_size, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            # this is from open review
            self.lambda_ = self.final_lambda * (1 - 0.95 ** (i + 1))
            a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
            if i < self.iters - 1:
                r = self.e_step(v, a_out, pose_out, sigma_sq)

        # [b, l, B*psize]
        pose_out = pose_out.squeeze(2).view(b, l, -1)
        # [b, B*psize, l]
        pose_out = pose_out.transpose(1, 2)
        # [b, B, l]
        a_out = a_out.transpose(1, 2).contiguous()

        oh = ow = math.floor(l ** (1 / 2))

        a_out = a_out.view(b, -1, oh, ow)
        pose_out = pose_out.view(b, -1, oh, ow)

        return a_out, pose_out

    def initialize(self):
        weight_init(self)


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                )

    def forward(self, x):
        return self.conv(x)

## original reduction=16
## original reduction=8 0.058
class SE_block(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=8):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_inter = convbnrelu(in_channel, out_channel)
        self.fc = nn.Sequential(nn.Linear(out_channel, out_channel // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(out_channel // reduction, out_channel),
                                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_inter(x)
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y
        return out

class Superpixel_Injector_v5(nn.Module):
    def __init__(self, in_channel, out_channel, hw):
        super(Superpixel_Injector_v5, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_rgb = convbnrelu(2 * out_channel, out_channel)
        self.conv_mask = convbnrelu(out_channel, out_channel)
        self.hw = hw
        self.pool2 = nn.AvgPool2d((2, 2), stride=2)
        self.se_block = SE_block(in_channel, out_channel)
        self.out_se_block = SE_block(out_channel, out_channel)

    def forward(self, x_prompt, x_main, x_before):
        B, C, H, W = x_main.shape
        x_main_reshape = rearrange(x_main, 'b c h w -> b c (h w)')
        x_main_seg = get_graph_feature(x_main_reshape, k = 10)
        x_main_seg = self.conv_rgb(x_main_seg)
        x_main_seg = x_main_seg.max(dim=-1, keepdim=False)[0]
        x_main_seg = x_main_seg.view(B, C, H, W)
        mask_main = torch.sigmoid(x_main_seg)
        out = self.conv_mask(x_main * mask_main) + x_prompt
        if x_before != None:
            out = self.out_se_block(out + self.pool2(self.se_block(x_before)))
        else:
            out = self.out_se_block(out)

        return out

class CapsulesEM_Decoder_v4(nn.Module):
    def __init__(self, in_c1, in_c2, in_c3, in_c4):
        super(CapsulesEM_Decoder_v4, self).__init__()
        self.conv_trans1 = convbnrelu((in_c1 + in_c2), in_c2)
        self.conv_trans2 = convbnrelu((in_c3 + in_c4), in_c2)
        self.num_caps = 8
        planes = 16

        self.conv_m = nn.Conv2d(in_c2, self.num_caps, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(in_c2, self.num_caps * 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_m = nn.BatchNorm2d(self.num_caps)
        self.bn_pose = nn.BatchNorm2d(self.num_caps * 16)
        self.emrouting = EmRouting2d(self.num_caps, self.num_caps, 16, kernel_size=3, stride=2, padding=1)
        self.bn_caps = nn.BatchNorm2d(self.num_caps * planes)

        self.conv_rec = convbnrelu(self.num_caps * planes, in_c2)

        self.conv1 = convbnrelu(in_c1 + in_c2, in_c1)
        self.conv2 = convbnrelu(in_c2 + in_c2, in_c2)
        self.conv3 = convbnrelu(in_c3 + in_c2, in_c3)
        self.conv4 = convbnrelu(in_c4 + in_c2, in_c4)

        self.se_block1 = SE_block(in_c1, in_c1)
        self.se_block2 = SE_block(in_c2, in_c2)
        self.se_block3 = SE_block(in_c3, in_c3)
        self.se_block4 = SE_block(in_c4, in_c4)


    def forward(self, input1, input2, input3, input4):
        _, _, h1, w1 = input1.size()  # (80, 80)
        _, _, h2, w2 = input2.size()  # (40, 40)
        _, _, h3, w3 = input3.size()  # (20, 20)
        _, _, h4, w4 = input4.size()  # (10, 10)

        input1_down = F.interpolate(input1, size=(h2, w2), mode='bilinear', align_corners=True)
        input4_up = F.interpolate(input4, size=(h3, w3), mode='bilinear', align_corners=True)

        input_fuse1 = torch.cat((input1_down, input2), dim=1)
        input_fuse1 = self.conv_trans1(input_fuse1)

        input_fuse2 = torch.cat((input3, input4_up), dim=1)
        input_fuse2 = self.conv_trans2(input_fuse2)
        input_fuse2 = F.interpolate(input_fuse2, size=(h2, w2), mode='bilinear', align_corners=True)

        input_fuse = input_fuse1 + input_fuse2

        m, pose = self.conv_m(input_fuse), self.conv_pose(input_fuse)
        m, pose = torch.sigmoid(self.bn_m(m)), self.bn_pose(pose)
        m, pose = self.emrouting(m, pose)
        pose = self.bn_caps(pose)

        pose = self.conv_rec(pose)

        pose1_align = F.interpolate(pose, size=(h1, w1), mode='bilinear', align_corners=True)
        pose2_align = F.interpolate(pose, size=(h2, w2), mode='bilinear', align_corners=True)
        pose3_align = F.interpolate(pose, size=(h3, w3), mode='bilinear', align_corners=True)
        pose4_align = F.interpolate(pose, size=(h4, w4), mode='bilinear', align_corners=True)

        out1 = torch.cat((input1, pose1_align), dim=1)
        out2 = torch.cat((input2, pose2_align), dim=1)
        out3 = torch.cat((input3, pose3_align), dim=1)
        out4 = torch.cat((input4, pose4_align), dim=1)

        out1 = self.conv1(out1) + self.se_block1(input1)
        out2 = self.conv2(out2) + self.se_block2(input2)
        out3 = self.conv3(out3) + self.se_block3(input3)
        out4 = self.conv4(out4) + self.se_block4(input4)

        return out1, out2, out3, out4

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()

        self.stage12 = convbnrelu(out_channel, out_channel, s=1, d=2, p=2)
        self.fuse1 = convbnrelu(out_channel, out_channel, k=1, s=1, p=0, relu=True)

        self.conv_inter = convbnrelu(in_channel, out_channel)
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input1, input2=None):
        if input2 == None:
            x12 = self.stage12(input1)
            out = self.fuse1(input1 + x12)
        else:
            input2 = self.up_sample2(self.conv_inter(input2))
            x12 = self.stage12(input1)
            out = self.fuse1(x12 + input2)
        return out


class prompt_model_4_v5(nn.Module):
    def __init__(self,):
        super(prompt_model_4_v5, self).__init__()
        # Backbone model
        self.main = DFormer_Base(pretrained=True)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.injector1 = Superpixel_Injector_v5(stage1_channel_m, stage1_channel_m, stage1_hw)
        self.injector2 = Superpixel_Injector_v5(stage1_channel_m, stage2_channel_m, stage2_hw)
        self.injector3 = Superpixel_Injector_v5(stage2_channel_m, stage3_channel_m, stage3_hw)
        self.injector4 = Superpixel_Injector_v5(stage3_channel_m, stage4_channel_m, stage4_hw)

        self.capsules_decoder = CapsulesEM_Decoder_v4(stage1_channel_m, stage2_channel_m, stage3_channel_m, stage4_channel_m)

        self.decoder1 = Decoder(stage2_channel_m, stage1_channel_m)
        self.decoder2 = Decoder(stage3_channel_m, stage2_channel_m)
        self.decoder3 = Decoder(stage4_channel_m, stage3_channel_m)
        self.decoder4 = Decoder(stage4_channel_m, stage4_channel_m)

        self.Head1 = SalHead(stage1_channel_m)
        self.Head2 = SalHead(stage2_channel_m)
        self.Head3 = SalHead(stage3_channel_m)
        self.Head4 = SalHead(stage4_channel_m)

    def forward(self, x_rgb, x_depth, prompt_list):
        main_list = self.main(x_rgb, x_depth)

        injector_d2r1 =  self.injector1(prompt_list[0], main_list[0], None)
        injector_d2r2 =  self.injector2(prompt_list[1], main_list[1], injector_d2r1)
        injector_d2r3 =  self.injector3(prompt_list[2], main_list[2], injector_d2r2)
        injector_d2r4 =  self.injector4(prompt_list[3], main_list[3], injector_d2r3)

        x_fuse1, x_fuse2, x_fuse3, x_fuse4 = self.capsules_decoder(injector_d2r1, injector_d2r2, injector_d2r3, injector_d2r4)

        fuse_4 = self.decoder4(x_fuse4, None)
        fuse_3 = self.decoder3(x_fuse3, fuse_4)
        fuse_2 = self.decoder2(x_fuse2, fuse_3)
        fuse_1 = self.decoder1(x_fuse1, fuse_2)

        fuse_1 = self.upsample4(self.Head1(fuse_1))
        fuse_2 = self.upsample8(self.Head2(fuse_2))
        fuse_3 = self.upsample16(self.Head3(fuse_3))
        fuse_4 = self.upsample32(self.Head4(fuse_4))
        # print("fuse_1", fuse_1.shape)
        # print("fuse_2", fuse_2.shape)
        # print("fuse_3", fuse_3.shape)
        # print("fuse_4", fuse_4.shape)

        return fuse_1, fuse_2, fuse_3, fuse_4
        # return fuse_1
#
#

if __name__ == '__main__':
    # input_rgb = torch.randn(2, 3, 320, 320)
    # input_depth = torch.randn(2, 1, 320, 320)

    input_rgb = torch.randn(2, 384, 8, 8)
    # input_depth = torch.randn(2, 384, 8, 8)
    # input1 = torch.randn(8, 1, 6400, 64)
    # input2 = torch.randn(8, 2, 1600, 64)
    # input5 = torch.randn(8, 5, 400, 64)
    # input8 = torch.randn(8, 8, 100, 64)
    # net = AxialAttention(dim=384, dim_index=1)
    # # net = prompt_model_4()
    # out = net(input_rgb)
    # print("out", out.shape)
    # print("out1", out[0].shape)
    # print("out2", out[1].shape)
    # print("out3", out[2].shape)
    # print("out4", out[3].shape)
    #
    #
    a = torch.randn(1, 3, 320, 320).cuda()
    b = torch.randn(1, 1, 320, 320).cuda()
    model = prompt_model_4_v5().cuda()
    net = shunted_b().cuda()
    prompt_list = net(a)
    from FLOP import CalParams_three, CalParams

    CalParams_three(model, a, b, prompt_list)
    print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
