import torch
import torch.nn as nn
from torch.nn import Dropout,Conv1d
import torch.nn.functional as F
import math
import numpy as np

from torch.nn.utils import spectral_norm as SN
class SN_Linear_Block(nn.Module):
    def __init__(self, dim, emb):
        super(SN_Linear_Block, self).__init__()
        self.line_layer = SN(nn.Linear(dim, emb))
    def forward(self, x):
        x = x.reshape(x.shape[0],  -1)  #
        x = self.line_layer(x)
        return x
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

class SN_Conv1D_Norm_Act(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, act_fn=None):
        super(SN_Conv1D_Norm_Act, self).__init__()
        self.act_fn = act_fn
        self.conv_block = nn.ModuleList()
        self.conv_block.add_module("conv0",SN(nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                                                                  stride=stride, padding=padding)))
        self.conv_block.apply(init_weights)
    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, c_in, c_hid,  compress=True):
        super(Encoder, self).__init__()
        self.pre_block = SN_Conv1D_Norm_Act(c_in, c_hid)
        self.line = SN(nn.Linear(c_hid, c_hid))
        self.compress = compress

    # B len dim
    def forward(self, x):
        x = self.pre_block(x)  # B dim len
        if self.compress:
            x = F.avg_pool1d(x, kernel_size=2, ceil_mode=True)  # len/2
        x = x.transpose(1, 2)
        x = self.line(x)
        x = x.transpose(1, 2)
        return x  # B out len//2

class SN_Conv2D_Norm_Act(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, pool_height,pool_width,act_fn=None, norm_fn=None,dropout=0.1):
        super(SN_Conv2D_Norm_Act, self).__init__()
        self.conv_block = nn.ModuleList()
        self.conv_block.add_module("conv0", SN(nn.Conv2d(c_in, c_out, kernel_size=kernel_size,stride=stride, padding=padding)))
        self.conv_block.add_module("drop0",Dropout(dropout))
        if act_fn is not None:
            self.conv_block.add_module("act0", act_fn)
        if norm_fn is not None:
            self.conv_block.add_module("BN0", nn.BatchNorm2d(c_out))
        self.conv_block.add_module("ada_avgpool0",nn.AdaptiveAvgPool2d((pool_height, pool_width)))
    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        return x

### 子带特征处理 B dim len , split len and dim
### 对于任意长的语音采用压缩编码，对于
### # 5 128 cont_emb 5 128 len16
class ContentDiscrimer(nn.Module):
    def __init__(self,c_in=80,c_hid=160,c_out=32):
        super(ContentDiscrimer, self).__init__()
        self.pre_block = nn.Sequential(
            SN_Conv1D_Norm_Act(c_in, c_hid,kernel_size=3,stride=1,padding=1,act_fn=nn.ReLU()),
            SN_Conv1D_Norm_Act(c_hid, c_hid,kernel_size=3,stride=1,padding=1,act_fn=nn.ReLU()),

        )
        self.encoder_block = nn.Sequential(
            Encoder(c_hid, c_hid, compress=True),
            Encoder(c_hid,c_hid//2,compress=True),
            Encoder(c_hid//2,c_hid//4,compress=False),
            Encoder(c_hid//4,c_hid,compress=True),
        )
        self.fusion_block = nn.Sequential(

            Encoder(c_hid, c_hid, compress=True),
            Encoder(c_hid, c_hid, compress=False),
            Encoder(c_hid,c_hid//4,compress=False),
            Encoder(c_hid//4,c_hid,compress=True)
        )
        self.post_block = nn.Sequential(
            Encoder(c_hid, c_hid, compress=True),
            Encoder(c_hid, c_hid, compress=False),
            Encoder(c_hid, c_hid // 4, compress=False),
            Encoder(c_hid // 4, c_hid, compress=True)
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.line = SN(nn.Linear(c_hid, c_out))
    # cond B 1 256
    # cont_emb 5 128 len16
    def forward(self,x,cont_emb):
        x = self.pre_block(x)
        dlen = x.shape[2]
        seg_list = [math.ceil(dlen * index) for index in [0, 0.25, 0.5, 0.75, 1]]
        seg_emb_list = []
        for index in range(len(seg_list) - 1):
            start = seg_list[index]
            end = seg_list[index + 1]
            x_seg = x[:, :, start:end]
            x_seg_emb = self.encoder_block(x_seg)
            seg_emb_list.append(x_seg_emb)
        seg_cat_emb = torch.cat(seg_emb_list,dim=2)
        clip_feat = torch.cat([seg_cat_emb,cont_emb],dim=2)
        clip_feat = self.post_block(clip_feat)
        clip_avg = self.avg(clip_feat)  # B dim 1
        clip_avg = clip_avg.transpose(1, 2) # B 1 dim
        clip_out = self.line(clip_avg)# B 1 dim
        return clip_out

class SpeakerDiscrimer(nn.Module):
    def __init__(self,c_in=80,c_hid=256,c_out=32):
        super(SpeakerDiscrimer, self).__init__()
        bank_size = 4
        bank_scale = 1
        c_bank = 32
        self.conv_bank = nn.ModuleList([SN(nn.Conv1d(c_in, c_bank, kernel_size=k)) for k in range(bank_scale, bank_size+1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.pre_block = nn.Sequential(
            SN_Conv1D_Norm_Act(in_channels, c_hid)
        )
        self.encoder_block = nn.Sequential(
            Encoder(c_hid, c_hid, compress=False),
            Encoder(c_hid,c_hid//4,compress=True),
            Encoder(c_hid//4,c_hid//4,compress=True),
            Encoder(c_hid//4,c_hid,compress=True),
            Encoder(c_hid,c_hid,compress=False),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.line = SN(nn.Linear(c_hid, c_out))
        self.last_line1 = SN(nn.Linear(c_out*4+256, c_out))
        self.last_line2 = SN(nn.Linear(c_out, c_out))

    # spk B 128
    def forward(self,x,spk_emb):
        x = x.transpose(1,2)
        x = conv_bank(x, self.conv_bank, act=nn.ReLU())
        x = self.pre_block(x) # B hid len
        xlen = x.shape[2]
        seg_list = [xlen * index for index in [0.25, 0.5, 0.75, 1]]
        seg_emb_list = []
        seg_emb_list.append(spk_emb.unsqueeze(1))
        for seg_point in seg_list:
            seg = x[:, :, :math.ceil(seg_point)] # B 128 len
            seg_emb = self.encoder_block(seg)
            clip_avg = self.avg(seg_emb)  # B dim 1
            clip_avg = clip_avg.transpose(1, 2)  # B 1 dim
            clip_out = self.line(clip_avg)  # B 1 dim
            seg_emb_list.append(clip_out)
        fusion_emb = torch.cat(seg_emb_list,dim=2)
        out_emb = self.last_line2(torch.sigmoid(self.last_line1(fusion_emb)))
        return out_emb

class FakeDiscrimer(nn.Module):
    def __init__(self):
        super(FakeDiscrimer, self).__init__()
        self.disc_block = nn.Sequential(
            SN_Conv2D_Norm_Act(1, 128, 3, 1, 1, 40, 64, act_fn=nn.ReLU(), norm_fn="BN", dropout=0.2),
            SN_Conv2D_Norm_Act(128, 64, 3, 1, 1, 20, 32, act_fn=nn.ReLU(), norm_fn="BN", dropout=0.2),
            SN_Conv2D_Norm_Act(64, 32, 3, 1, 1, 10, 16, act_fn=nn.ReLU(), norm_fn="BN", dropout=0.2),
            SN_Conv2D_Norm_Act(32, 8, 3, 1, 1, 5, 8, act_fn=nn.ReLU(), norm_fn="BN", dropout=0.2),
            SN_Linear_Block(320, 32),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.transpose(1,2)
        x = x.unsqueeze(1) # B 1 dim len
        emb = self.disc_block(x)
        return emb

def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if
                  "auxiliary" not in name) / 1e6

if __name__ == '__main__':
    # x = torch.randn(5, 128, 80)
    # cont_emb = torch.randn(5, 160, 19)

    # spk_emb = torch.randn(5,256)
    # sdis = SpeakerDiscrimer()
    # print("model params:", count_parameters_in_M(sdis))
    # clip_out = sdis(x,spk_emb)
    # print("output:",clip_out.shape)

    x = torch.randn(5, 128, 80)
    fakeDisc = FakeDiscrimer()
    print("model params:", count_parameters_in_M(fakeDisc))
    clip_out = fakeDisc(x)
    print("output:", clip_out.shape)

