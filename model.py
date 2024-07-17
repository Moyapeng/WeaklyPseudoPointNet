import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from options import Options
from torchsummary import summary



def split_patch(img, patch_size):
    # Unfold the image into patches
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Rearrange the dimensions to get patches in the form of a batch
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(-1, img.size(1), patch_size, patch_size)
    return patches


def merge_patch(patches, img_size, patch_size):
    # Get the number of patches along height and width
    num_h_patches = img_size[2] // patch_size
    num_w_patches = img_size[3] // patch_size

    # Reshape patches to merge them back
    patches = patches.view(-1, num_h_patches, num_w_patches, patches.size(1), patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    img = patches.view(-1, img_size[1], img_size[2], img_size[3])
    return img

class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """

    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, in_channels)
        )

    def forward(self, seg, gauss):
        q = self.q_conv(seg)
        k = self.k_conv(seg)
        v = self.v_conv(gauss)
        q_ = q.view(q.size(0), q.size(1), -1).permute(0, 2, 1)
        k_ = k.view(k.size(0), k.size(1), -1)
        v_ = v.view(v.size(0), v.size(1), -1).permute(0, 2, 1)

        attention_weights = self.softmax(torch.bmm(q_, k_)/torch.sqrt(torch.tensor(q_.size(-1))))
        output = torch.bmm(attention_weights, v_)
        x = v_ + output
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + ffn_output
        output = self.norm2(x)

        output = output.permute(0,2,1).view(seg.shape[0], seg.shape[1], seg.shape[-2], seg.shape[-1])

        return output



class PatchAttention(nn.Module):
    def __init__(self, in_channels, patch_size = 7):
        super(PatchAttention, self).__init__()
        self.patch_size = patch_size
        self.attention_block = AttentionBlock(in_channels)


    def forward(self, seg, gauss):
        batch_size, channels, height, width = seg.size()
        # Reshape inputs to patches
        t1 = time.time()
        # seg_patches = split_patch(seg,height,width,self.patch_size)
        # gauss_patches = split_patch(gauss,height,width,self.patch_size)

        seg_patches = split_patch(seg,  self.patch_size)
        gauss_patches = split_patch(gauss,  self.patch_size)


        # Apply attention to each patch
        # outputs = []
        # for i in range(len(seg_patches)):
        #     seg_patch = seg_patches[i]
        #     gauss_patch = gauss_patches[i]
        #     attention_output = self.attention_block(seg_patch, gauss_patch)
        #     outputs.append(attention_output)

        # Concatenate the attention outputs and reshape to the original size
        result= self.attention_block(seg_patches, gauss_patches)

        # result = merge_patch(outputs,seg.size(),self.patch_size)
        result = merge_patch(result,seg.size(),self.patch_size)
        # print(result.shape)
        t2 = time.time()
        # print('patch:',t2-t1)
        return result

class PointSegMixedLayer(nn.Module):
    def __init__(self, in_channels,num_branches,patch_size=7):
        super(PointSegMixedLayer, self).__init__()
        self.attn_seg = PatchAttention(in_channels, patch_size)
        self.num_branches = num_branches
        self.gauss_branches = nn.ModuleList([
            PatchAttention(in_channels, patch_size) for _ in range(num_branches)
        ])

    def forward(self, seg,g):
        stacked_tensors = torch.stack(g)
        g_all = torch.mean(stacked_tensors, dim=0)
        g_ = []
        for i,branch in enumerate(self.gauss_branches):
            g_.append(branch(g[i],g[i]))




        with torch.no_grad():
            g_all_copy = g_all.clone()
        seg_ = self.attn_seg(g_all_copy,seg)
        return seg_, g_

class ConvUpBlock(nn.Module):
    def __init__(self, in_channel , out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, stride=2)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate,
                                  dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        x = self.up(x)
        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        x_skip = F.pad(x_skip, (0, W_diff, 0, H_diff), mode='reflect')
        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class ConvUpLayer(nn.Module):
    def __init__(self, in_channel , out_channel, num_branches,dropout_rate=0.0, dilation=1):
        super().__init__()
        self.seg = ConvUpBlock(in_channel, out_channel,dropout_rate,dilation)
        self.num_branches = num_branches
        self.gauss_branches = nn.ModuleList([
            ConvUpBlock(in_channel, out_channel, dropout_rate,dilation) for _ in range(num_branches)
        ])

        self.attn = PointSegMixedLayer(in_channel,num_branches)

    def forward(self, x0, g, c):  #input : tensor,[tensor*num_branches],encoder

        d0,dg = self.attn(x0,g)
        d0_ = self.seg(d0,c)
        dg_=[]
        for i, branch in enumerate(self.gauss_branches):
            dg_.append(branch(dg[i],c))

        return d0_, dg_




class DecoderBranch(nn.Module):
    def __init__(self, num_branches, out_c=2):
        super(DecoderBranch, self).__init__()
        l = [64, 64, 128, 256, 512]
        self.u5 = ConvUpLayer(l[4], l[3], num_branches,dropout_rate=0.1)
        self.u6 = ConvUpLayer(l[3], l[2], num_branches,dropout_rate=0.1)
        self.u7 = ConvUpLayer(l[2], l[1], num_branches,dropout_rate=0.1)
        self.u8 = ConvUpLayer(l[1], l[0], num_branches,dropout_rate=0.1)
        self.ce = nn.ConvTranspose2d(l[0], out_c, 2, stride=2)
        self.num_branches = num_branches
        self.gauss_branches = nn.ModuleList([
            nn.ConvTranspose2d(l[0], 1, 2, stride=2) for _ in range(num_branches)
        ])

    def forward(self, x, c4, c3, c2, c1):
        d4g = [x for _ in range(self.num_branches)]
        d5s,d5g = self.u5(x,d4g,c4)   #input : tensor,[tensor*num_branches],encoder
        d6s,d6g = self.u6(d5s,d5g,c3)
        d7s,d7g = self.u7(d6s,d6g,c2)
        d8s,d8g = self.u8(d7s,d7g,c1)

        seg = self.ce(d8s)
        result = [seg]
        for i,branch in enumerate(self.gauss_branches):
            x_branch = branch(d8g[i])
            result.append(x_branch)

        return result



# Transfer Learning ResNet as Encoder part of UNet
class ResUNet34(nn.Module):
    def __init__(self, out_c=2, pretrained=True, fixed_feature=False):
        super().__init__()
        # load weight of pre-trained resnet
        self.resnet = models.resnet34(pretrained=pretrained)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # up conv
        l = [64, 64, 128, 256, 512]
        self.u5 = ConvUpBlock(l[4], l[3], dropout_rate=0.0)
        self.u6 = ConvUpBlock(l[3], l[2], dropout_rate=0.0)
        self.u7 = ConvUpBlock(l[2], l[1], dropout_rate=0.0)
        self.u8 = ConvUpBlock(l[1], l[0], dropout_rate=0.0)
        # final conv
        self.ce = nn.ConvTranspose2d(l[0], out_c, 2, stride=2)
        self.xx = nn.Conv2d(3, 3, kernel_size=1)
        self.d1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

        opt = Options(isTrain=True)
        decoder_radius = opt.train['decoder_radius']
        gbranch_num = len(decoder_radius)

        self.decoder_branches = DecoderBranch(gbranch_num)



    def forward(self, img):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        result = self.decoder_branches(x, c4, c3, c2, c1)


        return result

# from ptflops import get_model_complexity_info
# model=ResUNet34()
# flops,params = get_model_complexity_info(model,(3,224,224))
# print("flops:",flops)
# print('params:',params)
# img=torch.randn(1,3,224,224)

# result=model(img)
# for i,out in enumerate(result):
#     print(out.shape)

# model=ResUNet34()
# a=torch.randn(8,3,224,224)
# out,a,b,c,d=model(a)
# conv_up_block_weights = model.u5.conv1.conv.weight.data
# a=1
# print(out.shape,a.shape,b.shape,c.shape,d.shape)

# from thop import profile
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model=ResUNet34().to(device)
# channels,height,width,batch_size = 3,224,224,8
# summary(model, input_size=(channels, height, width))
#
# input_tensor = torch.randn(batch_size, channels, height, width).to(device)
# flops,params = profile(model, inputs=(input_tensor,))
# print(f"FLOPs: {flops}")

# from ptflops import get_model_complexity_info
# macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
#
# print("MACs=", macs )
# print("MACs=", str(macs / 1e6) )
# m = ResUNet34()
# a = torch.randn(1,3,224,224)
# time1 = time.time()
# aa = m(a)
# print(len(aa))
# print(aa[0].shape)
# print(aa[1].shape)
# print(aa[4].shape)
# time12 = time.time()
# print(time12-time1)

