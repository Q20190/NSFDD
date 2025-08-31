from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch
from block import encoder_V2
from block import DANetHead
from block import SE_block
from block import HAM
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):  # 修改部分代码添加SEblock
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )  # imgsize=（imgsize+2*padding-kernalsize）/stride+1图像经过卷积之后的大小保持一致
        self.up = up_conv(in_channels + skip_channels, out_channels)
        self.da1 = DANetHead(64, 64)
        self.da2 = DANetHead(128, 128)
        self.da3 = DANetHead(256, 256)
        self.da4 = DANetHead(512, 512)  # 图像的大小经过DA处理之后始终保持不变
        self.se = SE_block(in_channels + skip_channels, ratio=16)
        self.ham1 = HAM(64)
        self.ham2 = HAM(128)
        self.ham3 = HAM(256)
        self.ham4 = HAM(512)
        self.da_list = {64: self.da1,
                        128: self.da2,
                        256: self.da3,
                        512: self.da4}
        self.ham_list = {64: self.ham1,
                         128: self.ham2,
                         256: self.ham3,
                         512: self.ham4}

    def forward(self, x, skip=None, n_skip=None, attn_type=None):
        x = self.up(x)
        if skip is not None:
            if n_skip == 4:
                target_channels = [512, 256, 128, 64]
            elif n_skip == 3:
                target_channels = [512, 256, 128]
            elif n_skip == 2:
                target_channels = [512, 256]
            elif n_skip == 1:
                target_channels = [512]
            else:
                target_channels = []
            if target_channels:
                for channel in target_channels:
                    if skip.size(1) and x.size(1) == channel:
                        if attn_type == 'DA':
                            skip = self.da_list[channel](skip)
                        elif attn_type == 'HAM':
                            skip = self.ham_list[channel](skip)
            x = torch.cat([x, skip], dim=1)
            x = self.se(x)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        in_channels = config.decoder_channels  # (512，256, 128, 64)
        out_channels = config.decoder_channels  # [512, 256, 64, 16]

        # if self.config.n_skip != 0:
        #     skip_channels = self.config.skip_channels#[512, 256, 128,64]
        #     for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
        #         skip_channels[3-i]=0
        # else:
        #     skip_channels=[0,0,0,0]

        self.attn_type = config.attn_type
        skip_channels = self.config.skip_channels
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
            # 构建解码路径
        ]
        self.blocks = nn.ModuleList(blocks)
        self.n_skip = config.n_skip

    def forward(self, x, features=None):
        B, c, h, w = x.size()
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i]
            else:
                skip = None
            x = decoder_block(x, skip=skip, n_skip=self.n_skip, attn_type=self.attn_type)
        return x


class block(nn.Module):
    def __init__(self,in_filters,n_filters):
        super(block,self).__init__()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_filters, n_filters, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU())
    def forward(self, x):
        x=self.deconv1(x)
        return x

class generator(nn.Module):
    # initializers
    def __init__(self,config, n_classes, n_filters=32):
        super(generator, self).__init__()
        self.num_classes = n_classes
        self.encoder = encoder_V2(in_ch=config.input_chanel)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.n_classes,
            kernel_size=3,
        )
        self.config = config
        self.se2 = SE_block(64, ratio=16)
        self.dropout = Dropout(0.05)
        self.input_chanel = config.input_chanel

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, self.input_chanel, 1, 1)
        x1, features = self.encoder(x)
        x1 = self.decoder(x1, features)
        x3 = x1
        x3 = self.se2(x3)
        x3 = self.dropout(x3)
        logits = self.segmentation_head(x3)

        return logits

class discriminator(nn.Module):
    def __init__(self,n_filters):
        super(discriminator,self).__init__()
        self.down1 = nn.Sequential(
            block(2, n_filters),
            block(n_filters, n_filters),
            nn.MaxPool2d((2, 2)))
        self.down2 = nn.Sequential(
            block(n_filters, 2 * n_filters),
            block(2 * n_filters, 2 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down3 = nn.Sequential(
            block(2 * n_filters, 4 * n_filters),
            block(4 * n_filters, 4 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down4 = nn.Sequential(
            block(4 * n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down5 = nn.Sequential(
            block(8 * n_filters, 16 * n_filters),
            block(16 * n_filters, 16 * n_filters))
        self.out = nn.Linear(16*n_filters,1)
    def forward(self, x):
        x=self.down1(x)
        #print(x.size())
        x = self.down2(x)
        #print(x.size())
        x = self.down3(x)
        #print(x.size())
        x = self.down4(x)
        #print(x.size())
        x = self.down5(x)
        #print(x.size())
        x=F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size()[0], -1)

        x=self.out(x)
        x = F.sigmoid(x)

        return x#b,1


if __name__=='__main__':
    D=discriminator(32).cuda()
    t=torch.ones((2,2,512,512)).cuda()
    print(D(t).size())