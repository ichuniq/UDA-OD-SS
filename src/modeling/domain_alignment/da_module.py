import torch
import torch.nn.functional as F
from torch import nn

from .build import DAHEAD_REGISTRY
from .da_utils import GradientScalarLayer, FocalLoss
from fvcore.nn import sigmoid_focal_loss, sigmoid_focal_loss_jit

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class netD_pixel(nn.Module): 
    """ Local (strong) domain classifier """
    def __init__(self, cin, context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = conv1x1(cin, 256)
        # self.conv1 = conv1x1(256, 256)
        self.conv2 = conv1x1(256, 128)
        self.conv3 = conv1x1(128, 1)

        for l in [self.conv1, self.conv2, self.conv3]:
            torch.nn.init.normal_(l.weight, std=0.01)

        self.context = context

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          return F.sigmoid(x)


class netD(nn.Module):
    """ Global (weak) domain classifier """
    def __init__(self, cin, context=False):
        super(netD, self).__init__()
        #self.conv1 = conv3x3(512, 512, stride=2)
        self.conv1 = conv3x3(cin, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)

        self.context = context

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)  # flatten
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x, feat
        else:
          return x

class DAImgModule(torch.nn.Module):
    """
    Image-level domain adversarial training module
    """
    def __init__(self, cfg, in_channels, grl_w=1.0):
        super(DAImgModule, self).__init__()

        self.cin = in_channels
        self.da_img_head = netD(self.cin)
        # scale for img level da loss, default 0.5
        self.da_img_scale = cfg.MODEL.DA.IMG_LOSS_SCALE 
        # scale for overall da loss, default 1.0
        self.da_scale = 1.0 

        self.grl = GradientScalarLayer(-1.0 * grl_w)
        self.FL = FocalLoss(gamma=0.5)

    def forward(self, features, img_feat_level, domain):
        if self.training:
            losses = {}
            img_feat = features[img_feat_level]
            out_d = self.da_img_head(
              self.grl(img_feat)
            )
            # print(out_d.size()) # torch.Size([1, 2])
            if domain == 'source':
                d_label = torch.zeros(out_d.size(0)).long().cuda()
            else:
                d_label = torch.ones(out_d.size(0)).long().cuda()
            
            # print(out_d, d_label)
            img_d_loss = self.da_img_scale * self.FL(out_d, d_label) 
            # loss = torch.ones((1)).cuda()# sigmoid_focal_loss(x, d_label)
            losses = {'loss_img_d': img_d_loss * self.da_scale}
            return losses

        return {}


@DAHEAD_REGISTRY.register()
def build_img_da_head(cfg, input_shape):
    in_channels = input_shape[cfg.MODEL.DA.IMG_FEAT_LEVEL].channels
    return DAImgModule(cfg, in_channels)


class DASWModule(torch.nn.Module):
    """
    Pixels & Image-level domain adversarial training module like Stong-Weak DA 
    but currently without the context vector regularization)
    """
    def __init__(self, cfg, pix_cin, img_cin, grl_w=1.0):
        super(DASWModule, self).__init__()

        self.da_pix_head = netD_pixel(pix_cin)
        self.da_pix_scale = 0.5 #cfg.MODEL.DA.PIX_LOSS_SCALE 

        self.da_img_head = netD(img_cin)
        # scale for img level da loss, default 0.5
        self.da_img_scale = cfg.MODEL.DA.IMG_LOSS_SCALE 

        # scale for overall da loss, default 1.0
        self.da_scale = 1.0 

        self.grl = GradientScalarLayer(-1.0 * grl_w)
        self.FL = FocalLoss(gamma=0.5)

    def forward(self, features, pix_feat_level, img_feat_level, domain):
        if self.training:
            losses = {}

            # local
            pix_feat = features[pix_feat_level]
            out_d_pixel = self.da_pix_head(
              self.grl(pix_feat)
            )

            # global
            img_feat = features[img_feat_level]
            out_d = self.da_img_head(
              self.grl(img_feat)
            )
            # print(out_d.size()) # torch.Size([1, 2])
            if domain == 'source':
                d_label = torch.zeros(out_d.size(0)).long().cuda()
                pix_d_loss = torch.mean(out_d_pixel ** 2)
            else:
                d_label = torch.ones(out_d.size(0)).long().cuda()
                pix_d_loss = torch.mean((1 - out_d_pixel) ** 2)
            
            # print(out_d, d_label)
            pix_d_loss = self.da_pix_scale * pix_d_loss
            img_d_loss = self.da_img_scale * self.FL(out_d, d_label) 
            
            losses = {
                'loss_pix_d': pix_d_loss * self.da_scale,
                'loss_img_d': img_d_loss * self.da_scale
            }
            return losses

        return {}

@DAHEAD_REGISTRY.register()
def build_sw_da_head(cfg, input_shape):
    pix_in_channels = input_shape[cfg.MODEL.DA.PIX_FEAT_LEVEL].channels
    img_in_channels = input_shape[cfg.MODEL.DA.IMG_FEAT_LEVEL].channels
    return DASWModule(cfg, pix_in_channels, img_in_channels)