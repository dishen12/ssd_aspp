import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
"""
extra layer 中存在两个问题：
1.每次conv后没有relu激活
2.batchNormalation没有启用
但是貌似跑benchmark的时候，都没有启用，之后刷指标可以考虑用下
考虑RFBnet中dilate conv前边加的conv的作用
"""
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Aspp_a(nn.Module):
    """
    并联操作的aspp，每个branch前边加conv1x1（保持stride），这里的stride是为了保持feature map大小
    """
    def __init__(self,in_planes,out_planes,stride=1,scale=0.1,rate=[6,3,2,1]):
        #rate can also be [1,2,3], [3,4,5], [1,2,5], [5,9,17], [1,2,5,9],try it one by one, or just concat some of them to test the map,go go go ,just move on 
        super(Aspp_a,self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        self.rate = rate
        inter_planes = in_planes // 8
        """这里的降采样要认真考虑下
        """
        if(len(rate)==4):
            self.branch0 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0], relu=False)
                    )
            self.branch1 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1], relu=False)
                    )
            self.branch2 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2], relu=False)
                    )
            self.branch3 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[3], dilation=rate[3], relu=False)
            )
            self.ConvLinear = BasicConv(8*inter_planes,out_planes,kernel_size=1,stride=1,relu=False)
            self.shortcut = BasicConv(in_planes,out_planes,kernel_size=1,stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)
        elif(len(rate)==3):
            self.branch0 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0], relu=False)
                    )
            self.branch1 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1], relu=False)
                    )
            self.branch2 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2], relu=False)
                    )
            self.ConvLinear = BasicConv(6*inter_planes,out_planes,kernel_size=1,stride=1,relu=False)
            self.shortcut = BasicConv(in_planes,out_planes,kernel_size=1,stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)
        else:
                print("error! the rate is incorrect!")
    def forward(self,x):
        if(len(self.rate)==4):
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            out = torch.cat((x0,x1,x2,x3),1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out*self.scale + short
            out = self.relu(out)
            return out
        elif(len(self.rate)==3):
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = torch.cat((x0,x1,x2),1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out*self.scale + short
            out = self.relu(out)
            return out
        else:
            print("error!")
            return 

class Aspp_a_1(nn.Module):
    """
    并联操作的aspp, 先统一加conv 1x1，然后再并联
    """
    def __init__(self,in_planes,out_planes,stride=1,scale=0.1,rate=[6,3,2,1]):
        #rate can also be [1,2,3], [3,4,5], [1,2,5], [5,9,17], [1,2,5,9],try it one by one, or just concat some of them to test the map,go go go ,just move on 
        super(Aspp_a_1,self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        self.rate = rate
        inter_planes = in_planes // 8
        """这里的降采样要认真考虑下
        """
        self.conv1 = BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride)
        if(len(rate)==4):
            self.branch0 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0], relu=False)
            self.branch1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1], relu=False)
            self.branch2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2], relu=False)
            self.branch3 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[3], dilation=rate[3], relu=False)
            self.ConvLinear = BasicConv(8*inter_planes,out_planes,kernel_size=1,stride=1,relu=False)
            self.shortcut = BasicConv(in_planes,out_planes,kernel_size=1,stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)
        elif(len(rate)==3):
            self.branch0 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0], relu=False)
            self.branch1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1], relu=False)
            self.branch2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2], relu=False)
            self.ConvLinear = BasicConv(6*inter_planes,out_planes,kernel_size=1,stride=1,relu=False)
            self.shortcut = BasicConv(in_planes,out_planes,kernel_size=1,stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)
        else:
                print("error! the rate is incorrect!")
    def forward(self,x):
        x = self.conv1(x)
        if(len(self.rate)==4):
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            out = torch.cat((x0,x1,x2,x3),1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out*self.scale + short
            out = self.relu(out)
            return out
        elif(len(self.rate)==3):
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = torch.cat((x0,x1,x2),1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out*self.scale + short
            out = self.relu(out)
            return out
        else:
            print("error!")
            return         
        
class Aspp_b_1(nn.Module):
    """
    串联加并联的操作的aspp
    """
    def __init__(self,in_planes,out_planes,stride=1,scale=0.1,rate=[6,3,2,1]):
        #rate 1 2 5   9
        #     2 4 10  18
        #     3 6 15  27
        super(Aspp_b_1,self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        self.rate = rate
        inter_planes = in_planes // 8
        if(len(rate)==4):
            self.branch0 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[0], dilation=2*rate[0], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[0], dilation=3*rate[0], relu=False)
                    )
            self.branch1 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[1], dilation=2*rate[1], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[1], dilation=3*rate[1], relu=False)  
                    )
            self.branch2 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[2], dilation=2*rate[2], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[2], dilation=3*rate[2], relu=False)
                    )
            self.branch3 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[3], dilation=rate[3], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[3], dilation=2*rate[3], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[3], dilation=3*rate[3], relu=False)
            )
            self.ConvLinear = BasicConv(8*inter_planes,out_planes,kernel_size=1,stride=1,relu=False)
            self.shortcut = BasicConv(in_planes,out_planes,kernel_size=1,stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)
        elif(len(rate)==3):
            self.branch0 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[0], dilation=2*rate[0], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[0], dilation=3*rate[0], relu=False)
                    )
            self.branch1 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[1], dilation=2*rate[1], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[1], dilation=3*rate[1], relu=False)
                    )
            self.branch2 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[2], dilation=2*rate[2], relu=False),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[2], dilation=3*rate[2], relu=False)
                    )
            self.ConvLinear = BasicConv(6*inter_planes,out_planes,kernel_size=1,stride=1,relu=False)
            self.shortcut = BasicConv(in_planes,out_planes,kernel_size=1,stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)
        else:
                print("error! the rate is incorrect!")
    def forward(self,x):
        # some thing there
        if(len(self.rate)==4):
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            out = torch.cat((x0,x1,x2,x3),1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out*self.scale + short
            out = self.relu(out)
            return out
        elif(len(self.rate)==3):
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = torch.cat((x0,x1,x2),1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out*self.scale + short
            out = self.relu(out)
            return out
        else:
            print("error!")
            return 

class Aspp_b_2(nn.Module):
    """
    串联加并联的操作的aspp,每层延伸出去，相当于一个fpn
    """
    def __init__(self,in_planes,out_planes,stride=1,scale=0.1,rate=[6,3,2,1]):
        #rate 1 2 5   9
        #     2 4 10  18
        #     3 6 15  27
        super(Aspp_b_2,self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        self.rate = rate
        inter_planes = in_planes // 8   # 后边这个值，考虑微调
        if(len(rate)==4):
            self.branch0 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0], relu=False)
            )
            self.branch0_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[0], dilation=2*rate[0], relu=False)
            self.branch0_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[0], dilation=3*rate[0], relu=False)
            self.branch1 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1], relu=False))
            self.branch1_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[1], dilation=2*rate[1], relu=False)
            self.branch1_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[1], dilation=3*rate[1], relu=False)  
            self.branch2 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2], relu=False))
            self.branch2_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[2], dilation=2*rate[2], relu=False)
            self.branch2_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[2], dilation=3*rate[2], relu=False)
            self.branch3 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[3], dilation=rate[3], relu=False))
            self.branch3_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[3], dilation=2*rate[3], relu=False)
            self.branch3_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[3], dilation=3*rate[3], relu=False)
            self.ConvLinear = BasicConv(24*inter_planes,out_planes,kernel_size=1,stride=1,relu=False)
            self.shortcut = BasicConv(in_planes,out_planes,kernel_size=1,stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)
        elif(len(rate)==3):
            self.branch0 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0], relu=False)
            )
            self.branch0_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[0], dilation=2*rate[0], relu=False)
            self.branch0_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[0], dilation=3*rate[0], relu=False)
            self.branch1 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1], relu=False))
            self.branch1_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[1], dilation=2*rate[1], relu=False)
            self.branch1_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[1], dilation=3*rate[1], relu=False)  
            self.branch2 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2], relu=False))
            self.branch2_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[2], dilation=2*rate[2], relu=False)
            self.branch2_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[2], dilation=3*rate[2], relu=False)
            self.ConvLinear = BasicConv(18*inter_planes,out_planes,kernel_size=1,stride=1,relu=False)
            self.shortcut = BasicConv(in_planes,out_planes,kernel_size=1,stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)
        else:
                print("error! the rate is incorrect!")
    def forward(self,x):
        # some thing there
        if(len(self.rate)==4):
            x0 = self.branch0(x)
            X01 = self.branch0_1(x0)
            x02 = self.branch0_2(x01)
            x1 = self.branch1(x)
            x11 = self.branch1_1(x1)
            x12 = self.branch1_2(x11)
            x2 = self.branch2(x)
            x21 = self.branch2_1(x2)
            x22 = self.branch2_2(x21)
            x3 = self.branch3(x)
            x31 = self.branch3_1(x3)
            x32 = self.branch3_2(x31)
            out = torch.cat((x0,x01,x02,x1,x11,x12,x2,x21,x22,x3,x31,x32),1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out*self.scale + short
            out = self.relu(out)
            return out
        elif(len(self.rate)==3):
            x0 = self.branch0(x)
            X01 = self.branch0_1(x0)
            x02 = self.branch0_2(x01)
            x1 = self.branch1(x)
            x11 = self.branch1_1(x1)
            x12 = self.branch1_2(x11)
            x2 = self.branch2(x)
            x21 = self.branch2_1(x2)
            x22 = self.branch2_2(x21)
            out = torch.cat((x0,x01,x02,x1,x11,x12,x2,x21,x22),1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out*self.scale + short
            out = self.relu(out)
            return out
        else:
            print("error!")
            return 
        
        
class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes,Rate=[6,3,2,1]):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.aspp_a_4 = Aspp_b_1(512,512,stride=1,scale=1,rate=Rate)
        self.aspp_a_7 = Aspp_b_1(1024,1024,stride=1,scale=1,rate=Rate)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        #print("the size of conv4_3 is :",x.size())
        s = self.aspp_a_4(x)
        #s = Aspp_a(x.size(1),x.size(1),stride=1,scale=1,rate=[6,3,2,1])(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        #print("the size of conv7_3 is :",x.size())
        s = self.aspp_a_7(x)
        #s = Aspp_a(x.size(1),x.size(1),stride=1,scale=1,rate=[6,3,2,1])(x)
        #s = x
        sources.append(s)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':  #[256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def add_extras_aspp(cfg, i, batch_norm=False,Rate=[6,3,2,1]):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':  #[256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
                layers += [Aspp_b_1(in_channels,cfg[k+1],stride=2,scale=1,rate=Rate)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21,rate="6,3,2,1"):
    Rate = [int(i) for i in rate.strip().split(",")]
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3), #输入，三通道图片
                                     add_extras_aspp(extras[str(size)], 1024,Rate),#vgg最后一层1024
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes,Rate)
