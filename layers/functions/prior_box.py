from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch

"""需要用到的参数：
min_dim = 300 
	"输入图最短边的尺寸"

feature_maps = [38, 19, 10, 5, 3, 1]
steps = [8, 16, 32, 64, 100, 300]
	"共有6个特征图：
	feature_maps指的是在某一层特征图中，遍历一行/列需要的步数
	steps指特征图中两像素点相距n则在原图中相距steps[k]*n
	由于steps由于网络结构所以为固定，所以作者应该是由300/steps[k]得到feature_maps"

min_sizes = [30, 60, 111, 162, 213, 264]
max_sizes = [60, 111, 162, 213, 264, 315]
	"min_sizes和max_sizes共同使用为用于计算aspect_ratios=1时
	rel size: sqrt(s_k * s_(k+1))时所用"

aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
	"各层除1以外的aspect_ratios，可以看出是各不相同的，
	这样每层特征图的每个像素点分别有[4,6,6,6,4,4]个default boxes
	作者也在原文中提到这个可以根据自己的场景适当调整"

"""

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        print("aspect_ratios is ",self.aspect_ratios)
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]  # 为毛不直接用f，后边尝试下，直接改为f
                """每个default box的中心点，从论文以及代码复现可知0<cx,cy<1
                 即对应于原图的一个比例"""
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1   小方框
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1   大方框
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios  剩余的小长方形
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        print("size of prior out is ",output.size())
        return output
