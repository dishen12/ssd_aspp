# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000,150000),  #之后此处试着调一下
    'max_iter': 200000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],  #min max 都是相对于原图size的，固定指定，之后可以考虑自适应，修改时，这块，可以先放放，毕竟dilate conv也有点pool的样子
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

voc_mid_priors = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000,150000),  #之后此处试着调一下
    'max_iter': 200000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],  #min max 都是相对于原图size的，固定指定，之后可以考虑自适应，修改时，这块，可以先放放，毕竟dilate conv也有点pool的样子
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

voc_most_priors = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000,150000),  #之后此处试着调一下
    'max_iter': 200000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],  #min max 都是相对于原图size的，固定指定，之后可以考虑自适应，修改时，这块，可以先放放，毕竟dilate conv也有点pool的样子
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# model i
voc_i = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 19, 19, 17, 15],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    #'steps': [8, 16, 16, 16, 18, 20],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],   #没用到
    'clip': True,
    'name': 'VOC',
}

# model a
voc_a = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 19, 19, 19, 19],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    #'steps': [8, 16, 16, 16, 16, 16],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# model h
voc_h = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 19, 19, 10, 5],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    #'steps': [8, 16, 16, 16, 32, 64],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
