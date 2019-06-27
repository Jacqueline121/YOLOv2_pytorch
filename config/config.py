import numpy as np
from easydict import EasyDict

__C = EasyDict()
cfg = __C

__C.TRAIN = EasyDict()
__C.TEST = EasyDict()


__C.ANCHORS = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892),
               (9.47112, 4.84053), (11.2364, 10.0071)]

__C.STRIDE = 32

__C.CLASS_NUM = 20


__C.SATURATION = 1.5
__C.EXPOSURE = 1.5
__C.HUE = 0.1

__C.JITTER = 0.3


__C.INPUT_SIZE = (416, 416)
__C.TEST_SIZE = (416, 416)


__C.DEBUG = False

####################################
# LOSS
####################################
__C.COORD_SCALE = 1.0
__C.NO_OBJECT_SCALE = 1.0
__C.OBJECT_SCALE = 5.0
__C.CLASS_SCALE = 1.0
__C.THRESH = 0.6


