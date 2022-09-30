from simsense import DepthSensor
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Input images and relative camera parameters
lr_size = (1280, 720)
k_l = np.array([
    [920., 0., 640.],
    [0., 920., 360.],
    [0., 0., 1.]
])
k_r = k_l
l2r = np.array([
    [1., 0, 0, -0.0545],
    [0, 1., 0, 0],
    [0, 0, 1., 0],
    [0, 0, 0, 1.]
])
rgb_size = (1920, 1080)
k_rgb = np.array([
    [1380., 0., 960.],
    [0., 1380., 540.],
    [0., 0., 1.]
])
l2rgb = np.array([
    [1., 0, 0, 0.0175],
    [0, 1., 0, 0],
    [0, 0, 1., 0],
    [0, 0, 0, 1.]
])
left = (plt.imread("img/left2.png") * 255).astype(np.uint8)
right = (plt.imread("img/right2.png") * 255).astype(np.uint8)

scale_percent = 50 # percent of original size
width = int(left.shape[1] * scale_percent / 100)
height = int(left.shape[0] * scale_percent / 100)
dim = (width, height)
left_half = cv2.resize(left, dim, interpolation = cv2.INTER_AREA)
right_half = cv2.resize(right, dim, interpolation = cv2.INTER_AREA)
lr_size_half = (640, 360)

for i in range(1000):
    if i % 2 == 0:
        depthSensor = DepthSensor(lr_size, k_l, k_r, l2r, rgb_size, k_rgb, l2rgb, min_depth=0.2, max_depth=2.0,
                            census_width=7, census_height=7, block_width=7, block_height=7,
                            uniqueness_ratio=15, depth_dilation=True)
        depth = depthSensor.compute(left, right)
    else:
        depthSensor = DepthSensor(lr_size_half, k_l, k_r, l2r, rgb_size, k_rgb, l2rgb, min_depth=0.2, max_depth=2.0,
                            census_width=7, census_height=7, block_width=7, block_height=7,
                            uniqueness_ratio=15, depth_dilation=True)
        depth = depthSensor.compute(left_half, right_half)

print("pass!")