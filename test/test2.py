from simsense import DepthSensor
import numpy as np
import matplotlib.pyplot as plt

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

depthSensor = DepthSensor(lr_size, k_l, k_r, l2r, rgb_size, k_rgb, l2rgb, min_depth=0.2, max_depth=2.0,
                            census_width=7, census_height=7, block_width=7, block_height=7,
                            uniqueness_ratio=15, depth_dilation=True)
result = depthSensor.compute(left, right)

cmap = plt.cm.get_cmap('jet').copy()
cmap.set_bad(color='black')
plt.imshow(result, cmap=cmap)
plt.show()
