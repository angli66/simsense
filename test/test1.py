from simsense import DepthSensor
import numpy as np
import matplotlib.pyplot as plt

# Input images and relative camera parameters
lr_size = (848, 480)
k_l = np.array([
    [427.12, 0., 426.93],
    [0., 427.12, 234.78],
    [0., 0., 1.]
])
k_r = k_l
l2r = np.array([
    [1., 0, 0, -0.05],
    [0, 1., 0, 0],
    [0, 0, 1., 0],
    [0, 0, 0, 1.]
])
left = (plt.imread("img/left1.png") * 255).astype(np.uint8)
right = (plt.imread("img/right1.png") * 255).astype(np.uint8)

depthSensor = DepthSensor(lr_size, k_l, k_r, l2r, max_depth=1.3)
result = depthSensor.compute(left, right).copy()

cmap = plt.get_cmap('jet').copy()
cmap.set_bad(color='black')
plt.imshow(result, cmap=cmap)
plt.show()
