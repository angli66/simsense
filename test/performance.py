from simsense import DepthSensor
import numpy as np
import matplotlib.pyplot as plt
import time

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

# Params
max_disp = 128
block_size = (1, 1)

depthSensor = DepthSensor(lr_size, k_l, k_r, l2r, max_disp=max_disp, block_width=block_size[0], block_height=block_size[1])

print(f"Image resolution: {lr_size}")
print(f"Max disparity: {max_disp}")
print(f"Block size: {block_size}")
start = time.process_time()
for i in range(1000):
    result = depthSensor.compute(left, right)
print("Average FPS for 1000 calls of compute():", 1000 / (time.process_time() - start), "FPS")
