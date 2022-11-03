> This library has been integrated into [SAPIEN](https://github.com/haosulab/SAPIEN.git). SAPIEN is a simulated part-based interactive environment equipped with several high performance modules. Combining SimSense and [Kuafu](https://github.com/jetd1/kuafu.git), SAPIEN is able to achieve real-time activestereo depth sensor simulation. Check branch `sapien` for more details.

# SimSense: A Real-Time Depth Sensor Simulator
This is a GPU-accelerated depth sensor simulator for python, implemented with CUDA. The goal of this project is to provide a real-time, highly adjustable, and easy-to-use module that can be utilized to simulate different kinds of depth sensors in the real world. An important application scene is when building a simulated environment. For example, [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning#:~:text=Reinforcement%20learning%20(RL)%20is%20an,supervised%20learning%20and%20unsupervised%20learning.) is getting dominant in robot control task. However, directly training RL algorithms on real-world robots is expensive and impractical. Therefore, the most common scene is to train the algorithm under a simulated environment first, and then apply it to the real robot. This brings the significance of building a decent simulated environment so that the performance of the algorithm under the simulated environment would be similar to the performance in the real world. In such cases, equipping the simulated environment with a simulated depth sensor can help close the sim-to-real gap.

There are available CPU or GPU implementations of some major parts of this module. However, putting all those implementations together is non-trivial, and the performance attained by such approaches is usually not satisfying. This module provides an encapsulated package that has not only accelerated major algorithms on GPU, but also minimized CPU-GPU transfer to achieve real-time performance.

The input of the module is simply two images captured by a pair of nearby parallel cameras, and it will directly output the computed depth map.

## Pipeline
The pipeline of the module is as follows:

![pipeline](doc/pipeline.png?)

### About Stereo Matching
The algorithm used in this part is **semi-global block matching**, which is a combination of semi-global matching and block matching algorithms. [Semi-global matching](https://en.wikipedia.org/wiki/Semi-global_matching) is a robust matching algorithm that can produce smooth disparity maps. The semi-global block matching algorithm implemented in this module differs from the original [H. Hirschmuller algorithm](https://core.ac.uk/download/pdf/11134866.pdf) as follows:
- The number of aggregation paths are four instead of eight. The four paths here are left-to-right, right-to-left, top-to-bottom and bottom-to-top.
- For two individual pixels in the left and right image, instead of directly matching them, the algorithm matches all of their neighboring pixels together, which is similar to block matching. The size of the neighborhood is defined by `block_width` and `block_height`. Setting both of them to 1 will reduce the blocks to single pixels.
- Mutual information cost is not implemented. Instead, the cost is the hamming distance between the CSCT features.

### About Uniqueness Test
Uniqueness test filters out matches that are not unique enough. For a pixel in the left image and its best match in the right image, uniqueness test will find the cost of second best match (excluding the left and right adjacent pixels of the best match), and compare it to the cost of the best one. If the best match's cost does not win the second best match's cost by a certain margin, the matching will be marked as invalid.

### About Disparity to Depth Conversion
Optional depth registration can be performed with little cost at this step, which will reproject the computed depth map from the left camera's frame to a specified RGB camera's frame. This allows easy integration with RGB data to get RGBD image. If RGB camera's frame is not specified, the output depth map will be in left camera's frame with the same size of the input images by default.

## Requirements
- CUDA ToolKit
- CMAKE 3.18 or later

CUDA Toolkit needs to be installed to compile .cu source files. Check https://developer.nvidia.com/cuda-downloads for instructions.

## Install
Run

    git clone git@github.com:angli66/simsense.git
    cd simsense
    git submodule update --init

to clone the repository and get the 3rd party library.

Before installing, check your NVIDIA GPU's compute capability at https://developer.nvidia.com/cuda-gpus. The default settings supports NVIDIA GPU with compute capability of 6.0, 6.1, 7.0, 7.5, 8.0 and 8.6. Check your GPU's compute capability and if it's not listed above, add the value into `CMakeLists.txt`. For example, if your GPU's compute capability is 8.9, change line 19 of `CmakeLists.txt` into

    set_target_properties(simsense PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86;89")

After that, run

    pip install .

to build the package with `pip`.

## Result
`test1.py` under the `test` directory shows an example usage of the module. The input are two 848x480 infrared images captured by [RealSense D435](https://www.intelrealsense.com/depth-camera-d435/) depth sensor:
### Left Image
<img src="test/img/left1.png?" width="424" height="240" />

### Right Image
<img src="test/img/right1.png?" width="424" height="240" />

As reference, the depth map generated by D435 is as follows (invalid points are marked as black):

<img src="doc/result_from_D435.png?" width="424" height="240" />

With certain parameters tuning to match the result by D435, the depth map computed by SimSense is as follows (invalid points are marked as dark blue):

<img src="doc/result_from_SimSense.png?" width="424" height="240" />

`test2.py` under the `test` directory shows another example of a harder case with transparent and reflective objects. It also shows the usage of depth registration.

## Performance
Experiment settings:
- GPU: RTX 2080Ti
- Input Image Size: 848x480
- Other parameters set to default

|                | 64 Disparity Level | 128 Disparity Level | 256 Disparity Level |
|----------------|--------------------|---------------------|---------------------|
| w/o Block Cost | 231.1 FPS          | 148.7 FPS           | 84.4 FPS            |
| w/ Block Cost  | 195.7 FPS          | 124.1 FPS           | 69.4 FPS            |

(w/o Block Cost means `block_width=1` and `block_height=1`)

## Known Issues
Doing numpy array slicing on column (e.g. `img = img[:, :-1]`) before feeding the image into the module will lead to incorrect output. This has something to do with the memory layout of the numpy array. To avoid this, use `numpy.delete()` instead of array slicing when necessary.

## Acknowledgement
Some acceleration techniques of the module are inspired by this great paper: [Embedded real-time stereo estimation via Semi-Global Matching on the GPU](http://www.sciencedirect.com/science/article/pii/S1877050916306561) by [D. Hernandez-Juarez](http://danihernandez.eu) et al..
