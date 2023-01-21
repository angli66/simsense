import numpy as np
import cv2
from .pysimsense import DepthSensorEngine

class DepthSensor:
    def __init__(
        self, lr_size, k_l, k_r, l2r, rgb_size=None, k_rgb=None, l2rgb=None, min_depth=0.0, max_depth=10.0,
        ir_noise_seed=0, speckle_shape=0.0, speckle_scale=0.0, gaussian_mu=0.0, gaussian_sigma=0.0, rectified=False,
        census_width=7, census_height=7, max_disp=128, block_width=7, block_height=7, p1_penalty=7, p2_penalty=86,
        uniqueness_ratio=15, lr_max_diff=1, median_filter_size=3, depth_dilation=False
    ):
        """
        Initiate the DepthSensor class. The camera frame follows the OpenCV coordinate system (x right, y down, z forward). Left,
        right and RGB camera are assumed to be undistorted. By default, the final depth map will be presented in left camera's frame
        with lr_size. Specifying rgb_size, k_rgb and l2rgb will turn on depth registration, which will tranform the final depth map
        from left camera's frame to specified RGB camera's frame with rgb_size. Specifying speckle_shape > 0 will turn on infrared
        noise simulation.

        :param lr_size: (width, height) of the left and right image.
        :param k_l: Intrinsic matrix of the left camera (in OpenCV coordinate system).
        :param k_r: Intrinsic matrix of the right camera (in OpenCV coordinate system).
        :param l2r: Change-of-coordinate matrix from left camera's frame to right camera's frame (in OpenCV coordinate system).
        :param rgb_size: (width, height) of the RGB image.
        :param k_rgb: Intrinsic matrix of the RGB camera (in OpenCV coordinate system).
        :param l2rgb: Change-of-coordinate matrix from left camera's frame to RGB camera's frame (in OpenCV coordinate system).
        :param min_depth: Minimum valid depth in meters.
        :param max_depth: Maximum valid depth (non-inclusive) in meters.
        :param ir_noise_seed: Random seed for simulating infrared noise.
        :param speckle_shape: Shape parameter for simulating infrared speckle noise (Gamma distribution). Set to 0 to disable
                              noise simulation.
        :param speckle_scale: Scale parameter for simulating infrared speckle noise (Gamma distribution).
        :param gaussian_mu: Mean for simulating infrared thermal noise (Gaussian distribution).
        :param gaussian_sigma: Standard deviation for simulating infrared thermal noise (Gaussian distribution).
        :param rectified: Whether the input has already been rectified. Set to true if no rectification is needed.
        :param census_width: Width of the center-symmetric census transform window. This must be an odd number.
        :param census_height: Height of the center-symmetric census transform window. This must be an odd number.
        :param max_disp: Maximum disparity search space (non-inclusive) for stereo matching.
        :param block_width: Width of the matched block. This must be an odd number.
        :param block_height: Height of the matched block. This must be an odd number.
        :param p1_penalty: P1 penalty for semi-global matching algorithm.
        :param p2_penalty: P2 penalty for semi-global matching algorithm.
        :param uniqueness_ratio: Margin in percentage by which the minimum computed cost should win the second best (not considering
                                 best match's adjacent pixels) cost to consider the found match valid.
        :param lr_max_diff: Maximum allowed difference in the left-right consistency check. Set it to 255 to disable the check.
        :param median_filter_size: Size of the median filter. Choices are 1, 3, 5, 7. Set to 1 to disable median filter.
        :param depth_dilation: Dilate the final depth map to avoid holes when depth registration is on. Recommended when rgb_size
                               is greater than lr_size.
        """
        img_w, img_h = lr_size
        registration = False
        if rgb_size != None or k_rgb != None or l2rgb != None:
            registration = True

        # Instance check
        if not isinstance(img_h, int) or not isinstance(img_w, int) or img_h < 32 or img_w < 32:
            raise TypeError("Image height and width must be integer no less than 32")
        
        if registration == True and (rgb_size is None or k_rgb is None or l2rgb is None):
            raise TypeError("Depth registration is on but missing one or two RGB camera's parameters")

        if speckle_shape > 0 and (speckle_scale <= 0 or gaussian_sigma <= 0):
            raise TypeError("Infrared noise simulation is on. Speckle_scale and gaussian_sigma must both be positive")
        
        if not isinstance(census_width, int) or not isinstance(census_height, int) or census_width <= 0 or census_height <= 0 or \
                census_width % 2 == 0 or census_height % 2 == 0 or census_width * census_height > 65:
            raise TypeError("census_width and census_height must be positive odd integers and their product should be no larger than 65")
        
        if not isinstance(max_disp, int) or max_disp < 32 or max_disp > 1024:
            raise TypeError("max_disp must be integer within range [32, 1024]")
        
        if not isinstance(block_width, int) or not isinstance(block_height, int) or block_width <= 0 or block_height <= 0 or \
                block_width % 2 == 0 or block_height % 2 == 0 or block_width * block_height > 256:
            raise TypeError("block_width and block_height must be positive odd integers and their product should be no larger than 256")
 
        if not isinstance(p1_penalty, int) or not isinstance(p2_penalty, int) or p1_penalty <= 0 or p2_penalty <= 0 or \
                p1_penalty >= p2_penalty or p2_penalty >= 224:
            raise TypeError("p1 must be positive integer less than p2 and p2 be positive integer less than 224")

        if not isinstance(uniqueness_ratio, int) or uniqueness_ratio < 0 or uniqueness_ratio > 255:
            raise TypeError("uniqueness_ratio must be positive integer no larger than 255")

        if not isinstance(lr_max_diff, int) or lr_max_diff < -1 or lr_max_diff > 255:
            raise TypeError("lr_max_diff must be integer within the range [0, 255]")

        if median_filter_size != 1 and median_filter_size != 3 and median_filter_size != 5 and median_filter_size != 7:
            raise TypeError("Median filter size choices are 1, 3, 5, 7")

        # Get rectification map
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            cameraMatrix1=k_l, distCoeffs1=None, cameraMatrix2=k_r, distCoeffs2=None,
            imageSize=lr_size, R=l2r[:3, :3], T=l2r[:3, 3:], alpha=1.0, newImageSize=lr_size
        )
        f_len = q[2][3] # focal length of the left camera in meters
        b_len = 1.0 / q[3][2] # baseline length in meters
        map_l = cv2.initUndistortRectifyMap(k_l, None, r1, p1, lr_size, cv2.CV_32F)
        map_r = cv2.initUndistortRectifyMap(k_r, None, r2, p2, lr_size, cv2.CV_32F)
        map_lx, map_ly = map_l
        map_rx, map_ry = map_r

        if registration:
            # Get registration matrix
            a1, a2, a3, b = self._get_registration_mat(lr_size, k_l, k_rgb, l2rgb)
            self.engine = DepthSensorEngine(img_h, img_w, rgb_size[1], rgb_size[0], f_len, b_len, min_depth, max_depth,
                                            ir_noise_seed, speckle_shape, speckle_scale, gaussian_mu, gaussian_sigma,
                                            rectified, census_width, census_height, max_disp, block_width, block_height,
                                            p1_penalty, p2_penalty, uniqueness_ratio, lr_max_diff, median_filter_size,
                                            map_lx, map_ly, map_rx, map_ry, a1, a2, a3, b[0], b[1], b[2], depth_dilation)
        else:
            self.engine = DepthSensorEngine(img_h, img_w, f_len, b_len, min_depth, max_depth, ir_noise_seed, speckle_shape,
                                            speckle_scale, gaussian_mu, gaussian_sigma, rectified, census_width, census_height,
                                            max_disp, block_width, block_height, p1_penalty, p2_penalty, uniqueness_ratio,
                                            lr_max_diff, median_filter_size, map_lx, map_ly, map_rx, map_ry)

    def compute(self, img_l, img_r):
        """
        Take two images captured by a pair of nearby parallel cameras, and output the computed depth map in meters.

        :param img_l: Grayscale/infrared image (uint8) captured by left camera.
        :param img_r: Grayscale/infrared image (uint8) captured by right camera.
        :return: Computed depth map (in meters) from left camera's view or rgb camera's view.
        """
        result = self.engine.compute(img_l, img_r)

        return result

    def set_ir_noise_parameters(self, speckle_shape, speckle_scale, gaussian_mu, gaussian_sigma):
        """
        :param speckle_shape: Shape parameter for simulating infrared speckle noise (Gamma distribution). Set to 0 to disable
                              noise simulation.
        :param speckle_scale: Scale parameter for simulating infrared speckle noise (Gamma distribution).
        :param gaussian_mu: Mean for simulating infrared thermal noise (Gaussian distribution).
        :param gaussian_sigma: Standard deviation for simulating infrared thermal noise (Gaussian distribution).
        """
        if speckle_shape > 0 and (speckle_scale <= 0 or gaussian_sigma <= 0):
            raise TypeError("Infrared noise simulation is on. Speckle_scale and gaussian_sigma must both be positive")
        self.engine.set_ir_noise_parameters(speckle_shape, speckle_scale, gaussian_mu, gaussian_sigma)
    
    def set_census_window_size(self, census_width, census_height):
        """
        :param census_width: Width of the center-symmetric census transform window. This must be an odd number.
        :param census_height: Height of the center-symmetric census transform window. This must be an odd number.
        """
        if not isinstance(census_width, int) or not isinstance(census_height, int) or census_width <= 0 or census_height <= 0 or \
                census_width % 2 == 0 or census_height % 2 == 0 or census_width*census_height > 65:
            raise TypeError("census_width and census_height must be positive odd integers and their product should be no larger than 65")
        self.engine.set_census_window_size(census_width, census_height)
    
    def set_matching_block_size(self, block_width, block_height):
        """
        :param block_width: Width of the matched block. This must be an odd number.
        :param block_height: Height of the matched block. This must be an odd number.
        """
        if not isinstance(block_width, int) or not isinstance(block_height, int) or block_width <= 0 or block_height <= 0 or \
                block_width % 2 == 0 or block_height % 2 == 0 or block_width*block_height > 256:
            raise TypeError("block_width and block_height must be positive odd integers and their product should be no larger than 256")
        self.engine.set_matching_block_size(block_width, block_height)
    
    def set_penalties(self, p1_penalty, p2_penalty):
        """
        :param p1_penalty: P1 penalty for semi-global matching algorithm.
        :param p2_penalty: P2 penalty for semi-global matching algorithm.
        """
        if not isinstance(p1_penalty, int) or not isinstance(p2_penalty, int) or p1_penalty <= 0 or p2_penalty <= 0 or \
                p1_penalty >= p2_penalty or p2_penalty >= 224:
            raise TypeError("p1 must be positive integer less than p2 and p2 be positive integer less than 224")
        self.engine.set_penalties(p1_penalty, p2_penalty)
    
    def set_uniqueness_ratio(self, uniqueness_ratio):
        """
        :param uniqueness_ratio: Margin in percentage by which the minimum computed cost should win the second best (not considering
                                 best match's adjacent pixels) cost to consider the found match valid.
        """
        if not isinstance(uniqueness_ratio, int) or uniqueness_ratio < 0 or uniqueness_ratio > 255:
            raise TypeError("uniqueness_ratio must be positive integer no larger than 255")
        self.engine.set_uniqueness_ratio(uniqueness_ratio)

    def set_lr_max_diff(self, lr_max_diff):
        """
        :param lr_max_diff: Maximum allowed difference in the left-right consistency check. Set it to 255 to disable the check.
        """
        if not isinstance(lr_max_diff, int) or lr_max_diff < -1 or lr_max_diff > 255:
            raise TypeError("lr_max_diff must be integer within the range [0, 255]")
        self.engine.set_lr_max_diff(lr_max_diff)

    def _get_registration_mat(self, ir_size, k_ir, k_rgb, ir2rgb):
        R = ir2rgb[:3, :3]
        t = ir2rgb[:3, 3:]
        
        w, h = ir_size
        x = np.arange(w)
        y = np.arange(h)
        u, v = np.meshgrid(x, y)
        w = np.ones_like(u)
        pixel_coords = np.stack([u, v, w], axis=-1) # pixel_coords[y, x] is (x, y, 1)

        A = np.einsum("ij,hwj->hwi", k_rgb @ R @ np.linalg.inv(k_ir), pixel_coords)
        B = k_rgb @ t

        return A[..., 0], A[..., 1], A[..., 2], B
