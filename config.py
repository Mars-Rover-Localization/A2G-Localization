"""
This module contains configuration data for aerial to ground matching pipeline.
For usage of each variable, please refer to in-line notations.
"""

# File locations
point_cloud_path = r"C:\Users\Lincoln\Project\1023_Flight_Noon\3_Clouds\Tile-0.ply"
rover_images_path = r"C:\Users\Lincoln\Desktop\1023\manual_sample"
ortho_image_path = r"C:\Users\Lincoln\Project\1023_Flight_Noon\5_Products\1023_Flight_Noon_OrthoMosaic_Fast.tif"

# Rover camera parameters
fx = 1059.58
fy = 1058.8199
cx = 994.56
cy = 549.375
k1 = -0.0424
k2 = 0.0113
p1 = 0.0001
p2 = 0.0003
k3 = -0.0053

x0 = 0
y0 = 0

# Resection iteration parameter
accept_error = 1e-5

# ASIFT matching parameter
"""
Currently PyASIFT cannot process large size image correctly.
We believe it's a bug in OpenCV's knnmatch algorithm.
While we are actively developing alternative matching algorithms, current input image size is deliberately limited.
After resizing, the program ensures that the width of image will not exceed MAX_SIZE.
Please notice that the keypoints returned and logged will be rescaled to original size.
From our testing, it's recommended that MAX_SIZE be set to 1500-2000.
If ASIFT module throws an error while executing, reduce the MAX_SIZE value may help. 
"""
MAX_SIZE = 1000

FLANN_INDEX_KDTREE = 1

FLANN_INDEX_LSH = 6

# Parameter for manual override mode
MAX_DISPLAY_SIZE = 1400
