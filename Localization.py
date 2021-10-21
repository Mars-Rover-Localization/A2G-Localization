import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d
import os.path
import utilities as utl
from Resection import quaternion
from asift import asift_main
from config import fx, fy, cx, cy, x0, y0, accept_error
from config import point_cloud_path, rover_images_path, ortho_image_path


geo_transform = utl.get_geotransform(ortho_image_path)
cloud = np.asarray(open3d.io.read_point_cloud(point_cloud_path).points)
images = [os.path.join(rover_images_path, name) for name in os.listdir(rover_images_path)]

image_count = len(os.listdir(rover_images_path))

# The following undistorting process only needs to be executed once
"""
for distort_image in images:
    utl.image_undistort(distort_image, fx, fy, cx, cy, k1, k2, p1, p2, distort_image)
"""


def localization():
    positions = np.zeros((image_count, 2))

    with utl.Timer("Building KDTree ..."):
        kd_tree = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        kd_tree.fit(cloud[:, 0: 2])

    for index in range(1, image_count + 1):
        print(f"Station {index}")

        # Affine-SIFT matching
        kp_pairs = asift_main(rf"C:\Users\Lincoln\Project\0529_left_cam\{index}.png",
                              r"C:\Users\Lincoln\Project\Mars Field 0529\5_Products\Mars Field 0529_OrthoMosaic_Fast.tif")
        query, base = utl.convert_keypoint(kp_pairs)

        data_size = len(query)

        # Transform pixel coordinates to image space coordinates
        image_space_coordinates = np.hstack((query[:, 0].reshape((data_size, 1)) - cx, cy - query[:, 1].reshape((data_size, 1))))

        # Query object space coordinate from point cloud
        object_space_coordinates = utl.get_XYZ_value(np.hstack((base[:, 1].reshape((data_size, 1)), base[:, 0].reshape((data_size, 1)))), geo_transform, cloud, kd_tree)

        # Perform space resection
        # Please notice that the indexing convention for resection method is (col, row) or (x, y)
        current_localization_result = quaternion(image_space_coordinates, object_space_coordinates, (fx + fy) * 0.5, x0, y0, accept_error).flatten()
        positions[index - 1] = current_localization_result[0: 2]

    if __name__ == '__main__':
        print(positions)

    return positions


# Invoke localization method and acquire object space positions
positions = localization()

# Transform object space coordinates back to image pixel coordinate of UAV-generated map
base_image_coordinates = utl.inverse_geotransform(positions, geo_transform)

# Visualize localization result
base_image = cv2.imread(ortho_image_path)
utl.visualize_localization(base_image, base_image_coordinates)
