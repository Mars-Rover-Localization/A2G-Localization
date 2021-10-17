from osgeo import gdal
import open3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
from contextlib import contextmanager
import time


@contextmanager
def Timer(msg):
    print(msg)
    start = time.perf_counter()
    try:
        yield
    finally:
        print("%.4f ms" % ((time.perf_counter() - start) * 1000))


def image_undistort(path: str, fx, fy, cx, cy, k1, k2, p1, p2, save_path=None):
    """
    Wrapper for cv2.undistort()

    All parameters here are provided by camera manufacturer, only change if the camera is recalibrated.
    """
    original_image = cv2.imread(path)
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distortion_coef = np.array([k1, k2, p1, p2])

    undistorted = cv2.undistort(original_image, camera_matrix, distortion_coef)

    if save_path:
        cv2.imwrite(save_path, undistorted)
    else:
        return undistorted


def get_geotransform(path: str):
    img = gdal.Open(path, 0)
    return img.GetGeoTransform()


def get_XYZ_value(xy: np.ndarray, geotransform, pointcloud):
    """
    Get object space coordinate from image point index.

    This method will perform the following in order:

    1) Transform the point index to the object space coordinate system;

    2) Perform a least distance search through the point cloud to find the corresponding object space coordinate

    For efficient querying of multiple points, this implementation first builds KDTree structure, in which perform query.

    :param xy: np.ndarray containing point index x (row) and y (col) in format of np.array([[x_i, y_i]])
    :param geotransform: Coordinate transformation parameters from get_geotransform()
    :param pointcloud: Point cloud data from SFM, LiDAR, etc. Notice that it should be preprocessed to numpy.ndarray type
    :return: numpy.ndarray, representing (X, Y, Z) value of ndarray
    """
    assert xy.shape[1] == 2, 'Invalid input points'
    assert len(geotransform) == 6, 'Invalid transformation parameters'

    # Please notice that there exists difference between image coordinate description in our implementation and GDAL docs
    trans_matrix = np.array([[geotransform[2], geotransform[5]], [geotransform[1], geotransform[4]]])

    XY = xy @ trans_matrix + np.array([geotransform[0], geotransform[3]])

    pointcloud_XY = pointcloud[:, 0: 2]

    with Timer("Optimized KDTree"):
        neighbour = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        neighbour.fit(pointcloud_XY)

        with Timer("KDTree query"):
            nearest_point_index = neighbour.kneighbors(XY, return_distance=False)

    return nearest_point_index, pointcloud[nearest_point_index]


# Demo
# Sample images may be provided in future
geotransform = get_geotransform(r"C:\Users\Lincoln\Project\Mars Test 210523 No GPS\5_Products\Mars Test 210523 No GPS_OrthoMosaic_Fast.tif")
cloud = np.asarray(open3d.io.read_point_cloud(r"C:\Users\Lincoln\Project\Mars Test 210523 No GPS\3_Clouds\Tile-0.ply").points)

test_xy = np.array([[2250, 1914]])
print(get_XYZ_value(test_xy, geotransform, cloud))
