from osgeo import gdal
import open3d
import numpy as np


def get_geotransform(path: str):
    img = gdal.Open(path, 0)
    return img.GetGeoTransform()


def get_XYZ_value(x: int, y: int, geotransform, pointcloud):
    """
    Get object space coordinate from image point index.

    This method will perform the following in order:

    1) Transform the point index to the object space coordinate system;

    2) Perform a least distance search through the point cloud to find the corresponding object space coordinate

    :param x: point index x (row)
    :param y: point index y (col)
    :param geotransform: Coordinate transformation parameters from get_geotransform()
    :param pointcloud: Point cloud data from SFM, LiDAR, etc. Notice that it should be preprocessed to numpy.ndarray type
    :return: numpy.ndarray, representing (X, Y, Z) value of ndarray
    """
    assert len(geotransform) == 6, 'Invalid transformation parameters'

    # Please notice that there exists difference between image coordinate description in our implementation and GDAL docs
    X = geotransform[0] + geotransform[1] * y + geotransform[2] * x
    Y = geotransform[3] + geotransform[4] * y + geotransform[5] * x

    distance = np.fromfunction(lambda i, j: (pointcloud[i, 0] - X) ** 2 + (pointcloud[i, 1] - Y) ** 2, (pointcloud.shape[0], 1), dtype=int)
    nearest_point_index = np.argmin(distance)

    return pointcloud[nearest_point_index]


# Demo
# Sample images may be provided in future
geotransform = get_geotransform(r"C:\Users\Lincoln\Project\Mars Test 210523 No GPS\5_Products\Mars Test 210523 No GPS_OrthoMosaic_Fast.tif")
cloud = np.asarray(open3d.io.read_point_cloud(r"C:\Users\Lincoln\Project\Mars Test 210523 No GPS\3_Clouds\Tile-0.ply").points)

print(get_XYZ_value(2250, 1914, geotransform, cloud))
