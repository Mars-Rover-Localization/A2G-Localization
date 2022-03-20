from osgeo import gdal
import open3d
import numpy as np
from cv2 import cv2
import sys
from contextlib import contextmanager
import time
from enum import Enum
from config import MAX_DISPLAY_SIZE


@contextmanager
def Timer(msg):
    print(msg)
    start = time.perf_counter()
    try:
        yield
    finally:
        print("%.4f ms" % ((time.perf_counter() - start) * 1000))


def convert_keypoint(kp_pairs):
    data_size = len(kp_pairs)

    query_points = np.zeros((data_size, 2))
    base_points = np.zeros((data_size, 2))

    for index in range(data_size):
        query_points[index] = np.int32(kp_pairs[index][0].pt)
        base_points[index] = np.int32(kp_pairs[index][1].pt)

    return query_points, base_points


def log_keypoints(kp_pairs, path: str = 'sample/keypoints.txt'):
    with open(path, 'w') as log:
        for kp1, kp2 in kp_pairs:
            log.write(f"{np.int32(kp1.pt)}      {np.int32(kp2.pt)}\n")
    log.close()

    print(f"Keypoints logged at {path}")


def image_resize(src, ratio: float):
    dim = (int(src.shape[-1] * ratio), int(src.shape[0] * ratio))
    return cv2.resize(src, dim, interpolation=cv2.INTER_AREA)


def image_split(src):
    w = src.shape[1]
    half = int(w / 2)
    left_img = src[:, half:]
    right_img = src[:, :half]

    return left_img, right_img


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


def concatenate_point_cloud(paths) -> np.ndarray:
    result = np.concatenate([np.asarray(open3d.io.read_point_cloud(path).points) for path in paths], axis=0)
    return result


def get_geotransform(path: str):
    img = gdal.Open(path, 0)
    return img.GetGeoTransform()


def get_XYZ_value(xy: np.ndarray, geotransform, pointcloud, kd_tree):
    """
    Get object space coordinate from image point index.

    This method will perform the following in order:

    1) Transform the point index to the object space coordinate system;

    2) Perform a least distance search through the point cloud to find the corresponding object space coordinate

    For efficient querying of multiple points, this implementation first builds KDTree structure, in which perform query.

    :param xy: np.ndarray containing point index x (row) and y (col) in format of np.array([[x_i, y_i]])
    :param geotransform: Coordinate transformation parameters from get_geotransform()
    :param pointcloud: Point cloud data from SFM, LiDAR, etc. Notice that it should be preprocessed to numpy.ndarray type
    :param kd_tree: KDTree structure built from point cloud
    :return: numpy.ndarray, representing (X, Y, Z) value of ndarray
    """
    assert xy.shape[1] == 2, 'Invalid input points'
    assert len(geotransform) == 6, 'Invalid transformation parameters'

    # Please notice that there exists difference between image coordinate description in our implementation and GDAL docs
    trans_matrix = np.array([[geotransform[2], geotransform[5]], [geotransform[1], geotransform[4]]])

    XY = xy @ trans_matrix + np.array([geotransform[0], geotransform[3]])

    nearest_point_index = kd_tree.kneighbors(XY, return_distance=False)

    return pointcloud[nearest_point_index].reshape((-1, 3))


def inverse_geotransform(object_space_coordinate, geotransform):
    assert object_space_coordinate.shape[1] == 2, 'Invalid coordinate'
    assert len(geotransform) == 6, 'Invalid transformation parameters'

    trans_matrix = np.array([[geotransform[2], geotransform[5]], [geotransform[1], geotransform[4]]])

    image_space_coordinate = (object_space_coordinate - np.array([geotransform[0], geotransform[3]])) @ np.linalg.inv(trans_matrix)

    return image_space_coordinate


def visualize_localization(image, coordinates):
    data_size = coordinates.shape[0]

    for index in range(data_size):
        image = cv2.circle(image, (int(coordinates[index, 1]), int(coordinates[index, 0])), radius=10, color=(0, 0, 255), thickness=-1)

    cv2.imwrite("sample/Localization_Result.png", image)


# GMS test
class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output


def manual_pick_keypoints(img1, img2, index):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    print(img1, img2)

    if image1 is None or image2 is None:
        print("Image reading failed")
        sys.exit(1)

    query, base = [], []
    ratio_1, ratio_2 = 1, 1

    if image1.shape[0] > MAX_DISPLAY_SIZE or image1.shape[1] > MAX_DISPLAY_SIZE:
        ratio_1 = MAX_DISPLAY_SIZE / image1.shape[1]
        image1 = cv2.resize(image1, (int(ratio_1 * image1.shape[1]), int(ratio_1 * image1.shape[0])), cv2.INTER_AREA)

    if image2.shape[0] > MAX_DISPLAY_SIZE or image2.shape[1] > MAX_DISPLAY_SIZE:
        ratio_2 = MAX_DISPLAY_SIZE / image2.shape[1]
        image2 = cv2.resize(image2, (int(ratio_2 * image2.shape[1]), int(ratio_2 * image2.shape[0])), cv2.INTER_AREA)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, image1.shape[-1]), np.uint8)
    vis[:h1, :w1] = image1
    vis[:h2, w1:w1 + w2] = image2

    is_current_query = True
    count = 0

    def on_mouse(event, x, y, flags, param):
        nonlocal is_current_query
        nonlocal vis
        nonlocal count
        nonlocal w1
        nonlocal index

        if event == cv2.EVENT_LBUTTONDOWN:
            if is_current_query:
                query.append((x, y))

                count += 1

                print(f"Picked query image point: ({x}, {y})")

                cv2.drawMarker(vis, (int(x), int(y)), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20,
                               thickness=2, line_type=cv2.LINE_AA)
                cv2.putText(vis, str(count), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                is_current_query = False

                cv2.imshow(f"Pick Point Pairs (image pair {index})", vis)
            else:
                base.append((x - w1, y))

                print(f"Picked base image point: ({x - w1}, {y})")

                cv2.drawMarker(vis, (int(x), int(y)), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20,
                               thickness=2, line_type=cv2.LINE_AA)
                cv2.putText(vis, str(count), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                cv2.line(vis, (int(query[-1][0]), int(query[-1][1])), (int(base[-1][0]) + w1, int(base[-1][1])), (0, 255, 0), thickness=1)

                is_current_query = True

                cv2.imshow(f"Pick Point Pairs (image pair {index})", vis)

    cv2.namedWindow(f"Pick Point Pairs (image pair {index})")
    cv2.setMouseCallback(f"Pick Point Pairs (image pair {index})", on_mouse)

    while True:
        cv2.imshow(f"Pick Point Pairs (image pair {index})", vis)

        k = cv2.waitKey(0)

        if k == 27 and is_current_query and count >= 4:
            break

    print(f"Selected pairs: {count}")

    cv2.destroyAllWindows()

    cv2.waitKey(0)

    return (np.array(query) / ratio_1).astype(int), (np.array(base) / ratio_2).astype(int)


def visualize_trajectory(image: np.ndarray, locations: np.ndarray):
    assert locations.shape[1] == 2, 'Invalid input data shape'

    locations[:, [1, 0]] = locations[:, [0, 1]]

    locations = locations.astype(np.int32)
    image = cv2.polylines(image, [locations], isClosed=False, color=(0, 255, 0), thickness=7, lineType=cv2.LINE_AA)

    for loc in locations:
        image = cv2.drawMarker(image, position=loc, color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=25, thickness=4)

    return image


if __name__ == '__main__':
    image = visualize_trajectory(cv2.imread(r"C:\Users\Lincoln\Project\Mars Field 0529_highacc\5_Products\Mars Field 0529_OrthoMosaic_Fast.tif"), np.load('result.npy'))

    cv2.imwrite('vis.png', image)
