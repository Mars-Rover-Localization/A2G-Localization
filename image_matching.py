"""
Feature Extraction & Matching Implementation.

Based on OpenCV Python samples at https://github.com/opencv/opencv/blob/master/samples/python/find_obj.py.

Interactive features were removed for better performance.

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/PyASIFT
"""

import numpy as np
from cv2 import cv2
from adalam import AdalamFilter
import torch


from config import FLANN_INDEX_KDTREE, FLANN_INDEX_LSH


def init_feature(name):
    chunks = name.split('-')

    if chunks[0] == 'sift':
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None   # Return None if unknown detector name

    # The selection of the following parameters are partially explained in https://docs.opencv.org/4.5.4/dc/dc3/tutorial_py_matcher.html
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2

        matcher = cv2.FlannBasedMatcher(flann_params)
    else:
        matcher = cv2.BFMatcher(norm)

    return detector, matcher


def adalam_parser(kp1, kp2, desc1: np.ndarray, desc2: np.ndarray, img1_shape, img2_shape):
    pts1 = np.array([k.pt for k in kp1], dtype=np.float32)
    ors1 = np.array([k.angle for k in kp1], dtype=np.float32)
    scs1 = np.array([k.size for k in kp1], dtype=np.float32)

    pts2 = np.array([k.pt for k in kp2], dtype=np.float32)
    ors2 = np.array([k.angle for k in kp2], dtype=np.float32)
    scs2 = np.array([k.size for k in kp2], dtype=np.float32)
    
    custom_config = {
        'area_ratio': 100,  # Ratio between seed circle area and image area. Higher values produce more seeds with smaller neighborhoods.
        'search_expansion': 4,  # Expansion factor of the seed circle radius for the purpose of collecting neighborhoods. Increases neighborhood radius without changing seed distribution
        'ransac_iters': 128,  # Fixed number of inner GPU-RANSAC iterations
        'min_inliers': 6,  # Minimum number of inliers required to accept inliers coming from a neighborhood
        'min_confidence': 200,  # Threshold used by the confidence-based GPU-RANSAC
        'orientation_difference_threshold': 30,  # Maximum difference in orientations for a point to be accepted in a neighborhood. Set to None to disable the use of keypoint orientations.
        'scale_rate_threshold': 1.5,  # Maximum difference (ratio) in scales for a point to be accepted in a neighborhood. Set to None to disable the use of keypoint scales.
        'detected_scale_rate_threshold': 5,  # Prior on maximum possible scale change detectable in image couples. Affinities with higher scale changes are regarded as outliers.
        'refit': True,  # Whether to perform refitting at the end of the RANSACs. Generally improves accuracy at the cost of runtime.
        'force_seed_mnn': True,  # Whether to consider only MNN for the purpose of selecting seeds. Generally improves accuracy at the cost of runtime. You can provide a MNN mask in input to skip MNN computation and still get the improvement.
        'device': torch.device('cpu')  # Override to use CPU only.
    }
    
    matcher = AdalamFilter()
    matches = matcher.match_and_filter(k1=pts1, k2=pts2,
                                       o1=ors1, o2=ors2,
                                       d1=desc1, d2=desc2,
                                       s1=scs1, s2=scs2,
                                       im1shape=img1_shape, im2shape=img2_shape).cpu().numpy()

    kp1 = kp1[matches[:, 0]]
    kp2 = kp2[matches[:, 1]]
    kp_pairs = zip(kp1, kp2)

    return kp_pairs


def resize_kp_pairs(kp_pairs, r1, r2):
    for index, (kp1, kp2) in enumerate(kp_pairs):
        new_kp1 = cv2.KeyPoint(kp1.pt[0] / r1, kp1.pt[1] / r1, kp1.size)
        new_kp2 = cv2.KeyPoint(kp2.pt[0] / r2, kp2.pt[1] / r2, kp2.size)

        kp_pairs[index] = (new_kp1, new_kp2)

    return kp_pairs


def filter_matches(matches, ratio=0.75):
    filtered_matches = []

    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])

    return filtered_matches


def unpack_matches(kp1, kp2, matches):
    mkp1, mkp2 = [], []

    for m in matches:
        mkp1.append(kp1[m.queryIdx])
        mkp2.append(kp2[m.trainIdx])

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)

    return p1, p2, list(kp_pairs)


def draw_match(result_title, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create visualized result image
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking

    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            color = green
            cv2.circle(vis, (x1, y1), 2, color, -1)
            cv2.circle(vis, (x2, y2), 2, color, -1)
        else:
            color = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), color, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), color, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), color, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), color, thickness)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    if __name__ == '__main__':
        cv2.imshow(result_title, vis)

    cv2.imwrite("sample/match_result.png", vis)

    return vis
