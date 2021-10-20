"""
Calculate camera location / exterior elements from image space and object space coordinates.

The primary method used currently is based on the following paper:
龚辉, et al. "一种基于四元数的空间后方交会全局收敛算法." 测绘学报 40.5 (2011): 0.

More methods will be added in the near future.

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn
"""

import numpy as np
from numpy.linalg import inv, svd, eig


def generate_M(q0, q1, q2, q3):
    return np.array([[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                     [2 * (q2 * q1 + q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 - q0 * q1)],
                     [2 * (q3 * q1 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]])


def quaternion(image_coordinate, object_space_coordinate, focal_length, x0, y0, accept_error=1e-5):
    data_size = image_coordinate.shape[0]
    assert data_size == object_space_coordinate.shape[0], 'Invalid input data'

    q0, q1, q2, q3 = 0, 0, 0, 0

    M = generate_M(q0, q1, q2, q3)

    iteration = 0

    while True:
        # Calculate rsm
        rsm_left = np.zeros((3, 3))
        rsm_right = np.zeros((3, 1))

        for index in range(data_size):
            x, y = image_coordinate[index]  # Image pixel coordinate
            X, Y, Z = object_space_coordinate[index]  # Object space coordinate
            r = np.array([[X], [Y], [Z]])

            v = np.array([[x - x0], [y - y0], [-focal_length]])
            V = (v @ v.transpose()) / (v.transpose() @ v)  # Projection matrix

            I = np.identity(3)
            H = (I - V).transpose() @ (I - V)

            rsm_left = rsm_left + M.transpose() @ H @ M
            rsm_right = rsm_right + M.transpose() @ H @ M @ r

        # rsm = inv(rsm_left) @ rsm_right
        # The following calculation use total least squares method to avoid potential error caused by near-singular matrix
        # For more info, refer to: https://people.duke.edu/~hpgavin/SystemID/CourseNotes/TotalLeastSquares.pdf
        LR = np.hstack((rsm_left, rsm_right))
        U, S, Vh = svd(LR)
        Vh = Vh.transpose()

        rsm = (-Vh[0:3, 3] / Vh[3, 3]).reshape((3, 1))

        r_tilde = np.zeros((3, 1))
        re_tilde = np.zeros((3, 1))

        for index in range(data_size):
            x, y = image_coordinate[index]
            X, Y, Z = object_space_coordinate[index]
            r = np.array([[X], [Y], [Z]])

            v = np.array([[x - x0], [y - y0], [-focal_length]])
            V = (v @ v.transpose()) / (v.transpose() @ v)

            r_tilde = r_tilde + r

            rm_derivative = M @ (r - rsm)

            re_tilde = re_tilde + V @ rm_derivative

        r_tilde = r_tilde / data_size
        re_tilde = re_tilde / data_size

        delta_r = np.zeros((data_size, 3))
        delta_re = np.zeros((data_size, 3))

        for index in range(data_size):
            x, y = image_coordinate[index]
            X, Y, Z = object_space_coordinate[index]
            r = np.array([[X], [Y], [Z]])

            v = np.array([[x - x0], [y - y0], [-focal_length]])
            V = (v @ v.transpose()) / (v.transpose() @ v)

            rm_derivative = M @ (r - rsm)
            r_Ei = V @ rm_derivative

            current_delta_r = r - r_tilde
            current_delta_re = r_Ei - re_tilde

            delta_r[index] = current_delta_r.flatten()
            delta_re[index] = current_delta_re.flatten()

        Nxx, Nxy, Nxz, Nyx, Nyy, Nyz, Nzx, Nzy, Nzz = np.zeros(9)

        # Need inspection on potential vectorization
        for index in range(data_size):
            Nxx += delta_r[index, 0] * delta_re[index, 0]
            Nxy += delta_r[index, 0] * delta_re[index, 1]
            Nxz += delta_r[index, 0] * delta_re[index, 2]

            Nyx += delta_r[index, 1] * delta_re[index, 0]
            Nyy += delta_r[index, 1] * delta_re[index, 1]
            Nyz += delta_r[index, 1] * delta_re[index, 2]

            Nzx += delta_r[index, 2] * delta_re[index, 0]
            Nzy += delta_r[index, 2] * delta_re[index, 1]
            Nzz += delta_r[index, 2] * delta_re[index, 2]

        # Construct matrix N
        N = np.array(
            [[Nxx + Nyy + Nzz, Nyz - Nzy, Nzx - Nxz, Nxy - Nyx], [Nyz - Nzy, Nxx - Nyy - Nzz, Nxy + Nyx, Nzx + Nxz],
             [Nzx - Nxz, Nxy + Nyx, -Nxx + Nyy - Nzz, Nyz + Nzy], [Nxy - Nyx, Nzx + Nxz, Nyz + Nzy, -Nxx - Nyy + Nzz]])

        eigenvalues, eigenvectors = eig(N)

        max_eigenvalue_index = np.argmax(eigenvalues)
        corresponding_vector = eigenvectors[:, max_eigenvalue_index]

        q0, q1, q2, q3 = corresponding_vector

        previous_M = np.copy(M)

        M = generate_M(q0, q1, q2, q3)

        # End iteration if error < accept_error
        if np.count_nonzero(np.abs(M - previous_M) > accept_error) == 0:
            break

        iteration += 1

    print(f"Converge after {iteration} iterations")

    return rsm


# Test
x0 = 0
y0 = 0
f = 306.1

xy = np.array([[7.312142, -64.009330], [74.393220, -63.426510], [8.368604, 5.115163], [75.490780, -4.375459], [21.336530, 81.856790], [74.842120, 77.767140]])
XYZ = np.array([[465620.380728, 846401.191184, 106.898610], [465798.101865, 846420.730346, 119.282546], [465606.704168, 846587.523539, 106.758524], [465788.971326, 846578.042224, 109.561501], [465623.188035, 846795.766471, 106.136904], [465767.758773, 846798.049289, 105.742106]])
print(quaternion(xy, XYZ, f, x0, y0))
