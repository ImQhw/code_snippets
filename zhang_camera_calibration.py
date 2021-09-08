#! /usr/bin/env python3

"""Zhang's camera calibration algorithm

references:
    1. Z. Zhang, A flexible new technique for camera calibration
"""

import cv2
import levmar
import numpy as np
from scipy.spatial.transform import Rotation


class SingleCameraSimulator(object):
    def __init__(
        self, img_size_wh, fx, fy, cx, cy, skew=0.0, min_z_mm=100, max_z_mm=200
    ):
        self._w, self._h = img_size_wh

        self._K_3x3 = np.asarray(
            [
                [fx, skew, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        self._min_z_mm = min_z_mm
        self._max_z_mm = max_z_mm

    @property
    def K_3x3(self):
        return self._K_3x3

    def random_img_pts(self, obj_pts_nx3: np.ndarray, noise_sigma=None):
        t_3x1 = np.random.uniform(-20, 20, (3, 1))
        t_3x1[2, 0] = np.random.uniform(200, 250)
        R_3x3 = Rotation.from_euler(
            "zyx",
            [
                np.random.uniform(-np.pi, np.pi),  # z
                np.random.uniform(-np.pi / 4, np.pi / 4),  # y
                np.random.uniform(-np.pi / 4, np.pi / 4),  # x
            ],
        ).as_matrix()

        P_3x4 = self._K_3x3 @ np.hstack([R_3x3, t_3x1])

        homo_obj_pts_4xn = np.vstack([obj_pts_nx3.T, np.ones([1, len(obj_pts_nx3)])])

        homo_img_pts_3xn = P_3x4 @ homo_obj_pts_4xn

        img_pts_2xn = homo_img_pts_3xn[:2] / homo_img_pts_3xn[2]

        if noise_sigma is not None:
            noise = np.random.normal(0, noise_sigma, img_pts_2xn.shape)
            img_pts_2xn += noise

        return img_pts_2xn.T, P_3x4

    def show_img_pts(self, img_pts_nx2: np.ndarray, color=None):
        img_pts_nx2 = np.round(img_pts_nx2).astype("int")

        canvas = np.zeros([self._h, self._w, 3], "uint8")
        color = color or (0, 255, 0)

        for pt in img_pts_nx2:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, color, -1)

        cv2.namedWindow("img pts", cv2.WINDOW_NORMAL)
        cv2.imshow("img pts", canvas)
        cv2.waitKey(0)


class ParameterPackage(object):
    def __init__(self, alpha, beta, gamma, u0, v0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.u0 = u0
        self.v0 = v0

        self._extrinsic_params = []
        self._img_pts = []
        self._obj_pts = []

        self._NUM_INTRINSIC_PARAMS = 5

    @property
    def num_scenes(self):
        return len(self._img_pts)

    def add_scene_data(self, r, t, img_pts_nx2: np.ndarray, obj_pts_nx3: np.ndarray):
        rt_list = np.hstack([np.ravel(r), np.ravel(t)]).tolist()

        self._extrinsic_params += rt_list
        self._img_pts.append(img_pts_nx2.T)
        self._obj_pts.append(obj_pts_nx3.T)

    def pack_params_as_array(self):
        param_array = np.asarray(
            [self.alpha, self.beta, self.gamma, self.u0, self.v0]  # intrinsic
            + self._extrinsic_params  # extrinsic params
        )

        return param_array

    def unpack_params(self, param_array):
        param_array = np.asarray(param_array).ravel()

        self.alpha = param_array[0]
        self.beta = param_array[1]
        self.gamma = param_array[2]
        self.u0 = param_array[3]
        self.v0 = param_array[4]

        self._extrinsic_params = list(param_array[self._NUM_INTRINSIC_PARAMS :])

    def img_pts_2xn(self, index) -> np.ndarray:
        return self._img_pts[index]

    def obj_pts_3xn(self, index) -> np.ndarray:
        return self._obj_pts[index]

    def rt(self, index):
        index = index * 6

        r = np.asarray(self._extrinsic_params[index : index + 3])
        t = np.asarray(self._extrinsic_params[index + 3 : index + 6])

        return r, t

    def K_3x3(self) -> np.ndarray:
        return np.asarray(
            [
                [self.alpha, self.gamma, self.u0],
                [0, self.beta, self.v0],
                [0, 0, 1],
            ]
        )

    def repr_project_errors(self):
        num_scenes = len(self._img_pts)
        K = self.K_3x3()

        repr_errors = []

        for i in range(num_scenes):
            m_2xn = self.img_pts_2xn(i)
            M_3xn = self.obj_pts_3xn(i)
            r, t = self.rt(i)

            R, _ = cv2.Rodrigues(r)

            repr_m_3xn = K @ (R @ M_3xn + np.reshape(t, (3, 1)))
            repr_m_2xn = repr_m_3xn[:2] / repr_m_3xn[2]

            repr_errors_2xn = repr_m_2xn - m_2xn

            repr_errors.append(repr_errors_2xn)

        return np.asarray(repr_errors).ravel()


class CameraCalibrator(object):
    def __init__(self, board_resolution_mm, board_size_wh):
        self._board_resolution_mm = float(board_resolution_mm)
        self._board_size_wh = board_size_wh

        self._img_pts_nx2_list = []
        self._obj_pts_nx3_list = []

        self._obj_pts_nx3 = self.obj_pts_nx3()
        self._num_pts_each_pattern = board_size_wh[0] * board_size_wh[1]

    def obj_pts_nx3(self) -> np.ndarray:
        w, h = self._board_size_wh
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_z = np.zeros_like(grid_x)

        grid_xyz_nx3 = np.hstack(
            [grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)]
        )

        obj_pts_nx3 = grid_xyz_nx3 * self._board_resolution_mm

        return obj_pts_nx3

    def _check_img_pts_shape(self, img_pts_nx2: np.ndarray):
        assert img_pts_nx2.ndim == 2
        assert img_pts_nx2.shape == (self._num_pts_each_pattern, 2)

    def add_img_pts(self, img_pts_nx2: np.ndarray):
        self._check_img_pts_shape(img_pts_nx2)
        self._img_pts_nx2_list.append(img_pts_nx2)

    def calibrate(self):
        pass

    def estimate_homography(self, img_pts_nx2: np.ndarray):
        """Estimate homography matrix from calibration board plane to image plane.
        Refers to 1.Appendix A for more details

        """
        self._check_img_pts_shape(img_pts_nx2)

        obj_pts_nx3 = self.obj_pts_nx3()
        X = obj_pts_nx3[:, 0]
        Y = obj_pts_nx3[:, 1]
        W = np.ones_like(X)

        M_3xn = np.vstack([X, Y, W])

        u = img_pts_nx2[:, 0]
        v = img_pts_nx2[:, 1]
        m_2xn = np.vstack([u, v])
        m_3xn = np.vstack([u, v, np.ones_like(u)])

        A_2nx9 = []

        # construct coefficient matrix
        for i in range(self._num_pts_each_pattern):
            u, v = img_pts_nx2[i]
            X, Y = obj_pts_nx3[i, :2]  # ignore Z, cause Z=0 always

            a_2x9 = np.asarray(
                [
                    [X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u],
                    [0, 0, 0, X, Y, 1, -v * X, -v * Y, -v],
                ]
            )

            A_2nx9.append(a_2x9)

        A_2nx9 = np.vstack(A_2nx9)

        assert A_2nx9.shape == (2 * self._num_pts_each_pattern, 9)

        # solve Ah = 0 to get an initial guess for H
        # do SVD to A, we have A = U * S * Vh
        # and h is the right singular vector of A associated with smallest singular value
        U, S, Vh = np.linalg.svd(A_2nx9)

        print("smallest singular value: ", S[-1])

        # h = [h1, h2, h3], where h1, h2, h3 are rows of H
        init_h = Vh[-1]

        # make h33 = 1
        init_h /= init_h[-1]

        # refine H by minimize re-projection error
        def repr_error_func(h_vec):
            h_vec /= h_vec[-1]
            H_3x3 = np.asarray([h_vec[:3], h_vec[3:6], h_vec[6:]])

            # m_3xn = H @ M_3xn
            repr_m_3xn = H_3x3 @ M_3xn
            repr_m_2xn = repr_m_3xn[:2] / repr_m_3xn[2]

            # repr_errors_vec = np.linalg.norm(repr_m_2xn - m_2xn, axis=0)

            return (repr_m_2xn - m_2xn).ravel()

        init_repr_errors = repr_error_func(init_h)

        opt_h, *_ = levmar.levmar(
            repr_error_func, init_h, np.zeros_like(init_repr_errors)
        )

        opt_repr_errors = repr_error_func(opt_h)

        init_repr_error = np.mean(np.abs(init_repr_errors))
        opt_repr_error = np.mean(np.abs(opt_repr_errors))

        print(f"init repr error: {init_repr_error}, opt repr error: {opt_repr_error}")

        opt_H_3x3 = np.asarray([opt_h[:3], opt_h[3:6], opt_h[6:]])

        cv_H, mask = cv2.findHomography(M_3xn.T, m_3xn.T, cv2.RANSAC)

        cv_repr_error = np.mean(repr_error_func(cv_H.flatten()))
        print("cv repr error: ", cv_repr_error)
        print(cv_H.flatten())

        return opt_H_3x3, opt_repr_error

    def estimate_camera_matrix(self, img_pts_mxnx2):
        def vij_vec(H, i, j):
            hi1, hi2, hi3 = H[:, i]
            hj1, hj2, hj3 = H[:, j]

            vij_vec6 = np.asarray(
                [
                    hi1 * hj1,
                    hi1 * hj2 + hi2 * hj1,
                    hi2 * hj2,
                    hi3 * hj1 + hi1 * hj3,
                    hi3 * hj2 + hi2 * hj3,
                    hi3 * hj3,
                ]
            )

            return vij_vec6

        A_2mx6 = []
        H_nx3x3 = []
        for img_pts_nx2 in img_pts_mxnx2:
            Hi, repr_err = self.estimate_homography(img_pts_nx2)

            # const coefficients matrix of eq.8
            a_2x6 = np.vstack(
                [
                    vij_vec(Hi, 0, 1),
                    vij_vec(Hi, 0, 0) - vij_vec(Hi, 1, 1),
                    [0, 1, 0, 0, 0, 0],  # gamma = 0
                ]
            )
            H_nx3x3.append(Hi)
            A_2mx6.append(a_2x6)

        A_2mx6 = np.vstack(A_2mx6)

        # solve Ab = 0
        U, S, Vh = np.linalg.svd(A_2mx6)
        b = Vh[-1]
        B11, B12, B22, B13, B23, B33 = b

        # compute camera intrinsic parameters
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
        lambda_ = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
        alpha = np.sqrt(lambda_ / B11)
        beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12 ** 2))
        gamma = -B12 * alpha ** 2 * beta / lambda_
        u0 = gamma * v0 / beta - B13 * alpha ** 2 / lambda_

        init_K = np.asarray(
            [
                [alpha, gamma, u0],
                [0, beta, v0],
                [0, 0, 1],
            ],
        )

        # compute extrinsic matrix
        params_package = ParameterPackage(alpha, beta, gamma, u0, v0)

        inv_K = np.linalg.inv(init_K)
        for i, Hi in enumerate(H_nx3x3):
            h1, h2, h3 = Hi[:, 0], Hi[:, 1], Hi[:, 2]

            r1 = inv_K @ h1
            r2 = inv_K @ h2

            lambda_ = 1 / ((np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0)

            r1 *= lambda_
            r2 *= lambda_
            r3 = np.cross(r1, r2)
            t = lambda_ * inv_K @ h3

            # r1, r2, r3 are columns or rotation matrix
            Ri_3x3 = np.vstack([r1, r2, r3]).T

            # estimate best rotation matrix from Ri, cause Ri may not satisfy rotation matrix properties
            U, S, Vh = np.linalg.svd(Ri_3x3)
            R = U @ Vh
            r, _ = cv2.Rodrigues(R)

            params_package.add_scene_data(r, t, img_pts_mxnx2[i], self.obj_pts_nx3())

        # refine by minimize re-projection error
        def repr_error_func(params_array):
            params_package.unpack_params(params_array)
            return params_package.repr_project_errors()

        init_errors = repr_error_func(params_package.pack_params_as_array())

        print(
            "init repr errors: ",
            np.mean(np.abs(init_errors)),
            np.max(np.abs(init_errors)),
            np.std(np.abs(init_errors)),
        )
        print("init intrinsic matrix: ")
        print(init_K)

        opt_params, *_ = levmar.levmar(
            repr_error_func,
            params_package.pack_params_as_array(),
            np.zeros_like(init_errors),
        )

        opt_errors = repr_error_func(opt_params)
        print(
            "opt repr errors: ",
            np.mean(np.abs(opt_errors)),
            np.max(np.abs(opt_errors)),
            np.std(np.abs(opt_errors)),
        )

        params_package.unpack_params(opt_params)

        K = params_package.K_3x3()
        print("opt intrinsic matrix: ")
        print(K)

        return K


def main():
    simulator = SingleCameraSimulator(
        [1280, 1024], fx=2557.1254, fy=2556.48784, cx=662.1456, cy=544.5881, skew=1.5
    )
    calibrator = CameraCalibrator(3, (7, 7))

    num_images = 15

    img_pts_mxnx2 = []
    obj_pts_mxnx3 = []

    for _ in range(num_images):
        img_pts_nx2, P_3x4 = simulator.random_img_pts(
            calibrator.obj_pts_nx3(), noise_sigma=0.2
        )

        img_pts_mxnx2.append(img_pts_nx2)
        obj_pts_mxnx3.append(calibrator.obj_pts_nx3())

        # simulator.show_img_pts(img_pts_nx2)

    calibrated_K_3x3 = calibrator.estimate_camera_matrix(img_pts_mxnx2)
    print("idle intrinsic: ")
    print(simulator.K_3x3)
    print("calibrated intrinsic: ")
    print(calibrated_K_3x3)


if __name__ == "__main__":
    main()
