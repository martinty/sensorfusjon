from typing import Tuple
import numpy as np
from scipy.linalg import block_diag
import scipy.linalg as la
from gradedSLAM.utils import rotmat2d
from gradedSLAM.JCBB import JCBB
from gradedSLAM import utils

# import line_profiler
# import atexit

# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


class EKFSLAM:
    def __init__(
        self,
        Q,
        R,
        do_asso=False,
        alphas=np.array([0.001, 0.0001]),
        sensor_offset=np.zeros(2),
    ):

        self.Q = Q
        self.R = R
        self.do_asso = do_asso
        self.alphas = alphas
        self.sensor_offset = sensor_offset

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Add the odometry u to the robot state x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray, shape = (3,)
            the predicted state
        """
        # TODO, eq (11.7). Should wrap heading angle between (-pi, pi), see utils.wrapToPi

        u_k = u[0]
        v_k = u[1]
        phi_k = u[2]

        x_k_1 = x[0]
        y_k_1 = x[1]
        psi_k_1 = utils.wrapToPi(x[2])

        x_pred = np.array([
            x_k_1 + u_k*np.cos(psi_k_1) - v_k*np.sin(psi_k_1),
            y_k_1 + u_k*np.sin(psi_k_1) + v_k*np.cos(psi_k_1),
            psi_k_1 + phi_k
        ])

        assert x_pred.shape == (3,), "EKFSLAM.f: wrong shape for xpred"
        return x_pred

    def Fx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. x.
        """
        # TODO, eq (11.13)
        u_k = u[0]
        v_k = u[1]
        psi = x[2]

        Fx = np.array([
            [1, 0, -u_k*np.sin(psi) - v_k*np.cos(psi)],
            [0, 1, u_k*np.cos(psi) - v_k*np.sin(psi)],
            [0, 0, 1]
        ])

        assert Fx.shape == (3, 3), "EKFSLAM.Fx: wrong shape"
        return Fx

    def Fu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to u.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. u.
        """
        # TODO, eq (11.14)
        psi = x[2]

        Fu = np.array([
            [np.cos(psi),   -np.sin(psi),   0],
            [np.sin(psi),   np.cos(psi),    0],
            [0,                 0,          1]
        ])

        assert Fu.shape == (3, 3), "EKFSLAM.Fu: wrong shape"
        return Fu

    def predict(
        self, eta: np.ndarray, P: np.ndarray, z_odo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the robot state using the zOdo as odometry the corresponding state&map covariance.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z_odo : np.ndarray, shape=(3,)
            the measured odometry

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes= (3 + 2*#landmarks,), (3 + 2*#landmarks,)*2
            predicted mean and covariance of eta.
        """
        # check inout matrix
        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P input"
        assert np.all(
            np.linalg.eigvals(P) >= 0
        ), "EKFSLAM.predict: non-positive eigen values in P input"
        assert (
            eta.shape * 2 == P.shape
        ), "EKFSLAM.predict: input eta and P shape do not match"
        etapred = np.empty_like(eta)

        # eq. 11.17 and 11.18

        x = eta[:3]
        etapred[:3] = self.f(x, z_odo)  # TODO robot state prediction
        etapred[3:] = eta[3:]  # TODO landmarks: no effect

        Fx = self.Fx(x, z_odo)  # TODO
        Fu = self.Fu(x, z_odo)  # TODO

        # evaluate covariance prediction in place to save computation
        # only robot state changes, so only rows and colums of robot state needs changing
        # cov matrix layout:
        # [[P_xx, P_xm],
        # [P_mx, P_mm]]
        P[:3, :3] = Fx @ P[:3, :3] @ Fx.T + Fu @ self.Q @ Fu.T  # TODO robot cov prediction
        P[:3, 3:] = Fx @ P[:3, 3:]  # TODO robot-map covariance prediction
        P[3:, :3] = P[3:, :3] @ Fx.T  # TODO map-robot covariance: transpose of the above

        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P"
        assert np.all(
            np.linalg.eigvals(P) > 0
        ), "EKFSLAM.predict: non-positive eigen values"
        assert (
            etapred.shape * 2 == P.shape
        ), "EKFSLAM.predict: calculated shapes does not match"
        return etapred, P

    def h(self, eta: np.ndarray) -> np.ndarray:
        """Predict all the landmark positions in sensor frame.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks,)
            The landmarks in the sensor frame.
        """
        # extract states and map
        x = eta[0:3]
        pos = np.array([x[0], x[1]])
        psi = x[2]
        # reshape map (2, #landmarks), m[:, j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T

        Rot_world = rotmat2d(psi)
        Rot_body = rotmat2d(-psi)
        sensor_offset_world = Rot_world @ self.sensor_offset

        delta_m = (m.T - pos).T
        zc = (delta_m.T - sensor_offset_world).T  # world frame
        zb = Rot_body @ zc  # body frame

        # TODO: eq. 11.11,  Cartesian-to-polar conversion
        zpred_r = la.norm(zb, axis=0)
        zpred_theta = np.arctan2(zb[1, :], zb[0, :])

        zpred = np.vstack((zpred_r, zpred_theta))  # [ranges; bearings], shape (2, #lmrk)
        zpred = zpred.T.ravel()  # [range1 bearing1 range2 bearing2 ...], shape (1, #lmrk * 2)

        assert (
            zpred.ndim == 1 and zpred.shape[0] == eta.shape[0] - 3
        ), "SLAM.h: Wrong shape on zpred"
        return zpred

    def H(self, eta: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of h.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """
        # extract states and map
        x = eta[0:3]
        pos = np.array([x[0], x[1]])
        psi = x[2]
        #  reshape map (2, #landmarks), m[j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T

        numM = m.shape[1]

        Rpihalf = rotmat2d(np.pi / 2)
        Rot_world = rotmat2d(psi)
        sensor_offset_world = Rot_world @ self.sensor_offset

        delta_m = (m.T - pos).T
        zc = (delta_m.T - sensor_offset_world).T  # world frame

        # zpred = self.h(eta)
        # zr = zpred[::2]

        # In what follows you can be clever and avoid making this for all the landmarks you _know_
        # you will not detect (the maximum range should be available from the data).
        # But keep it simple to begin with.

        # Allocate H and set submatrices as memory views into H
        # You may or may not want to do this like this
        H = np.zeros((2 * numM, 3 + 2 * numM))  # TODO, see eq (11.15), (11.16), (11.17)
        Hx = H[:, :3]  # slice view, setting elements of Hx will set H as well
        Hm = H[:, 3:]  # slice view, setting elements of Hm will set H as well

        # proposed way is to go through landmarks one by one
        jac_z_cb = -np.eye(2, 3)  # preallocate and update this for some speed gain if looping
        for i in range(numM):  # But this whole loop can be vectorized
            ind = 2 * i  # starting postion of the ith landmark into H
            inds = slice(ind, ind + 2)  # the inds slice for the ith landmark into H

            jac_z_cb[:, 2] = -Rpihalf @ delta_m[:, i]
            zc_T = zc[:, i].reshape((1, 2))

            # TODO: Set H or Hx and Hm here (eq. 11.15 and 11.16)
            Hx[ind, :] = (zc_T / la.norm(zc_T)) @ jac_z_cb
            Hx[ind + 1, :] = (zc_T @ Rpihalf.T) / (la.norm(zc_T) ** 2) @ jac_z_cb
            Hm[inds, inds] = -Hx[inds, 0:2]  # Columns are the same

        # TODO: You can set some assertions here to make sure that some of the structure in H is correct
        return H

    def add_landmarks(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate new landmarks, their covariances and add them to the state.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z : np.ndarray, shape(2 * #newlandmarks,)
            A set of measurements to create landmarks for

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes=(3 + 2*(#landmarks + #newlandmarks,), (3 + 2*(#landmarks + #newlandmarks,)*2
            eta with new landmarks appended, and its covariance
        """
        n = P.shape[0]
        assert z.ndim == 1, "SLAM.add_landmarks: z must be a 1d array"

        numLmk = z.shape[0] // 2

        lmnew = np.empty_like(z)

        Gx = np.empty((numLmk * 2, 3))
        Rall = np.zeros((numLmk * 2, numLmk * 2))

        psi = eta[2]
        Rot_world = rotmat2d(psi)
        sensor_offset_world = Rot_world @ self.sensor_offset  # For transforming landmark position into world frame
        sensor_offset_world_der = rotmat2d(psi + np.pi / 2) @ self.sensor_offset  # Used in Gx
        I2 = np.eye(2)  # Preallocate, used for Gx

        for j in range(numLmk):
            ind = 2 * j
            inds = slice(ind, ind + 2)

            zj = z[inds]
            z_r = zj[0]
            z_phi = zj[1]

            # TODO: from polar to cartesian coordinates
            pos_x = z_r * np.cos(z_phi)
            pos_y = z_r * np.sin(z_phi)
            pos = np.array([pos_x, pos_y])

            # TODO, calculate position of new landmark in world frame
            lmnew[inds] = Rot_world @ pos + sensor_offset_world + eta[:2]

            # TODO: Gx from task 1g
            Gx[inds, :2] = I2
            sin_cos_matrix = np.array([-np.sin(z_phi + psi), np.cos(z_phi + psi)])
            Gx[inds, 2] = z_r * sin_cos_matrix + sensor_offset_world_der

            # TODO Gz from task 1g
            Gz = rotmat2d(z_phi + psi) @ np.diag([1, z_r])

            # TODO, Gz * R * Gz^T, transform measurement covariance from polar to cartesian coordinates
            Rall[inds, inds] = Gz @ self.R @ Gz.T

        assert len(lmnew) % 2 == 0, "SLAM.add_landmark: lmnew not even length"

        # TODO, append new landmarks to state vector
        etaadded = np.block([eta, lmnew])

        # TODO, block diagonal of P_new, see problem text in 1g) in graded assignment 3
        Padded = np.zeros((n + Rall.shape[0], n + Rall.shape[0]))
        # TODO: diagonal of P_new
        Padded[:n, :n] = P
        Padded[n:, n:] = Gx @ P[:3, :3] @ Gx.T + Rall
        # TODO: top right corner of P_new
        Padded[:n, n:] = P[:, :3] @ Gx.T
        # TODO, transpose of above. Should yield the same as calcualion, but this enforces symmetry and should be cheaper
        Padded[n:, :n] = Padded[:n, n:].T

        assert (
            etaadded.shape * 2 == Padded.shape
        ), "EKFSLAM.add_landmarks: calculated eta and P has wrong shape"
        assert np.allclose(
            Padded, Padded.T
        ), "EKFSLAM.add_landmarks: Padded not symmetric"
        assert np.all(
            np.linalg.eigvals(Padded) >= 0
        ), "EKFSLAM.add_landmarks: Padded not PSD"

        return etaadded, Padded

    def associate(
        self, z: np.ndarray, zpred: np.ndarray, H: np.ndarray, S: np.ndarray,
    ):  # -> Tuple[*((np.ndarray,) * 5)]:
        """Associate landmarks and measurements, and extract correct matrices for these.

        Parameters
        ----------
        z : np.ndarray,
            The measurements all in one vector
        zpred : np.ndarray
            Predicted measurements in one vector
        H : np.ndarray
            The measurement Jacobian matrix related to zpred
        S : np.ndarray
            The innovation covariance related to zpred

        Returns
        -------
        Tuple[*((np.ndarray,) * 5)]
            The extracted measurements, the corresponding zpred, H, S and the associations.

        Note
        ----
        See the associations are calculated  using JCBB. See this function for documentation
        of the returned association and the association procedure.
        """
        if self.do_asso:
            # Associate
            a = JCBB(z, zpred, S, self.alphas[0], self.alphas[1])

            # Extract associated measurements
            zinds = np.empty_like(z, dtype=bool)
            zinds[::2] = a > -1  # -1 means no association
            zinds[1::2] = zinds[::2]
            zass = z[zinds]

            # extract and rearange predicted measurements and cov
            zbarinds = np.empty_like(zass, dtype=int)
            zbarinds[::2] = 2 * a[a > -1]
            zbarinds[1::2] = 2 * a[a > -1] + 1

            zpredass = zpred[zbarinds]
            Sass = S[zbarinds][:, zbarinds]
            Hass = H[zbarinds]

            assert zpredass.shape == zass.shape
            assert Sass.shape == zpredass.shape * 2
            assert Hass.shape[0] == zpredass.shape[0]

            return zass, zpredass, Hass, Sass, a
        else:
            # should one do something her
            pass

    def update(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Update eta and P with z, associating landmarks and adding new ones.

        Parameters
        ----------
        eta : np.ndarray
            [description]
        P : np.ndarray
            [description]
        z : np.ndarray, shape=(#detections, 2)
            [description]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, np.ndarray]
            [description]
        """
        numLmk = (eta.size - 3) // 2
        assert (len(eta) - 3) % 2 == 0, "EKFSLAM.update: landmark lenght not even"

        if numLmk > 0:
            # Prediction and innovation covariance
            zpred = self.h(eta)  # TODO
            H = self.H(eta)  # TODO

            # Here you can use simply np.kron (a bit slow) to form the big (very big in VP after a while) R,
            # or be smart with indexing and broadcasting (3d indexing into 2d mat) realizing you are adding the same R on all diagonals
            # kron = np.kron(np.eye(numLmk), self.R)
            # S = H @ P @ H.T + kron

            # TODO: calculate S matrix without kron (eq. 11.19)
            idx = np.arange(2 * numLmk).reshape(-1, 2)
            S = H @ P @ H.T
            S[idx[:, :, None], idx[:, None]] += self.R[None]

            assert (
                S.shape == zpred.shape * 2
            ), "EKFSLAM.update: wrong shape on either S or zpred"
            z = z.ravel()  # 2D -> flat

            # Perform data association
            za, zpred, Ha, Sa, a = self.associate(z, zpred, H, S)

            # No association could be made, so skip update
            if za.shape[0] == 0:
                etaupd = eta  # TODO
                Pupd = P  # TODO
                NIS = 1  # TODO: beware this one when analysing consistency.

            else:
                # Create the associated innovation
                v = za.ravel() - zpred  # za: 2D -> flat
                v[1::2] = utils.wrapToPi(v[1::2])

                # Kalman mean update
                # S_cho_factors = la.cho_factor(Sa) # Optional, used in places for S^-1, see scipy.linalg.cho_factor and scipy.linalg.cho_solve

                # W = P @ Ha.T @ la.inv(Sa)  # TODO, Kalman gain, can use S_cho_factors
                S_a_cho_factors = la.cho_factor(Sa, lower=True)
                S_a_cho_inv = la.cho_solve(S_a_cho_factors, np.eye(Sa.shape[0]))
                W = P @ Ha.T @ S_a_cho_inv
                etaupd = eta + W @ v  # TODO, Kalman update

                # TODO: Kalman cov update: use Joseph form for stability
                jo = -W @ Ha
                jo[np.diag_indices(jo.shape[0])] += 1  # same as adding Identity mat
                Pupd = jo @ P  # TODO, Kalman update. This is the main workload on VP after speedups

                # TODO: calculate NIS, can use S_cho_factors
                # NIS = v.T @ la.inv(Sa) @ v
                NIS = v.T @ S_a_cho_inv @ v

                # When tested, remove for speed
                assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd not symmetric"
                assert np.all(
                    np.linalg.eigvals(Pupd) > 0
                ), "EKFSLAM.update: Pupd not positive definite"

        else:  # All measurements are new landmarks,
            a = np.full(z.shape[0], -1)
            z = z.flatten()
            NIS = 0  # TODO: beware this one, you can change the value to for instance 1
            etaupd = eta
            Pupd = P

        # Create new landmarks if any is available
        if self.do_asso:
            is_new_lmk = a == -1
            if np.any(is_new_lmk):
                z_new_inds = np.empty_like(z, dtype=bool)
                z_new_inds[::2] = is_new_lmk
                z_new_inds[1::2] = is_new_lmk
                z_new = z[z_new_inds]
                etaupd, Pupd = self.add_landmarks(etaupd, Pupd, z_new)  # TODO, add new landmarks.

        assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd must be symmetric"
        assert np.all(np.linalg.eigvals(Pupd) >= 0), "EKFSLAM.update: Pupd must be PSD"

        return etaupd, Pupd, NIS, a

    @classmethod
    def NEESes(cls, x: np.ndarray, P: np.ndarray, x_gt: np.ndarray,) -> np.ndarray:
        """Calculates the total NEES and the NEES for the substates
        Args:
            x (np.ndarray): The estimate
            P (np.ndarray): The state covariance
            x_gt (np.ndarray): The ground truth
        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties
        Returns:
            np.ndarray: NEES for [all, position, heading], shape (3,)
        """

        assert x.shape == (3,), f"EKFSLAM.NEES: x shape incorrect {x.shape}"
        assert P.shape == (3, 3), f"EKFSLAM.NEES: P shape incorrect {P.shape}"
        assert x_gt.shape == (3,), f"EKFSLAM.NEES: x_gt shape incorrect {x_gt.shape}"

        d_x = x - x_gt
        d_x[2] = utils.wrapToPi(d_x[2])
        assert (
            -np.pi <= d_x[2] <= np.pi
        ), "EKFSLAM.NEES: error heading must be between (-pi, pi)"

        d_p = d_x[0:2]
        P_p = P[0:2, 0:2]
        assert d_p.shape == (2,), "EKFSLAM.NEES: d_p must be 2 long"
        d_heading = d_x[2]  # Note: scalar
        assert np.ndim(d_heading) == 0, "EKFSLAM.NEES: d_heading must be scalar"
        P_heading = P[2, 2]  # Note: scalar
        assert np.ndim(P_heading) == 0, "EKFSLAM.NEES: P_heading must be scalar"

        # NB: Needs to handle both vectors and scalars! Additionally, must handle division by zero
        NEES_all = d_x @ (np.linalg.solve(P, d_x))
        NEES_pos = d_p @ (np.linalg.solve(P_p, d_p))
        try:
            NEES_heading = d_heading ** 2 / P_heading
        except ZeroDivisionError:
            NEES_heading = 1.0  # TODO: beware

        NEESes = np.array([NEES_all, NEES_pos, NEES_heading])
        NEESes[np.isnan(NEESes)] = 1.0  # We may divide by zero, # TODO: beware

        assert np.all(NEESes >= 0), "ESKF.NEES: one or more negative NEESes"
        return NEESes
