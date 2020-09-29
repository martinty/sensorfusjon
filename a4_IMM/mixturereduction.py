from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights shape=(N,)
    x: np.ndarray,  # the mixture means shape(N, n)
    P: np.ndarray,  # the mixture covariances shape (N, n, n)
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the mean and covariance of of the mixture shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""

    # mean
    x_bar = np.average(x, axis=0, weights=w)  # TODO: hint np.average using axis and weights argument

    # covariance
    # # internal covariance
    P_int = np.average(P, axis=0, weights=w)  # TODO: hint, also an average

    # # spread of means
    # Optional calc: mean_diff =
    x_diff = x - x_bar[None]  # TODO: hint, also an average
    P_ext = np.average(x_diff[:, :, None] * x_diff[:, None, :], axis=0, weights=w)

    # # total covariance
    P_bar = P_int + P_ext  # TODO

    return P_bar, x_bar
