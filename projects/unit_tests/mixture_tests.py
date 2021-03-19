import unittest

import numpy as np

from utils.mixturereduction import gaussian_mixture_moments


class TestMixtureReduction(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # N - amount of mixtures , n - dimension of each mixture
        cls.N = 3
        cls.n = 2
        cls.sigma = np.random.randn(1)  # One sample from a random normal distribution

        # Initialize with equal weight
        cls.random_weights = np.ones(cls.N) / cls.N
        cls.random_means = np.random.randint(0, 2, size=(cls.N, cls.n))
        cls.covs = np.array([np.eye(cls.n)*cls.sigma for it in range(cls.N)])  # Repeat cov matr, N times

    def test_gaussian_reduction_shape(self):
        reduced_gaussian = gaussian_mixture_moments(self.random_weights, self.random_means, self.covs ** 2)
        self.assertEqual((reduced_gaussian[0].shape, reduced_gaussian[1].shape),
                         ((self.n,), (self.n, self.n)))


if __name__ == "__main__":
    unittest.main()