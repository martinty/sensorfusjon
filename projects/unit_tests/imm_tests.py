import os
import unittest

import numpy as np
import scipy

from utils.dynamicmodels import WhitenoiseAccelleration, ConstantTurnrate
from filters.ekf import EKF
from utils.gaussparams import GaussParams
from filters.imm import IMM
from utils.measurementmodels import CartesianPosition
from utils.mixturedata import MixtureParameters
from run_files.run_imm import loaded_data


class PreGenData(object):
    def __init__(self):
        self.data_filename = os.path.join("data_for_imm.mat")
        self.loaded_data = scipy.io.loadmat(self.data_filename)
        self.Z = loaded_data["Z"].T
        self.K = loaded_data["K"].item()
        self.Ts = loaded_data["Ts"].item()
        self.Xgt = loaded_data["Xgt"].T

class TestIMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.pregen = PreGenData()
        # Init settings
        sigma_z = 3
        sigma_a_CV = 0.2
        sigma_a_CT = 0.1
        sigma_omega = 0.002 * np.pi
        cls.ts = 0.1
        # Transition matrix
        PI = np.array([[0.95, 0.05],
                       [0.05, 0.95]])

        assert np.allclose(PI.sum(axis=1), 1), "rows of PI must sum to 1"

        measurement_model = CartesianPosition(sigma_z, state_dim=5)
        CV = WhitenoiseAccelleration(sigma_a_CV, n=5)
        CT = ConstantTurnrate(sigma_a_CT, sigma_omega)
        ekf_filters = [EKF(CV, measurement_model), EKF(CT, measurement_model)]

        cls.imm_filter = IMM(ekf_filters, PI)

        # IMM init weights
        init_weights = np.array([0.5] * 2)
        init_mean = [0] * 5
        init_cov = np.diag([1] * 5)  # HAVE TO BE DIFFERENT: use intuition, eg. diag guessed distance to true values squared.
        init_mode_states = [GaussParams(init_mean, init_cov)] * 2  # copy of the two modes
        cls.init_immstate = MixtureParameters(init_weights, init_mode_states)

    def test_mix_probabilities(self):
        self.imm_filter.mix_probabilities(self.init_immstate, self.ts)

    def test_mix_states(self):
        _, mix_prob = self.imm_filter.mix_probabilities(self.init_immstate, self.ts)
        mix_state = self.imm_filter.mix_states(self.init_immstate, mix_prob)

    def test_update_mode_probability(self):
        self.imm_filter.update_mode_probabilities(self.pregen.Z, self.init_immstate)


if __name__ == "__main__":
    unittest.main()