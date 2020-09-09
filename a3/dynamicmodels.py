#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic models to be used with eg. EKF.

"""
# %%
from typing import Optional, Sequence
from typing_extensions import Final, Protocol
from dataclasses import dataclass, field

import numpy as np

# %% the dynamic models interface declaration


class DynamicModel(Protocol):
    n: int
    def f(self, x: np.ndarray, Ts: float) -> np.ndarray: ...
    def F(self, x: np.ndarray, Ts: float) -> np.ndarray: ...
    def Q(self, x: np.ndarray, Ts: float) -> np.ndarray: ...

# %%


@dataclass
class WhitenoiseAccelleration:
    """
    A white noise accelereation model also known as CV, states are position and speed.

    The model includes the discrete prediction equation f, its Jacobian F, and
    the process noise covariance Q as methods.
    """
    # noise standard deviation
    sigma: float
    # number of dimensions
    dim: int = 2
    # number of states
    n: int = 4

    def f(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """
        Calculate the zero noise Ts time units transition from x.

        x[:2] is position, x[2:4] is velocity
        """
        # TODO
        return None

    def F(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """ Calculate the transition function jacobian for Ts time units at x."""
        # TODO
        return None

    def Q(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """
        Calculate the Ts time units transition Covariance.
        """
        # TODO
        # Hint: sigma can be found as self.sigma, see variable declarations
        # Note the @dataclass decorates this class to create an init function that takes
        # sigma as a parameter, among other things.
        return None
