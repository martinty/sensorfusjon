#%%
from typing import Dict, Any, Generic, TypeVar
from typing_extensions import Protocol, runtime

from utils.dynamicmodels import DynamicModel
from utils.measurementmodels import MeasurementModel
from utils.mixturedata import MixtureParameters
from utils.gaussparams import GaussParams

import numpy as np


T = TypeVar("T")


@runtime
class StateEstimator(Protocol[T]):
    # A Protocol so duck typing can be used
    dynamic_model: DynamicModel
    # A Protocol so duck typing can be used
    sensor_model: MeasurementModel

    def predict(self, eststate: T, Ts: float) -> T:
        ...

    def update(
        self, z: np.ndarray, eststate: T, *, sensor_state: Dict[str, Any] = None
    ) -> T:
        ...

    def step(self, z: np.ndarray, eststate: T, Ts: float) -> T:
        ...

    def estimate(self, eststate: T) -> GaussParams:
        ...

    def init_filter_state(self, init: Any) -> T:
        ...

    def innovation(self, z: np.ndarray, eststate: GaussParams, *, sensor_state: Dict[str, Any] = None,
    ) -> GaussParams:
        ...

    def loglikelihood(
        self, z: np.ndarray, eststate: T, *, sensor_state: Dict[str, Any] = None
    ) -> float:
        ...

    def reduce_mixture(self, estimator_mixture: MixtureParameters[T]) -> T:
        ... #todo must not

    def gate(self, z: np.ndarray, eststate: T, gate_size: float, *, sensor_state: Dict[str, Any] = None
    ) -> bool:
        ...

    def NIS(self, z: np.ndarray, eststate: T, *, sensor_state: Dict[str, Any] = None,
    ) -> float: ...

    def NEES(self, x_true: np.ndarray, eststate: T, *, sensor_state: Dict[str, Any] = None,
    ) -> float: ...

    def NEES_from_gt(self, x_pred: np.ndarray, x_gt: np.ndarray, cov_matr: np.ndarray) -> float:
        ...