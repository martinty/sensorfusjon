"""

"""
# %% Imports

# types
from typing import (
    Tuple,
    List,
    TypeVar,
    Optional,
    Dict,
    Any,
    Union,
    Sequence,
    Generic,
    Iterable, Sized,
)

from utils import discretebayes
from utils.mixturedata import MixtureParameters
from utils.gaussparams import GaussParams
from utils.estimatorduck import StateEstimator

# packages
from dataclasses import dataclass
import numpy as np
from scipy import linalg

from scipy.special import logsumexp

# local

# %% TypeVar and aliases
from utils.mixturereduction import gaussian_mixture_moments

MT = TypeVar("MT")  # a type variable to be the mode type

# %% IMM
@dataclass
class IMM(Generic[MT]):
    # The M filters the IMM relies on
    filters: List[StateEstimator[MT]]
    # the transition matrix. PI[i, j] = probability of going from model i to j: shape (M, M)
    PI: np.ndarray
    # init mode probabilities if none is given
    initial_mode_probabilities: Optional[np.ndarray] = None

    def __post_init__(self):
        # This have to be satisfied!
        if not np.allclose(self.PI.sum(axis=1), 1):
            raise ValueError("The rows of the transition matrix PI must sum to 1.")

        # Nice to have a reasonable initial mode probability
        if self.initial_mode_probabilities is None:
            eigvals, eigvecs = linalg.eig(self.PI)
            self.initial_mode_probabilities = eigvecs[:, eigvals.argmax()]
            self.initial_mode_probabilities = (
                self.initial_mode_probabilities / self.initial_mode_probabilities.sum()
            )

    def mix_probabilities(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> Tuple[
        np.ndarray, np.ndarray
    ]:  # predicted_mode_probabilities, mix_probabilities: shapes = ((M, (M ,M))).
        # mix_probabilities[s] is the mixture weights for mode s
        """Calculate the predicted mode probability and the mixing probabilities."""

        predicted_mode_probabilities, mix_probabilities = discretebayes.discrete_bayes(immstate.weights, self.PI)

        # Optional assertions for debugging
        assert np.all(np.isfinite(predicted_mode_probabilities))
        assert np.all(np.isfinite(mix_probabilities))
        assert np.allclose(mix_probabilities.sum(axis=1), 1)

        return predicted_mode_probabilities, mix_probabilities

    def mix_states(
        self,
        immstate: MixtureParameters[MT],
        # the mixing probabilities: shape=(M, M)
        mix_probabilities: np.ndarray,
    ) -> List[MT]:

        components = immstate.components
        means = np.array([comp.mean for comp in components])
        covs = np.array([comp.cov for comp in components])

        # Columnwise iteration on mix_probs, transponated for columns
        mixed_states = [GaussParams(*gaussian_mixture_moments(w=mix_prob, mean=means, cov=covs))
                        for mix_prob in mix_probabilities.T]
        return mixed_states

    def mode_matched_prediction(
        self,
        mode_states: List[MT],
        # The sampling time
        Ts: float,
    ) -> List[MT]:

        modestates_pred = [fs.predict(mode_state, Ts)
                           for fs, mode_state
                           in zip(self.filters, mode_states)]

        return modestates_pred

    def predict(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> MixtureParameters[MT]:
        """
        Predict the immstate Ts time units ahead approximating the mixture step.

        Ie. Predict mode probabilities, condition states on predicted mode,
        appoximate resulting state distribution as Gaussian for each mode, then predict each mode.
        """

        predicted_mode_probability, mixing_probability = self.mix_probabilities(immstate, Ts)

        mixed_mode_states: List[MT] = self.mix_states(immstate, mixing_probability)

        predicted_mode_states = self.mode_matched_prediction(mode_states=mixed_mode_states, Ts=Ts)

        predicted_immstate = MixtureParameters(
            predicted_mode_probability, predicted_mode_states
        )
        return predicted_immstate

    def mode_matched_update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> List[MT]:
        """Update each mode in immstate with z in sensor_state.
           We assume the state predictions and other parameters are enclosed in immstate,
           we only run state update from EKF cycle.
        """

        updated_state = [fs.update(z, mode_state, sensor_state=sensor_state)
                         for fs, mode_state
                         in zip(self.filters, immstate.components)]

        return updated_state

    def update_mode_probabilities(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Calculate the mode probabilities in immstate updated with z in sensor_state"""

        # returns list of log likelihood scalars for each mode [log(p_sk1), log(p_sk2), .. ]
        mode_cond_likelihoods = [self.mode_loglikelihood(z, filt, gauss_params, sensor_state=sensor_state)
                                 for filt, gauss_params in zip(self.filters, immstate.components)]

        log_joint = mode_cond_likelihoods + np.log(immstate.weights)  # Assume mode probs inside state
        updated_mode_probabilities = np.exp(log_joint - logsumexp(log_joint))
        # Optional debuging
        assert np.all(np.isfinite(updated_mode_probabilities))
        assert np.allclose(np.sum(updated_mode_probabilities), 1)

        return updated_mode_probabilities

    def update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Update the immstate with z in sensor_state."""

        updated_weights = self.update_mode_probabilities(z, immstate, sensor_state)
        updated_states = self.mode_matched_update(z, immstate, sensor_state)

        updated_immstate = MixtureParameters(updated_weights, updated_states)
        return updated_immstate

    def step(
        self,
        z,
        immstate: MixtureParameters[MT],
        Ts: float,
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Predict immstate with Ts time units followed by updating it with z in sensor_state"""

        predicted_immstate = self.predict(immstate, Ts)
        updated_immstate = self.update(z, predicted_immstate, sensor_state)

        return updated_immstate

    def loglikelihood(self, z: np.ndarray, immstate: MixtureParameters, *, sensor_state: Dict[str, Any] = None,
    ) -> float:
        """
        A loglikelihhod function adapted to different input parameters, essentially same as:
            mode_cond_likelihood + np.log(predicted_mode_probs)
        :param z: Measurements [ ] one-dimensional array
        :param immstate: Mixture, with predicted mode probabilities as weights.
        :return: float. Total summed loglikelihood.
        """

        mode_cond_likelihoods = [self.mode_loglikelihood(z, filt, state, sensor_state=sensor_state)
                                 for filt, state in zip(self.filters, immstate.components)]

        assert len(mode_cond_likelihoods) == len(immstate.weights), \
            'Length of weighted likelihood is unlike length of mode conditional likelihoods'

        summed_ll = logsumexp(a=mode_cond_likelihoods, b=immstate.weights)

        return summed_ll

    def mode_loglikelihood( self, z: np.ndarray, filter: StateEstimator[MT], component: GaussParams,
                            sensor_state: Dict[str, Any] = None,) -> float:
        """ Return loglikelihood for specific mode, with its filter and components inputted"""
        mode_conditioned_ll = filter.loglikelihood(z, component, sensor_state=sensor_state)
        return mode_conditioned_ll

    def reduce_mixture(
        self, immstate_mixture: MixtureParameters[MixtureParameters[MT]]
    ) -> MixtureParameters[MT]:
        """
        Approximate a mixture of immstates as a single immstate.
        :param immstate_mixture: Association probabilities weighted mode mixtures.
        { weights: [], mixtureComponents}
        {  Pr{a_k | Z_1:k} ,    }
        :return: Reduced mixture
        """

        #  P(a_k = j | Z_1:k) -> Length m_k + 1
        weights = immstate_mixture.weights

        # P(s_k=i | a_k=j, Z_1:k) -> Length of # modes
        component_conditioned_mode_prob = np.array(
            [c.weights.ravel() for c in immstate_mixture.components]
        )
        # where inner gaussParams inside immstate_mixture.components.components
        # would be P(x_k | s_k=i, a_k=j, Z_1:k) for each mode ->

        # flip conditioning order with Bayes, sum over a_k j's, so we get right dimension out ( mode dimensions)
        # Prior: P(a_k = j | Z_1:k)     Conditional: P(s_k=i |a_k = j, Z_1:k)
        mode_prob, mode_conditioned_component_prob = discretebayes.discrete_bayes(
            weights, component_conditioned_mode_prob)
        # Returns ->
        # Marginal: P(s_k=i, Z_1:k)      Joint: P(a_k = j | s_k=i, Z_1:k)

        # We want P(x_k | s_k =i, Z_1:k) =  Joint * P( x_k | a_k = j, s_k = i, Z_1:k)
        # Where we know the last term is in immstate_mixture.components[j].components[i] for i modes, j associations
        mode_states = []

        # For mode_idx, filter
        for i, fs in zip(range(len(self.filters)), self.filters):

            # This is (3) from graded assignment, summed over all association hypothesis
            mode_conditional_density_components =  \
                [immstate_mixture.components[j].components[i] for j in range(len(weights))]

            # Reduce with mode_conditional_association weights ( components = association here )
            reduced_mode_conditional = fs.reduce_mixture(
                    MixtureParameters(
                        weights=mode_conditioned_component_prob[i],
                        components=mode_conditional_density_components))

            mode_states.append(reduced_mode_conditional)

        # Final reduction over modes
        immstate_reduced = MixtureParameters(mode_prob, mode_states)

        return immstate_reduced

    def estimate(self, immstate: MixtureParameters[MT]) -> GaussParams:
        """Calculate a state estimate with its covariance from immstate"""

        # ! You can assume all the modes have the same reduce and estimate function
        # ! and use eg. self.filters[0] functionality
        # ! assuming all the modes have the same reduce and estimate
        data_reduction = self.filters[0].reduce_mixture(immstate)
        est = self.filters[0].estimate(data_reduction)
        return est

    def gate(self, z: np.ndarray, immstate: MixtureParameters[MT], gate_size: float,
             sensor_state: Dict[str, Any] = None) -> bool:
        """
        Check if z is within the gate of any mode in immstate in sensor_state
        :param z: Measurements z, single array. [state_dim]
        :param gate_size: NOT SQUARED
        :param sensor_state:
        :return: bool
        """
        _, nises = self.NISes(z, immstate, sensor_state=sensor_state)
        gated = nises < gate_size ** 2
        return any(gated)

    def NIS(self, z: np.ndarray, immstate: MixtureParameters[MT], *, sensor_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """ Avg NIS over predicted state"""
        return self.NISes(z=z, immstate=immstate, sensor_state=sensor_state)[0]

    def NISes(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, np.ndarray]:
        """Calculate NIS per mode and the average NIS for all modes"""
        NISes = np.array(
            [
                fs.NIS(z, ms, sensor_state=sensor_state)
                for fs, ms in zip(self.filters, immstate.components)
            ]
        )

        innovs = [
            fs.innovation(z, ms, sensor_state=sensor_state)
            for fs, ms in zip(self.filters, immstate.components)
        ]

        v_ave = np.average([gp.mean for gp in innovs], axis=0, weights=immstate.weights)
        S_ave = np.average([gp.cov for gp in innovs], axis=0, weights=immstate.weights)

        NIS = (v_ave * np.linalg.solve(S_ave, v_ave)).sum()
        return NIS, NISes

    def NEESes(
        self,
        immstate: MixtureParameters,
        x_true: np.ndarray,
        *,
        idx: Optional[Sequence[int]] = None,
    ):
        NEESes = np.array(
            [
                fs.NEES(eststate=est_comp, x_true=x_true, idx=idx)
                for fs, est_comp in zip(self.filters, immstate.components)
            ]
        )
        est = self.estimate(immstate)

        NEES = self.filters[0].NEES(eststate=est, x_true=x_true, idx=idx)  # HACK?
        return NEES, NEESes

    # @singledispatchmethod todo set back
    def init_filter_state(
        self,
        init: MixtureParameters,  # Union[
        #     MixtureParameters, Dict[str, Any], Tuple[Sequence, Sequence], Sequence
        # ],
    ) -> MixtureParameters:
        """
        Initialize the imm state to MixtureParameters.

        - If mode probabilities are not found they are initialized from self.initial_mode_probabilities.
        - If only one mode state is found, it is broadcasted to all modes.

        MixtureParameters: goes unaltered
        dict:
            ["weights", "probs", "probabilities", "mode_probs"]
                in this order can signify mode probabilities
            ["components", "modes"] signify the modes
        tuple: first element is mode probabilities and second is mode states
        Sequence: assumed to be only the mode states

        mode probabilities: array_like
        components:

        """

        raise NotImplementedError(
            f"IMM do not know how to initialize a immstate from: {init}"
        )

    # @init_filter_state.register todo set back
    def _(self, init: MixtureParameters[MT]) -> MixtureParameters[MT]:
        return init

    # @init_filter_state.register(dict)
    def _(self, init: dict) -> MixtureParameters[MT]:
        # extract weights
        got_weights = False
        got_components = False
        for key in init:
            if not got_weights and key in [
                "weights",
                "probs",
                "probabilities",
                "mode_probs",
            ]:
                weights = np.asfarray([key])
                got_weights = True
            elif not got_components and key in ["components", "modes"]:
                components = self.init_components(init[key])
                got_components = True

        if not got_weights:
            weights = self.initial_mode_probabilities

        if not got_components:
            components = self.init_components(init)

        assert np.allclose(weights.sum(), 1), "Mode probabilities must sum to 1 for"

        return MixtureParameters(weights, components)

    # @init_filter_state.register(tuple)
    def _(self, init: tuple) -> MixtureParameters[MT]:
        assert isinstance(init[0], Sized) and len(init[0]) == len(
            self.filters
        ), f"To initialize from tuple the first element must be of len(self.filters)={len(self.filters)}"

        weights = np.asfarray(init[0])
        components = self.init_compontents(init[1])
        return MixtureParameters(weights, components)

    # @init_filter_state.register(Sequence)
    def _(self, init: Sequence) -> MixtureParameters[MT]:
        weights = self.initial_mode_probabilities
        components = self.init_components(init)
        return MixtureParameters(weights, components)

    # @singledispatchmethod
    def init_components(self, init: "Union[Iterable, MT_like]") -> List[MT]:
        """ Make an instance or Iterable of the Mode Parameters into a list of mode parameters"""
        return [fs.init_filter_state(init) for fs in self.filters]

    # @init_components.register(dict)
    def _(self, init: dict):
        return [fs.init_filter_state(init) for fs in self.filters]

    # @init_components.register(Iterable)
    def _(self, init: Iterable) -> List[MT]:
        if isinstance(init[0], (np.ndarray, list)):
            return [
                fs.init_filter_state(init_s) for fs, init_s in zip(self.filters, init)
            ]
        else:
            return [fs.init_filter_state(init) for fs in self.filters]

    def estimate_sequence(
        self,
        # A sequence of measurements
        Z: Sequence[np.ndarray],
        # the initial KF state to use for either prediction or update (see start_with_prediction)
        init_immstate: MixtureParameters,
        # Time difference between Z's. If start_with_prediction: also diff before the first Z
        Ts: Union[float, Sequence[float]],
        *,
        # An optional sequence of the sensor states for when Z was recorded
        sensor_state: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
        # sets if Ts should be used for predicting before the first measurement in Z
        start_with_prediction: bool = False,
    ) -> Tuple[List[MixtureParameters], List[MixtureParameters], List[GaussParams]]:
        """Create estimates for the whole time series of measurements. """
        k = len(Z)

        # Create and amend the sampling array
        ts_start_idx = 0 if start_with_prediction else 1
        ts_arr = np.empty(k)  # Init timestep array
        ts_arr[ts_start_idx:] = Ts

        # Insert a zero time prediction for no prediction equivalence
        if not start_with_prediction:
            ts_arr[0] = 0

        # Make sure the sensor_state_list actually is a sequence
        sensor_state_seq = sensor_state or [None] * k

        init_immstate = self.init_filter_state(init_immstate)

        immstate_upd = init_immstate

        immstate_pred_list = []
        immstate_upd_list = []
        estimates = []

        for z_k, Ts_k, ss_k in zip(Z, ts_arr, sensor_state_seq):
            immstate_pred = self.predict(immstate_upd, Ts_k)
            immstate_upd = self.update(z_k, immstate_pred, sensor_state=ss_k)

            immstate_pred_list.append(immstate_pred)
            immstate_upd_list.append(immstate_upd)
            estimates.append(self.estimate(immstate_upd))

        return immstate_pred_list, immstate_upd_list, estimates
