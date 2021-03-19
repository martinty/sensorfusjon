from typing import Tuple

import numpy as np


def discrete_bayes(
    # the prior: shape=(n,) -> Mixture weights
    pr: np.ndarray,
    # the conditional/likelihood: shape=(n, m) -> Transition matrix
    cond_pr: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the new marginal and conditional: shapes=((m,), (m, n))
    """Swap which discrete variable is the marginal and conditional."""
    # (n, m)
    joint = pr[:, None] * cond_pr  # prior weights elementwise multiplied with transition -> mu_k-1 * pi

    marginal = np.sum(joint, axis=0)  # Pr{Sk | z_k-1} summed over Sk_1, axis = 0

    # Take care of rare cases of degenerate zero marginal,
    conditional = joint / marginal if np.nonzero(marginal) else joint / marginal + 10e-8

    # flip axes?? (n, m) -> (m, n) -> Expected output
    conditional = conditional.T

    # optional DEBUG
    assert np.all(
        np.isfinite(conditional)
    ), f"NaN or inf in conditional in discrete bayes"
    assert np.all(
        np.less_equal(0, conditional)
    ), f"Negative values for conditional in discrete bayes"
    assert np.all(
        np.less_equal(conditional, 1)
    ), f"Value more than on in discrete bayes"

    assert np.all(np.isfinite(marginal)), f"NaN or inf in marginal in discrete bayes"

    return marginal, conditional
