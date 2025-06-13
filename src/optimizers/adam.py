# ------------------------------------------------------------
# Adam optimiser — JAX implementation
# ------------------------------------------------------------
"""adam.py
Standard Adam optimizer (Kingma & Ba, 2015) implemented in JAX.
This variant *does not* apply decoupled weight‑decay (à la AdamW).
Public API:
    adam_init(params, loss_at_params, lr, key, *, beta1, beta2, eps)
    adam_step(state) -> new_state
The `loss_at_params` callback must accept a parameter vector and a
`jax.random.PRNGKey`, returning a scalar loss.
"""

from dataclasses import dataclass
from collections.abc import Callable
import jax
import jax.numpy as jnp
import jax.random as jr


@dataclass
class AdamState:
    """Container for the optimiser state at step *t*."""

    params: jnp.ndarray                                       # θₜ
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey],       # f(θ, key)
                             jnp.ndarray]
    lr: float                                                 # learning‑rate α
    key: jr.PRNGKey                                           # PRNG key
    m: jnp.ndarray                                            # 1st‑moment (β₁)
    v: jnp.ndarray                                            # 2nd‑moment (β₂)
    iteration: int                                            # time‑step t
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


# ---------- helper to start the optimiser ----------

def adam_init(
    initial_params: jnp.ndarray,
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey], jnp.ndarray],
    lr: float,
    key: jr.PRNGKey,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> AdamState:
    """Initialise Adam optimiser state."""
    zeros = jnp.zeros_like(initial_params)
    return AdamState(
        params=initial_params,
        loss_at_params=loss_at_params,
        lr=lr,
        key=key,
        m=zeros,
        v=zeros,
        iteration=0,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )


# ---------- single optimisation step ----------

def adam_step(state: AdamState) -> AdamState:
    """One Adam update (in‑place bias‑corrected)."""

    # fresh RNG for this gradient
    key, subkey = jr.split(state.key)

    # gradient of loss w.r.t. current parameters
    g = jax.grad(state.loss_at_params)(state.params, subkey)

    # update biased moments
    m = state.beta1 * state.m + (1.0 - state.beta1) * g
    v = state.beta2 * state.v + (1.0 - state.beta2) * (g * g)

    # bias‑correction
    t = state.iteration + 1
    m_hat = m / (1.0 - state.beta1 ** t)
    v_hat = v / (1.0 - state.beta2 ** t)

    # parameter update
    step = state.lr * m_hat / (jnp.sqrt(v_hat) + state.eps)
    new_params = state.params - step

    # pack next state
    return AdamState(
        params=new_params,
        loss_at_params=state.loss_at_params,
        lr=state.lr,
        key=key,
        m=m,
        v=v,
        iteration=t,
        beta1=state.beta1,
        beta2=state.beta2,
        eps=state.eps,
    )
