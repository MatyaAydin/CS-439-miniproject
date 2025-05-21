# ------------------------------------------------------------
# AdamW optimiser — JAX implementation
# ------------------------------------------------------------
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable

@dataclass
class AdamWState:
    params: jnp.ndarray                                      # θₜ
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey],      # f(θ, key)
                             jnp.ndarray]
    lr: float                                                # learning-rate α
    key: jr.PRNGKey                                          # PRNG key
    m: jnp.ndarray                                           # 1st-moment  (β₁)
    v: jnp.ndarray                                           # 2nd-moment  (β₂)
    iteration: int                                           # time-step t
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-2                               # λ

# ---------- helper to start the optimiser ----------
def adamw_init(initial_params: jnp.ndarray,
               loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey], jnp.ndarray],
               lr: float,
               key: jr.PRNGKey,
               *,
               beta1: float = 0.9,
               beta2: float = 0.999,
               eps: float = 1e-8,
               weight_decay: float = 1e-2) -> AdamWState:
    """Build the initial AdamW optimiser state."""
    zeros = jnp.zeros_like(initial_params)
    return AdamWState(
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
        weight_decay=weight_decay,
    )

# ---------- single optimisation step ----------
def adamw_step(state: AdamWState) -> AdamWState:
    """One AdamW update."""
    # split RNG so each gradient uses a fresh sub-key
    key, subkey = jr.split(state.key)

    # gradient of loss w.r.t. current parameters
    g = jax.grad(state.loss_at_params)(state.params, subkey)

    # update biased moments
    m = state.beta1 * state.m + (1.0 - state.beta1) * g
    v = state.beta2 * state.v + (1.0 - state.beta2) * (g * g)

    # bias-correction
    t = state.iteration + 1
    m_hat = m / (1.0 - state.beta1 ** t)
    v_hat = v / (1.0 - state.beta2 ** t)

    # parameter update (decoupled weight decay à la AdamW)
    step = state.lr * m_hat / (jnp.sqrt(v_hat) + state.eps)
    new_params = state.params - step - state.lr * state.weight_decay * state.params

    # package next state
    return AdamWState(
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
        weight_decay=state.weight_decay,
    )