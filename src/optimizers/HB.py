# ------------------------------------------------------------
# HB.py  – Heavy-Ball / Polyak-momentum optimiser (JAX)
# ------------------------------------------------------------
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable


@dataclass
class HBState:
    params: jnp.ndarray                                       # θₜ
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey],       # f(θ, key)
                             jnp.ndarray]
    lr: float                                                 # learning-rate α
    key: jr.PRNGKey                                           # PRNG key
    v: jnp.ndarray                                            # velocity term
    iteration: int                                            # time-step t
    beta: float = 0.9                                         # momentum coeff.
    weight_decay: float = 0.0                                 # L2 (decoupled)
    verbose: bool = False


# ---------- helper to build the initial state ----------
def hb_init(initial_params: jnp.ndarray,
            loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey], jnp.ndarray],
            lr: float,
            key: jr.PRNGKey,
            *,
            beta: float = 0.9,
            weight_decay: float = 0.0,
            verbose: bool = False) -> HBState:
    """Initialise Heavy-Ball optimiser."""
    zeros = jnp.zeros_like(initial_params)
    return HBState(
        params=initial_params,
        loss_at_params=loss_at_params,
        lr=lr,
        key=key,
        v=zeros,                   # v₀ = 0
        iteration=0,
        beta=beta,
        weight_decay=weight_decay,
        verbose=verbose,
    )


# ---------- single optimisation step ----------
def hb_step(state: HBState) -> HBState:
    """One Heavy-Ball update."""
    # fresh RNG for this gradient
    key, subkey = jr.split(state.key)

    # gradient of loss w.r.t. current parameters
    g = jax.grad(state.loss_at_params)(state.params, subkey)

    # decoupled weight-decay (optional, same spirit as AdamW)
    if state.weight_decay != 0.0:
        g = g + state.weight_decay * state.params

    # update velocity and parameters
    v = state.beta * state.v + g
    new_params = state.params - state.lr * v

    # optional progress print
    if state.verbose:
        loss_val = state.loss_at_params(new_params, key)
        print(f"Iteration {state.iteration+1:2d}: "
              f"Loss = {loss_val:.6f}  Params = {new_params}")

    # pack next state
    return HBState(
        params=new_params,
        loss_at_params = state.loss_at_params,
        lr  = state.lr,
        key = key,
        v   = v,
        iteration = state.iteration + 1,
        beta = state.beta,
        weight_decay = state.weight_decay,
        verbose = state.verbose,
    )
