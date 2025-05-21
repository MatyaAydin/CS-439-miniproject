# ============================================================
# line_search_gd.py  –  Backtracking / Armijo line-search GD
# ============================================================
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable


# ---------- tiny utility: Armijo backtracking ---------------
def _backtracking_ls(
    f: Callable[[jnp.ndarray, jr.PRNGKey], jnp.ndarray],
    x: jnp.ndarray,
    g: jnp.ndarray,
    p: jnp.ndarray,
    key: jr.PRNGKey,
    *,
    rho: float,
    theta: float,
) -> float:
    """Return step-length α that satisfies Armijo (sufficient-decrease)."""
    alpha = 1.0
    fx = f(x, key)
    dot_gp = jnp.dot(g, p)
    while f(x + alpha * p, key) > fx + alpha * rho * dot_gp:
        alpha *= theta                      # shrink
    return alpha


# ---------- optimiser state --------------------------------
@dataclass
class LSState:
    params: jnp.ndarray                                       # θₜ
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey],       # f(θ, key)
                             jnp.ndarray]
    key: jr.PRNGKey                                           # PRNG
    iteration: int                                            # t
    rho: float = 0.25                                         # Armijo const.
    theta: float = 0.5                                        # shrink factor
    weight_decay: float = 0.0                                 # L2 (decoupled)
    verbose: bool = False


# ---------- builder ----------------------------------------
def ls_init(
    initial_params: jnp.ndarray,
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey], jnp.ndarray],
    key: jr.PRNGKey,
    *,
    rho: float = 0.25,
    theta: float = 0.5,
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> LSState:
    return LSState(
        params=initial_params,
        loss_at_params=loss_at_params,
        key=key,
        iteration=0,
        rho=rho,
        theta=theta,
        weight_decay=weight_decay,
        verbose=verbose,
    )


# ---------- one optimisation step ---------------------------
def ls_step(state: LSState) -> LSState:
    key, subkey = jr.split(state.key)

    # Gradient
    g = jax.grad(state.loss_at_params)(state.params, subkey)
    if state.weight_decay != 0.0:
        g = g + state.weight_decay * state.params

    p = -g                                           # steepest descent dir.

    # Backtracking line-search
    alpha = _backtracking_ls(
        state.loss_at_params,
        state.params,
        g,
        p,
        subkey,
        rho=state.rho,
        theta=state.theta,
    )

    new_params = state.params + alpha * p

    if state.verbose:
        loss_val = state.loss_at_params(new_params, key)
        print(f"Iter {state.iteration+1:2d}  α={alpha:.2e}  "
              f"Loss={loss_val:.6f}  Params={new_params}")

    return LSState(
        params=new_params,
        loss_at_params=state.loss_at_params,
        key=key,
        iteration=state.iteration + 1,
        rho=state.rho,
        theta=state.theta,
        weight_decay=state.weight_decay,
        verbose=state.verbose,
    )
