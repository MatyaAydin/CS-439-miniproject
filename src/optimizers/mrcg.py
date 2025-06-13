from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable

def scaling_selection(g, loss_at_params, params,
                      sigma, key, constant_learning_rate=True, use_H_hat=False, H_hat=None, hv_fun=None):
    """Choose a scaled descent direction p = -s·g according to curvature."""

    if not use_H_hat:
        Hg = hv_fun(params, g) if hv_fun is not None else \
            jax.jvp(lambda p: jax.grad(loss_at_params)(p, key),
                    (params,), (g,))[1]
    else:
        Hg = hv_fun(g, H_hat)

    dot_product = jnp.dot(g, Hg)
    norm_g      = jnp.linalg.norm(g)

    # ------------------------------------------------------------
    # step-size bounds for low-positive-curvature (LPC) regime
    # ------------------------------------------------------------
    if sigma == 0.0:
    # dummy positive numbers; never used when σ = 0
        s_lpc_min = s_lpc_max = 1.0
    else:
        if constant_learning_rate:
            s_lpc_min = s_lpc_max = 1.0 / sigma
        else:
            s_lpc_min = (1.0 / sigma) * jr.uniform(key, shape=())
    # three classical step candidates
    s_CG = norm_g**2 / dot_product
    s_MR = dot_product / jnp.linalg.norm(Hg)**2
    s_GM = jnp.sqrt(s_CG * s_MR)

    # ------------------------------------------------------------
    # regime selection
    # ------------------------------------------------------------
    if dot_product > sigma * norm_g**2:                #   Strong +ve curv.
        s_choice = jr.choice(key, a=jnp.array([s_CG, s_MR, s_GM]))
        return -s_choice * g, "SPC"

    elif 0.0 < dot_product < sigma * norm_g**2:        #   Weak +ve curv.
        slpc = jr.uniform(key, shape=(), minval=s_lpc_min, maxval=1.0 / sigma)
        return -slpc * g, "LPC"

    else:                                              #   Non-convex region
        snc  = jr.uniform(key, shape=(), minval=s_lpc_min, maxval=s_lpc_max)
        return -snc * g, "NC"



def backtracking_LS(loss_at_params, key, theta, rho, x, g, p):

    alpha = 1.0
    while loss_at_params(x + alpha * p, key) > loss_at_params(x, key) + alpha * rho *jnp.dot(g, p):
        alpha *= theta


    return alpha

# algorithm 4 forward/backward tracking line search

def forward_backward_LS(loss_at_params, key, theta, rho, x, g, p):
    alpha = 1.0
    if loss_at_params(x + alpha * p, key) > loss_at_params(x, key) + alpha * rho *jnp.dot(g, p):
        backtracking_LS(loss_at_params, key, theta, rho, x, g, p)
    else:
        while loss_at_params(x + alpha * p, key) >= loss_at_params(x, key) + alpha * rho *jnp.dot(g, p):
            alpha /= theta

    return alpha * theta



def backtracking_LS(loss_at_params, key, theta, rho, x, g, p):

    alpha = 1.0
    while loss_at_params(x + alpha * p, key) > loss_at_params(x, key) + alpha * rho *jnp.dot(g, p):
        alpha *= theta


    return alpha




@dataclass
class MRCGState:
    params: jnp.ndarray
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    key: jr.PRNGKey
    sigma: float = 0.1
    rho: float = 0.25
    theta_bt: float = 0.5
    theta_fb: float = 0.5
    iteration: int = 0

def mrcg_step(state: MRCGState) -> MRCGState:
    grad = jax.grad(state.loss_at_params)(state.params, state.key)
    key, subkey = jr.split(state.key)
    p, flag = scaling_selection(grad, state.loss_at_params, state.params, state.sigma, subkey)
    
    if flag == "SPC" or flag == "LPC":
        alpha = backtracking_LS(state.loss_at_params, state.key, state.theta_bt, state.rho, state.params, grad, p)
    else:
        alpha = forward_backward_LS(state.loss_at_params, state.key, state.theta_fb, state.rho, state.params, grad, p)

    new_params = state.params + alpha * p
    return MRCGState(new_params, state.loss_at_params, key, state.sigma, state.rho, state.theta_bt, state.theta_fb, state.iteration + 1)