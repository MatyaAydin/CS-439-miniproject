from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable

def scaling_selection(g, loss_at_params, params,
                      sigma, key, constant_learning_rate=True, use_H_hat=False, H_hat=None):
    """Choose a scaled descent direction p = -s·g according to curvature."""
    # Hessian–vector product  H g
    if not use_H_hat:
        Hg = jax.jvp(lambda p: jax.grad(loss_at_params)(p, key),
                    (params,), (g,))[1]
    else:
        Hg =  (1 / H_hat) * g # Use diagonal only
    dot_product = jnp.dot(g, Hg)
    norm_g      = jnp.linalg.norm(g)

    # ------------------------------------------------------------
    # step-size bounds for low-positive-curvature (LPC) regime
    # ------------------------------------------------------------
    if constant_learning_rate:
        s_lpc_min = 1.0 / sigma
        s_lpc_max = 1.0 / sigma
    else:
        s_lpc_min = (1.0 / sigma) * jr.uniform(key, shape=())

    # three classical step candidates
    print(norm_g, dot_product)
    s_CG = norm_g**2 / dot_product
    s_MR = dot_product / jnp.linalg.norm(Hg)**2
    s_GM = jnp.sqrt(s_CG * s_MR)

    # ------------------------------------------------------------
    # regime selection
    # ------------------------------------------------------------
    if dot_product > sigma * norm_g**2:                #   Strong +ve curv.
        s_choice = jr.choice(key, a=jnp.array([s_CG, s_MR, s_GM]))
        # print(jnp.array([s_CG, s_MR, s_GM]))
        print("SPC")
        return -s_choice * g, "SPC"

    elif 0.0 < dot_product < sigma * norm_g**2:        #   Weak +ve curv.
        slpc = jr.uniform(key, shape=(), minval=s_lpc_min, maxval=1.0 / sigma)
        print("LPC")
        return -slpc * g, "LPC"

    else:                                              #   Non-convex region
        snc  = jr.uniform(key, shape=(), minval=s_lpc_min, maxval=s_lpc_max)
        print("NC")
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
    H_hat: jnp.ndarray
    g_k: jnp.ndarray
    sigma: float = 0.1
    rho: float = 0.25
    theta_bt: float = 0.5
    theta_fb: float = 0.5
    use_H_hat: bool = False
    beta: float = 0.9
    iteration: int = 0

def mrcg_step(state: MRCGState) -> MRCGState:
    grad = jax.grad(state.loss_at_params)(state.params, state.key)
    state.g_k = state.beta * state.g_k + (1 - state.beta) * grad
    grad = state.g_k # use momentum
    key, subkey = jr.split(state.key)
    p, flag = scaling_selection(grad, state.loss_at_params, state.params, state.sigma, subkey, constant_learning_rate=True, use_H_hat=state.use_H_hat, H_hat=state.H_hat)
    
    if flag == "SPC" or flag == "LPC":
        alpha = backtracking_LS(state.loss_at_params, state.key, state.theta_bt, state.rho, state.params, grad, p)
    else:
        alpha = forward_backward_LS(state.loss_at_params, state.key, state.theta_fb, state.rho, state.params, grad, p)

    new_params = state.params + alpha * p
    return MRCGState(params=new_params, loss_at_params=state.loss_at_params,
                    key=key, H_hat= 1 * state.H_hat + (1 - 0) * grad,
                    g_k = state.g_k,	
                    sigma=state.sigma, rho=state.rho,
                    theta_bt=state.theta_bt,
                    theta_fb=state.theta_fb,
                    use_H_hat=state.use_H_hat, iteration=state.iteration + 1)