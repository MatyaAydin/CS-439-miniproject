from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable

def scaling_selection(g, H, sigma, key, constant_learning_rate=True):
    
    Hg =jnp.dot(H, g)
    dot_product =jnp.dot(g, Hg)
    norm_g =jnp.linalg.norm(g)

    if constant_learning_rate:
        s_lpc_min = 1 / sigma #set to 1/sigma for constant learning rate
        s_lpc_max = 1 / sigma #set to 1/sigma for constant learning rate
    else:
        s_lpc_min = 1 / sigma *jr.random(key)

    s_CG =jnp.linalg.norm(g)**2 / dot_product
    s_MR = dot_product /jnp.linalg.norm(Hg)**2
    s_GM =jnp.sqrt(s_CG * s_MR)

    if dot_product > sigma * norm_g**2:
        spc =jr.choice(key, a=jnp.array([s_CG, s_MR, s_GM]))
        return -spc*g, "SPC"
    elif 0 < dot_product and dot_product < sigma * norm_g**2:
        slpc =jr.uniform(key, s_lpc_min, 1 / sigma)
        return -slpc * g, "LPC"
    else:
        snc =jr.uniform(key, s_lpc_min, s_lpc_max)
        return -snc * g, "NC"

def backtracking_LS(loss_at_params, theta, rho, x, g, p):

    alpha = 1.0
    while loss_at_params(x + alpha * p) > loss_at_params(x) + alpha * rho *jnp.dot(g, p):
        alpha *= theta


    return alpha

# algorithm 4 forward/backward tracking line search

def forward_backward_LS(loss_at_params, theta, rho, x, g, p):
    alpha = 1.0
    if loss_at_params(x + alpha * p) > loss_at_params(x) + alpha * rho *jnp.dot(g, p):
        backtracking_LS(loss_at_params, theta, rho, x, g, p)
    else:
        while loss_at_params(x + alpha * p) >= loss_at_params(x) + alpha * rho *jnp.dot(g, p):
            alpha /= theta

    return alpha * theta

# algorithm 2: scaled gradient descent with line search


def scaled_GD(loss_at_params, x0, sigma, rho, theta_bt, theta_fb, MAX_ITER, eps):
    """
    sigma <<< 1
    0 < theta < 1
    0 < rho < 1/2
    """

    x_k = x0
    flag_distribution = {"SPC": 0, "LPC": 0, "NC": 0}


    for _ in range(MAX_ITER):

        g_k = 2 * x_k

        if jnp.linalg.norm(g_k) < eps:
            break
        

        p_k, FLAG = scaling_selection(g_k,jnp.eye(len(x_k)), sigma)
        flag_distribution[FLAG] += 1

        if FLAG == "SPC" or FLAG == "LPC":
            alpha_k = backtracking_LS(loss_at_params, theta_bt, rho, x_k, g_k, p_k)

        else:
            alpha_k = forward_backward_LS(loss_at_params, theta_fb, rho, x_k, g_k, p_k)

        x_k += alpha_k * p_k

    return x_k, flag_distribution


#algorithm 3 backward tracking line search

def backtracking_LS(loss_at_params, theta, rho, x, g, p):

    alpha = 1.0
    while loss_at_params(x + alpha * p) > loss_at_params(x) + alpha * rho *jnp.dot(g, p):
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
    grad = jax.grad(state.loss_at_params)(state.params)
    hess = jax.hessian(state.loss_at_params)(state.params)
    key, subkey = jr.split(state.key)
    p, flag = scaling_selection(grad, hess, state.sigma, subkey)
    
    if flag == "SPC" or flag == "LPC":
        alpha = backtracking_LS(state.loss_at_params, state.theta_bt, state.rho, state.params, grad, p)
    else:
        alpha = forward_backward_LS(state.loss_at_params, state.theta_fb, state.rho, state.params, grad, p)

    new_params = state.params + alpha * p
    return MRCGState(new_params, state.loss_at_params, key, state.sigma, state.rho, state.theta_bt, state.theta_fb, state.iteration + 1)