from dataclasses import dataclass
import jax
import jax.numpy as jnp
from collections.abc import Callable  # Modern import

import jax.random as jr

@dataclass
class GradientDescentState:
    params: jnp.ndarray
    loss_at_params: Callable[[jnp.ndarray], jnp.ndarray]
    step_size: float
    iteration: int = 0

def gradient_descent_step(state: GradientDescentState) -> GradientDescentState:
    grad = jax.grad(state.loss_at_params)(state.params)
    new_params = state.params - state.step_size * grad
    
    return GradientDescentState(new_params, state.loss_at_params, state.step_size, state.iteration + 1)


@dataclass
class SGDState:
    params: jnp.ndarray
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey], jnp.ndarray]
    step_size: float
    key: jr.PRNGKey
    iteration: int = 0

def sgd_step(state: SGDState) -> SGDState:
    key, subkey = jr.split(state.key)
    loss_fn = lambda p: state.loss_at_params(p, subkey)
    grad = jax.grad(loss_fn)(state.params)
    new_params = state.params - state.step_size * grad

    return SGDState(
        params=new_params,
        loss_at_params=state.loss_at_params,
        step_size=state.step_size,
        key=key,
        iteration=state.iteration + 1
    )
