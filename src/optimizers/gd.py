from dataclasses import dataclass
import jax
import jax.numpy as jnp
from collections.abc import Callable  # Modern import

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