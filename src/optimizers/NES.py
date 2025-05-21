# ============================================================
# nes.py  –  Simple Natural-Evolution Strategies (Gaussian μ)
# ============================================================
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable


@dataclass
class NESState:
    params: jnp.ndarray                                       # μₜ (search mean)
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey],       # fitness fn
                             jnp.ndarray]
    key: jr.PRNGKey
    lr: float                                                 # η
    sigma: float                                              # std of search dist
    iteration: int
    pop_size: int                                             # λ
    verbose: bool = False
    baseline: float | None = None                             # running mean fitness


def nes_init(
    initial_params: jnp.ndarray,
    loss_at_params: Callable[[jnp.ndarray, jr.PRNGKey], jnp.ndarray],
    lr: float,
    key: jr.PRNGKey,
    *,
    sigma: float = 0.1,
    pop_size: int = 32,
    verbose: bool = False,
) -> NESState:
    return NESState(
        params=initial_params,
        loss_at_params=loss_at_params,
        key=key,
        lr=lr,
        sigma=sigma,
        iteration=0,
        pop_size=pop_size,
        verbose=verbose,
        baseline=None,
    )


def nes_step(state: NESState) -> NESState:
    """Follow the natural gradient of E[f(θ)] where θ~N(μ, σ²I)."""
    key, sample_key = jr.split(state.key)

    # 1) Sample a population of noise vectors ε_i
    eps = jr.normal(sample_key, shape=(state.pop_size, *state.params.shape))

    # 2) Evaluate fitness for each perturbed candidate
    def _eval(sample_eps):
        candidate = state.params + state.sigma * sample_eps
        return state.loss_at_params(candidate, key)           # same RNG for fairness

    fitness = jax.vmap(_eval)(eps)                            # shape [pop_size]

    # 3) Optional baseline to reduce variance
    baseline = jnp.mean(fitness) if state.baseline is None else 0.9 * state.baseline + 0.1 * jnp.mean(fitness)
    adj_fitness = fitness - baseline

    # 4) Monte-Carlo estimate of ∇_μ E[f]
    grad_mu = jnp.mean(adj_fitness[:, None] * eps, axis=0) / state.sigma

    # 5) Update mean with learning-rate
    new_params = state.params - state.lr * grad_mu

    if state.verbose:
        print(f"Iter {state.iteration+1:2d}  "
              f"Mean fitness={jnp.mean(fitness):.6f}  μ={new_params}")

    return NESState(
        params=new_params,
        loss_at_params=state.loss_at_params,
        key=key,
        lr=state.lr,
        sigma=state.sigma,
        iteration=state.iteration + 1,
        pop_size=state.pop_size,
        verbose=state.verbose,
        baseline=baseline,
    )
