import jax.numpy as jnp

def rosenbrock_loss(params):
    x, y = params
    return (1 - x)**2 + 100 * (y - x**2)**2

def beale_loss(params):
    x, y = params
    term1 = (1.5 - x + x * y)**2
    term2 = (2.25 - x + x * y**2)**2
    term3 = (2.625 - x + x * y**3)**2
    return term1 + term2 + term3


