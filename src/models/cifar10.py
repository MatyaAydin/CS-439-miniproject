# models/cifar10.py
# ===============================================================
# One–layer soft-max (logistic regression) on CIFAR-10 + ℓ2 penalty
# ===============================================================
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jax.example_libraries import stax
from jax.flatten_util import ravel_pytree
import models.datasets as datasets         # same helper you used for MNIST

# 1) -----------------------------------------------------------------
#  data  (images to float32 in [0,1], flatten to 3072 = 32*32*3)
# --------------------------------------------------------------------
train_imgs, train_lbls, test_imgs, test_lbls = datasets.cifar10()
train_imgs = train_imgs.reshape(len(train_imgs), -1) / 255.0
test_imgs  = test_imgs.reshape(len(test_imgs),  -1) / 255.0

num_features = 32 * 32 * 3
num_classes  = 10

# 2) -----------------------------------------------------------------
#  model = single Dense → LogSoftmax
# --------------------------------------------------------------------
init_fn, apply_fn = stax.serial(
    stax.Dense(num_classes), stax.LogSoftmax
)
_, params_tree      = init_fn(jr.PRNGKey(0), (-1, num_features))
flat_params, unflat = ravel_pytree(params_tree)

# 3) -----------------------------------------------------------------
#  helpers
# --------------------------------------------------------------------
l2_strength = 5e-4          # µ in the paper (adjust in grid search)

@jit
def _forward(params, x):
    return apply_fn(unflat(params), x)      # logits already in log-space

@jit
def loss(params, batch):
    x, y = batch
    logp  = _forward(params, x)
    ce    = -jnp.mean(jnp.sum(logp * y, axis=1))
    l2    = 0.5 * l2_strength * jnp.sum(params ** 2)
    return ce + l2

@jit
def accuracy(params, batch):
    x, y   = batch
    preds  = jnp.argmax(_forward(params, x), axis=1)
    target = jnp.argmax(y, axis=1)
    return jnp.mean(preds == target)

@jit
def predict(params, x):
    return _forward(params, x)

# 4) -----------------------------------------------------------------
#  mini-batch utility for SGD-like methods
# --------------------------------------------------------------------
BATCH = 256
def random_batch(key):
    idx = jr.permutation(key, train_imgs.shape[0])[:BATCH]
    return train_imgs[idx], train_lbls[idx]

def loss_at_params(params, key):           # “oracle” call
    return loss(params, random_batch(key))

train_accuracy  = jit(lambda p: accuracy(p, (train_imgs, train_lbls)))
test_accuracy   = jit(lambda p: accuracy(p, (test_imgs,  test_lbls)))
