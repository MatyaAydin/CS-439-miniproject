import time
import itertools

import numpy.random as npr

import jax.numpy as jnp
import jax.random as jr
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
import jax_examples.datasets as datasets
from tqdm import tqdm

from jax.flatten_util import ravel_pytree


@jit
def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

@jit
def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)

@jit
def predict(params, inputs):
   return predict_tree_params(unflatten_params(params), inputs)

init_random_params, predict_tree_params = stax.serial(
    # Dense(512), Relu,
    Dense(30), Relu,
    Dense(10), LogSoftmax)
_, initial_params_tree = init_random_params(jr.PRNGKey(0), (-1, 28*28))
flat_params, unflatten_params = ravel_pytree(initial_params_tree)



BATCH_SIZE = 128
train_images, train_labels, test_images, test_labels = datasets.mnist()
# print("train_images.shape", train_images.shape)
# print("train_labels.shape", train_labels.shape)
# print("test_images.shape", test_images.shape)
# print("test_labels.shape", test_labels.shape)

def random_batch(key):
  num_train = train_images.shape[0]
  perm = jr.permutation(key, num_train)
  batch_idx = perm[:BATCH_SIZE]
  return train_images[batch_idx], train_labels[batch_idx]

def loss_at_params(params, key):
  return loss(params, random_batch(key))

@jit
def train_accuracy(params):
    return accuracy(params, (train_images, train_labels))

@jit
def test_accuracy(params):
    return accuracy(params, (test_images, test_labels))