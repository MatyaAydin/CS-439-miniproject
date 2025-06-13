

import jax.numpy as jnp
import jax.random as jr

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load and preprocess the dataset
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)  # Make it a column vector

# Normalize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)
TOTAL_SAMPLES = X.shape[0]

# Convert to JAX arrays
x = jnp.array(X)
y_true = jnp.array(y)

# Initial parameters for each feature
initial_params = jnp.ones(x.shape[1])  # One param per feature

def batched_loss_at_params(batch_size):
    # Model: simple linear regression
    def model(params, x):
        return jnp.dot(x, params)

    # Loss: mean squared error

    # Loss function with parameters
    def loss_at_params(params, key):
        batch = jr.choice(key, jnp.arange(x.shape[0]), shape=(batch_size,), replace=False)
        x_batch = x[batch]
        y_batch = y_true[batch]
        def loss(y_pred, y_true):
            return jnp.mean((y_pred - y_true.squeeze()) ** 2)

        y_pred = model(params, x_batch)
        return loss(y_pred, y_batch)
    return loss_at_params