import jax
import jax.numpy as jnp


@jax.jit
def linear(X, w, b):
    return jnp.squeeze(jnp.dot(X, w) + b)


@jax.jit
def relu(X, w, b):
    return jnp.squeeze(jnp.maximum(linear(X, w, b), 0))


@jax.jit
def lrelu(X, w, b):
    return jnp.squeeze(jax.nn.leaky_relu(linear(X, w, b)))


@jax.jit
def elu(X, w, b):
    return jnp.squeeze(jax.nn.elu(linear(X, w, b)))


@jax.jit
def tanh(X, w, b):
    return jnp.squeeze(jax.nn.tanh(linear(X, w, b)))


@jax.jit
def sigmoid(X, w, b):
    return jnp.squeeze(jax.nn.sigmoid(linear(X, w, b)))


@jax.jit
def hard_sigmoid(X, w, b):
    return jnp.squeeze(jnp.minimum(jnp.maximum(linear(X, w, b), 0), 1))


@jax.jit
def interior_sigmoid(X, w, b):
    return (1e-5 / 2) + (1 - 1e-5) * sigmoid(X, w, b)


@jax.jit
def scaled_sigmoid(X, w, b, scale=100.):
    return jnp.squeeze(jax.nn.sigmoid(linear(X, w, b) / scale))


@jax.jit
def softplus(X, w, b):
    return jnp.squeeze(jax.nn.softplus(linear(X, w, b)))


@jax.jit
def softmax(X, w, b):
    return jnp.squeeze(jax.nn.softmax(linear(X, w, b)))


@jax.jit
def exp(X, w, b):
    return jnp.squeeze(jnp.exp(linear(X, w, b)))


@jax.jit
def log(X, w, b):
    return jnp.squeeze(jnp.log(linear(X, w, b)))


@jax.jit
def sqrt(X, w, b):
    return jnp.squeeze(jnp.sqrt(softplus(X, w, b)))


@jax.jit
def custom_value_fn(X, w, b):
    x = jnp.squeeze(linear(X, w, b)) / 100.
    return ((jnp.int32(x > 1) * (jnp.log(jnp.abs(x)) + 1)) +
           (jnp.int32(x < -1) * (-1 * (1/2)*jnp.power(x, 2) - (1/2))) +
           (jnp.int32(x <= 1) * jnp.int32(x >= -1) * x))


@jax.jit
def softmax(X, w, b):
    return jnp.squeeze(jax.nn.softmax(linear(X, w, b)))


@jax.jit
def fischer_burmeister(a, b):
    return a + b - jnp.sqrt(jnp.power(a, 2) + jnp.power(b, 2))