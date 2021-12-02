import jax
import jax.numpy as jnp


@jax.jit
def linear(X, w, b):
    return jnp.dot(X, w) + b


@jax.jit
def relu(X, w, b):
    return jnp.squeeze(jnp.maximum(linear(X, w, b), 0))


@jax.jit
def lrelu(X, w, b):
    return jnp.squeeze(jax.nn.leaky_relu(linear(X, w, b)))


@jax.jit
def tanh(X, w, b):
    return jnp.squeeze(jax.nn.tanh(linear(X, w, b)))


@jax.jit
def sigmoid(X, w, b):
    return jnp.squeeze(jax.nn.sigmoid(linear(X, w, b)))


@jax.jit
def hard_sigmoid(X, w, b):
    return jnp.squeeze(jax.nn.hard_sigmoid(linear(X, w, b)))


@jax.jit
def softplus(X, w, b):
    return jnp.squeeze(jax.nn.softplus(linear(X, w, b)))


@jax.jit
def softmax(X, w, b):
    return jnp.squeeze(jax.nn.softmax(linear(X, w, b)))