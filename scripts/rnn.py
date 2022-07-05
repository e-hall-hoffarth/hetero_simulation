import jax
import jax.numpy as jnp
import haiku as nn
import haiku as hk
import optax as tx

import pandas as pd
import numpy as np
import time
import pickle

SCALE = 1e-2
INIT_KEYS = jax.random.split(jax.random.PRNGKey(1), 8)
KEY = jax.random.PRNGKey(42)
TRAIN_NEW = True

BETA = 0.8
MAX_AGE = 2.
MIN_DP = 0.2
INTEREST = 0.5

R = (1 + INTEREST)
B = BETA
BR = B * R

HIDDEN_STATE_SIZE = 8
GRID_SIZE = 100
X = jnp.linspace(1e-5, 1, GRID_SIZE)
STATE_SPACE = jnp.concatenate((
    jnp.concatenate((X.reshape(-1, 1), jnp.ones((GRID_SIZE, 1))), axis=1),
    jnp.concatenate((X.reshape(-1, 1), jnp.zeros((GRID_SIZE, 1))), axis=1)),
axis=0)


class Attention_Layer(nn.Module):
    def __call__(self):
        return jnp.sum(jax.nn.softmax(nn.Linear(1)(self.ys)).reshape(-1, 1) * self.ys, axis=0)

    def __init__(self, ys, name=None):
        super().__init__(name=name)
        self.ys = ys


class Blue_GRU(nn.Module):
    def __call__(self, ys, carry):
        d0 = carry[:(2 * self.hidden_state_size)]
        u0 = carry[(2 * self.hidden_state_size):]
        attention_layer = Attention_Layer(ys)

        d1, u1 = nn.GRU(2 * self.hidden_state_size)(d0, jnp.concatenate((attention_layer(), u0)))
        u1 = u1[:(self.hidden_state_size)]
        return jnp.concatenate((d1, u1)), u1

    def __init__(self, state_size, name=None):
        super().__init__(name=name)
        self.hidden_state_size = state_size


class Custom_RNN(hk.Module):
    def __init__(self, name=None, hidden_state_size=2):
        super().__init__(name=name)
        self.hidden_state_size = hidden_state_size
        self.input_layer = hk.Linear(self.hidden_state_size)
        self.yellow_functions = [lambda x: hk.GRU(self.hidden_state_size)(x[0], x[1]) for i in
                                 range(STATE_SPACE.shape[0])]
        self.yellow_switch_function = lambda carry, inputs: hk.switch(inputs[0], self.yellow_functions,
                                                                      (inputs[1], carry))
        self.blue_functions = [lambda x: Blue_GRU(self.hidden_state_size)(x[0], x[1]) for i in
                               range(STATE_SPACE.shape[0])]
        self.blue_switch_function = lambda carry, inputs: hk.switch(inputs[0], self.blue_functions, (inputs[1], carry))
        self.us_flatten = hk.Linear(1)
        self.v_out = hk.Linear(STATE_SPACE.shape[0])
        self.index = jnp.arange(len(self.yellow_functions))
        assert self.index.shape[0] == STATE_SPACE.shape[0]

    def __call__(self, state_space, inputs):
        init_state = self.input_layer(inputs)

        hn, ys = hk.scan(self.yellow_switch_function, init_state, (self.index, state_space))
        dn, us = hk.scan(self.blue_switch_function, jnp.concatenate((hn, jnp.zeros(2 * self.hidden_state_size))),
                         (self.index, jnp.repeat(ys[jnp.newaxis, ...], state_space.shape[0], axis=0)))

        us_flat = jnp.squeeze(self.us_flatten(us))
        t = jnp.where(inputs == 1., size=1)[0][0]
        X = state_space[:, 0]
        o = state_space[:, 1]

        cu = X * jax.nn.sigmoid(us_flat / 1e3)  # (n,) vector
        c = hk.cond((t >= MAX_AGE), lambda _: X, lambda _: cu, 0.)
        # c = jax.lax.select((t >= MAX_AGE), X, cu)

        o1 = jnp.minimum(jnp.int32(X - c >= MIN_DP) + o, 1.)

        vu = self.v_out(jnp.concatenate((us_flat, o1)))
        v = hk.cond((t > MAX_AGE), lambda _: jnp.zeros(state_space.shape[0]), lambda _: vu, 0.)
        # v = jax.lax.select((t > MAX_AGE), jnp.zeros(state_space.shape[0]), vu)

        predictions = jnp.concatenate((v.reshape(-1, 1), c.reshape(-1, 1)), axis=1)

        return predictions


init, apply = hk.without_apply_rng(
    hk.transform(lambda state_space, inputs: Custom_RNN(hidden_state_size=HIDDEN_STATE_SIZE)(state_space, inputs)))
INIT_PARAMS = init(KEY, STATE_SPACE, jnp.zeros(4))
MODEL = jax.jit(apply)


@jax.jit
def increment_t(t):
    i = jnp.where(t == 1, size=1)[0]
    return t.at[i].set(0).at[i + 1].set(1)


@jax.jit
def neural_network(params, model_state, inputs):
    preds = MODEL(params, model_state, inputs)
    mat = jnp.concatenate((model_state, preds), axis=1)  # For a given t, we have a tensor of (X, o, v, c)
    # So we just need to add o1 and x1 columns to this tensor
    o1 = jnp.minimum(jnp.int32(mat[:, 0] - mat[:, 3] >= MIN_DP) + mat[:, 1], 1.)
    x1 = (1 + o1 * INTEREST) * (mat[:, 0] - mat[:, 3])
    preds = jnp.concatenate((mat[:, 2:], x1.reshape(-1, 1), o1.reshape(-1, 1)), axis=1)
    return preds


@jax.jit
def loss(params, model_state, t0):
    t1 = increment_t(t0)
    preds0 = neural_network(params, model_state, t0)
    predicted_state = preds0[:, 2:]
    preds1 = neural_network(params, predicted_state, t1)

    v0 = preds0[:, 0]
    c0 = preds0[:, 1]
    o0 = model_state[:, 1]
    v1 = preds1[:, 0]
    c1 = preds1[:, 1]

    vf = lambda model_state, t: neural_network(params, model_state, t)[:, 0]
    cf = lambda model_state, t: neural_network(params, model_state, t)[:, 1]

    c0x = jnp.diag(jax.jacfwd(cf)(model_state, t0)[..., 0])
    v0x = jnp.diag(jax.jacfwd(vf)(model_state, t0)[..., 0])
    v1x = jnp.diag(jax.jacfwd(vf)(preds0[:, 2:], t1)[..., 0])

    live_next = (jnp.where(t0 == 1, size=1)[0][0] < MAX_AGE)
    loss_euler = jax.lax.select(live_next, o0 * (c1 - BR * c0), jnp.zeros(STATE_SPACE.shape[0]))
    loss_bellman = ((jnp.log(c0) + BETA * v1) - v0)
    loss_foc = jax.lax.select(live_next, BR * v1x * c0 - 1, jnp.zeros(STATE_SPACE.shape[0]))
    loss_envelope = (c0x - c0 * v0x)

    return jnp.concatenate(
        (loss_euler.reshape(-1, 1), loss_bellman.reshape(-1, 1), loss_foc.reshape(-1, 1), loss_envelope.reshape(-1, 1)),
        axis=1)


@jax.jit
def batch_loss(params, model_state, batch, W):
    losses = jax.vmap(loss, in_axes=(None, None, 0))(params, model_state, batch)
    # losses = jnp.array([loss(params, model_state, x) for x in batch]).reshape(-1, 4)
    mean_losses = jnp.squeeze(jnp.mean(losses ** 2, axis=0))
    return jnp.sum(W * mean_losses), tuple((mean_losses[0], mean_losses[1], mean_losses[2], mean_losses[3]))


@jax.tree_util.Partial(jax.jit, static_argnums=(3,))
def step(params, model_state, opt_state, optimizer, batch, W):
    (loss, (loss_euler, loss_bellman, loss_foc, loss_envelope)), grad = jax.value_and_grad(batch_loss, has_aux=True)(
        params, model_state, batch, W)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = tx.apply_updates(params, updates)
    loss_components = tuple((loss_euler, loss_bellman, loss_foc, loss_envelope))
    return params, opt_state, loss, loss_components


def training_loop(grid, params, model_state, opt_state, optimizer, W=jnp.ones(4), max_iter=1000):
    j = 0
    while j < max_iter:
        params, opt_state, loss, loss_components = step(params, model_state, opt_state, optimizer, grid, W)

        if j == 0:
            start_time = time.time()

        if j % 1 == 0:
            print(
                f'Iteration: {j}\tCurrent Loss: {loss:.2f}\tAverage samples/s: {((grid.shape[0] * j) / (time.time() - start_time)):.2f}\n'
                f'Loss Euler: {loss_components[0]}\nLoss Bellman: {loss_components[1]}\nLoss FoC: {loss_components[2]}\nLoss Envelope: {loss_components[3]}')
        j += 1

    return params, opt_state, model_state


def main():
    optimizer = tx.adam(learning_rate=1e-2)

    if TRAIN_NEW:
        model_state = STATE_SPACE
        params = INIT_PARAMS
        opt_state = optimizer.init(INIT_PARAMS)
    else:
        params, opt_state, model_state, MODEL = pickle.load(open('rnn.pkl', 'rb'))

    W = jnp.array([1e1, 1e-1, 1., 1.])
    GRID = jnp.concatenate((jnp.eye(3), jnp.zeros(3).reshape(-1, 1)), axis=1)  # 3 different ages

    params, opt_state, model_state = training_loop(GRID, params, model_state, opt_state, optimizer, W=W, max_iter=2000)
    pickle.dump(tuple((params, opt_state, model_state, MODEL)), open('rnn.pkl', 'wb'))


if __name__ == '__main__':
    main()