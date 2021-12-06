import numpy as np
import pickle
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adam, unpack_optimizer_state, pack_optimizer_state
from hetero_simulation.archive.agent import log_utility
from hetero_simulation.ml.utils import *

# Parameters of wealth distribution
a = 0.5
b = 0.2

# Preference and production parameters
alpha = 0.36
beta = 0.96
delta = 0.025
rho_z = 0.95
rho_e = 0.9
sigma_z = 0.01
sigma_e = 0.2 * jnp.sqrt(1 - rho_e**2)

config = {
    'alpha': alpha, 'beta': beta, 'delta': delta,
    'sigma_z': sigma_z, 'sigma_e': sigma_e, 'rho_z': rho_z, 'rho_e': rho_e
}

# ML parameters
n = 100
mb = 32
n_epoch = 30000
n_iter = n // mb
n_forward = 10

k = 5 # number of agents (~ size of state space)
m = 32 # embedding dimension
config['p0'] = 1. # weight on envelope condition
config['p1'] = 1. # weight on k-t conditions
config['p2'] = 1. # weight on prediction error
nn_shapes = jnp.array([m, 2 * m, 2 * m, m])


@jax.jit
def opt_control(params, X, E, Z, e, x):
    X_tilde = jnp.concatenate([X.reshape(1, -1),
                               E.reshape(1, -1),
                               Z.reshape(1, -1),
                               e.reshape(1, -1),
                               x.reshape(1, -1)], axis=1)
    l1 = tanh(X_tilde, params['cw0'], params['cb0'])
    l2 = tanh(l1, params['cw1'], params['cb1'])
    l3 = tanh(l2, params['cw2'], params['cb2'])
    l4 = tanh(jnp.concatenate((l3, e[..., jnp.newaxis], x[..., jnp.newaxis])), params['cw3'], params['cb3'])
    return jnp.squeeze(x * ((tanh(l4, params['cwf'], params['cbf']) + 1) / 2))


@jax.jit
def lagrange_multiplier(params, X, E, Z, e, x):
    X_tilde = jnp.concatenate([X.reshape(1, -1),
                               E.reshape(1, -1),
                               Z.reshape(1, -1),
                               e.reshape(1, -1),
                               x.reshape(1, -1)], axis=1)
    l1 = tanh(X_tilde, params['lw0'], params['lb0'])
    l2 = tanh(l1, params['lw1'], params['lb1'])
    l3 = tanh(l2, params['lw2'], params['lb2'])
    l4 = tanh(jnp.concatenate((l3, e[..., jnp.newaxis], x[..., jnp.newaxis])), params['lw3'], params['lb3'])
    return jnp.squeeze(softplus(l4, params['lwf'], params['lbf']))


@jax.jit
def fischer_burmeister(a, b):
    return a + b - jnp.sqrt(jnp.power(a, 2) + jnp.power(b, 2))


# Continuous exo states
@jax.jit
def prices(config, X, Z, E):
    sumk = jnp.sum(X)
    sumexpl = jnp.sum(jnp.exp(E))
    w = (1 - alpha) * jnp.exp(Z) * jnp.power(sumk, config['alpha'])
    r = 1 - config['delta'] + config['alpha'] * jnp.exp(Z) * jnp.power(sumk, alpha - 1) * sumexpl
    return r, w


@jax.jit
def batch_loss(params, config, Xs, Zs, Es, keys):
    n = Xs.shape[0]
    Xs = jnp.concatenate((Xs, Xs), axis=0)
    Zs = jnp.concatenate((Zs, Zs), axis=0)
    Es = jnp.concatenate((Es, Es), axis=0)

    Z1s, E1s = next_state(Zs, Es, config, keys)
    rs, ws = jax.vmap(lambda X, Z, E: prices(config, X, Z, E))(Xs, Zs, Es)
    xs = jax.vmap(lambda X, E, rt, wt: jax.vmap(lambda x, e: (rt * x) + (wt * jnp.exp(e)))(X, E))(Xs, Es, rs, ws).reshape(Xs.shape[0], -1)
    cs = jax.vmap(lambda X, Z, E: jax.vmap(lambda i: opt_control(params, X, E, Z, E[i], X[i]))(jnp.arange(k)))(xs, Zs, Es)
    c_rels = cs / xs
    lms = jax.vmap(lambda X, Z, E: jax.vmap(lambda i: lagrange_multiplier(params, X, E, Z, E[i], X[i]))(jnp.arange(k)))(xs, Zs, Es)
    X1s = xs - cs
    r1s, w1s = jax.vmap(lambda X, Z, E: prices(config, X, Z, E))(X1s, Z1s, E1s)
    x1s = jax.vmap(lambda X, E, rt, wt: jax.vmap(lambda x, e: (rt * x) + (wt * jnp.exp(e)))(X, E))(X1s, E1s, r1s, w1s).reshape(Xs.shape[0], -1)
    c1s = jax.vmap(lambda X, Z, E: jax.vmap(lambda i: opt_control(params, X, E, Z, E[i], X[i]))(jnp.arange(k)))(x1s, Z1s, E1s)

    u = lambda c: log_utility()(c)
    gs = jax.vmap(lambda r, cs: jax.vmap(lambda c: config['beta'] * r * jax.grad(u)(c))(cs))(r1s, c1s)
    ups = jax.grad(u)(cs)

    # g_diff = ((gs / ups) - lms)
    g_diff = jax.vmap(lambda g, up, lm: (g / up) - lm)(gs, ups, lms)
    # g_diff = gs - ups
    g2 = jnp.mean(jax.vmap(lambda x, y: jnp.sum(x * y))(g_diff[:n], g_diff[n:]))

    kt_cond = jnp.mean(jax.vmap(lambda c, lm: fischer_burmeister(1-c, 1-lm)**2)(c_rels.reshape(-1, 1), lms.reshape(-1, 1)))

    return config['p0'] * g2 + config['p1'] * kt_cond, (g2, kt_cond, c_rels, Z1s[:n], E1s[:n], X1s[:n])
    # return g2, (g2, c_rels, Z1s[:n], E1s[:n], X1s[:n])


@jax.jit
def next_state(Zs, Es, config, keys):
    Zs_prime = jax.vmap(lambda z, k: config['rho_z'] * z + config['sigma_z'] * jax.random.normal(k))(Zs, keys[:, 0])
    Es_prime = jax.vmap(lambda e, k: config['rho_e'] * e + config['sigma_e'] * jax.random.normal(k))(Es.reshape(-1), keys[:, 1:].reshape(-1, 2)).reshape(Es.shape)
    return Zs_prime, Es_prime


def simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward):
    for _ in range(n_forward):
        keys = jax.random.split(key, 2 * (k + 1)).reshape(2, (k + 1), 2)
        keys = jnp.repeat(keys, n // 2, axis=0)
        Zs, Es, Xs = batch_loss(params, config, Xs, Zs, Es, keys)[1][-3:]
        key = keys[-1, -1]
    return Xs, Zs, Es, key


def generate_random_state(params, config, key, n_forward=0):
    Zs = jnp.zeros(shape=(n // 2,))
    Es = jnp.zeros(shape=(n // 2, k))
    Xs = jnp.exp(a * jax.random.normal(jax.random.PRNGKey(np.random.randint(1, int(1e8))), shape=(n // 2, k))) + b
    if n_forward > 0:
        Xs, Zs, Es, key = simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward)

    return Xs, Zs, Es, key


scale = 0.05
# theta0 = jax.random.gamma(jax.random.PRNGKey(1), scale, shape=(k, m))
w00 = scale * jnp.ones(nn_shapes[0] * (2 * k + 3)).reshape(2 * k + 3, nn_shapes[0])
w01 = scale * jnp.ones(nn_shapes[0] * nn_shapes[1]).reshape(nn_shapes[0], nn_shapes[1])
w02 = scale * jnp.ones(nn_shapes[1] * nn_shapes[2]).reshape(nn_shapes[1], nn_shapes[2])
w03 = scale * jnp.ones((nn_shapes[2] + 2) * nn_shapes[3]).reshape(nn_shapes[2] + 2, nn_shapes[3])
w0f = scale * jnp.ones(nn_shapes[3]).reshape(nn_shapes[3], 1)
b00 = scale * jnp.ones(nn_shapes[0]).reshape(1, nn_shapes[0])
b01 = scale * jnp.ones(nn_shapes[1]).reshape(1, nn_shapes[1])
b02 = scale * jnp.ones(nn_shapes[2]).reshape(1, nn_shapes[2])
b03 = scale * jnp.ones(nn_shapes[3]).reshape(1, nn_shapes[3])
b0f = scale * jnp.ones(1).reshape(1, 1)

c_params0 = {
    'cw0': w00, 'cw1': w01, 'cw2': w02, 'cw3': w03, 'cwf': w0f, 'cb0': b00, 'cb1': b01, 'cb2': b02, 'cb3': b03, 'cbf': b0f
}

l_params0 = {
    'lw0': w00, 'lw1': w01, 'lw2': w02, 'lw3': w03, 'lwf': w0f, 'lb0': b00, 'lb1': b01, 'lb2': b02, 'lb3': b03, 'lbf': b0f
}

params0 = {
    **c_params0, **l_params0
}


def training_loop(opt_state, tol=1e-10, max_iter=10 ** 4):
    j = 0
    key = jax.random.PRNGKey(np.random.randint(1, int(1e8)))
    val_loss = jnp.inf
    grad = {'0': jnp.inf}
    opt_init, opt_update, get_params = adam(step_size=0.01)
    params = get_params(opt_state)

    Xs, Zs, Es, key = generate_random_state(params, config, key, n_forward=n_forward)

    while j < max_iter and max([jnp.max(jnp.abs(v)) for k, v in grad.items()]) > tol and jnp.abs(val_loss) > tol:
        jj = 0
        while jj < n_iter:
            keys = jax.random.split(key, 2 * (k + 1)).reshape(2, k + 1, 2)
            keys = jnp.repeat(keys, mb // 2, axis=0)
            key = keys[-1, -1]
            params = get_params(opt_state)

            sample = jax.random.choice(jax.random.PRNGKey(np.random.randint(1, int(1e8))), jnp.arange(n // 2), shape=(mb // 2,))
            val, grad = jax.value_and_grad(batch_loss, has_aux=True)(params, config, Xs[sample], Zs[sample], Es[sample], keys)
            val_loss = jnp.abs(val[0])
            c_star_rel = val[1][-4]
            assert (c_star_rel < 1).all()
            if jnp.isnan(val_loss):
                raise ValueError('Loss is nan')

            c_val = jnp.abs(val[1][0])
            kt_val = jnp.abs(val[1][1])

            # v_opt_state = opt_update(j * n_iter + jj, v_grad, v_opt_state)
            # c_opt_state = opt_update(j * n_iter + jj, c_grad, c_opt_state)
            # x_opt_state = opt_update(j * n_iter + jj, x_grad, x_opt_state)
            opt_state = opt_update(j * n_iter + jj, grad, opt_state)

            jj += 1

        # Start from a new random position, before moving into the current implied ergodic set
        params = get_params(opt_state)
        if n_forward > 0:
            Xs, Zs, Es, key = simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward)
            # Xs, Zs, Es, key = generate_random_state(params, config, key, n_forward=n_forward)

        if j % 100 == 0:
            trained_params = unpack_optimizer_state(opt_state)
            pickle.dump(trained_params, open(f'./share/models/ks_model_{k}_{j}.pkl', 'wb'))
            # pickle.dump(params, open(f'./share/models/ks_model_{k}_{j}.pkl', 'wb'))
            print(f'Iteration: {j}\tTotal Loss: {val_loss}\tC Loss: {c_val}\tKT Loss: {kt_val}' +\
                  f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in grad.items()])}' +\
                  f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in params.items()])}')
        j += 1

        # update weights
        # config['p0'] = jnp.float32(10 ** (- jnp.round(jnp.log10(jnp.abs(c_val)))))
        # config['p1'] = jnp.float32(10 ** (- jnp.round(jnp.log10(jnp.abs(kt_val)))))
        # config['p2'] = jnp.float32(10 ** (- jnp.round(jnp.log10(jnp.abs(x_val)))))


    print(f'Terminating training with final statistics:\n' + \
          f'Iteration: {j}\tTotal Loss: {val_loss}\tC Loss: {c_val}\tKT Loss: {kt_val}' + \
          f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in grad.items()])}' + \
          f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in params.items()])}')
    return opt_state


def main():
    opt_init, opt_update, get_params = adam(step_size=0.001)
    # saved_params = pickle.load(open(f'./share/models/ks_model_{k}_final.pkl', 'rb'))
    # opt_state = pack_optimizer_state(saved_params)
    opt_state = opt_init(params0)

    opt_state = training_loop(opt_state, max_iter=n_epoch)
    # params = training_loop(max_iter=n_epoch)
    params = unpack_optimizer_state(opt_state)
    pickle.dump(params, open(f'./share/models/ks_model_{k}_final.pkl', 'wb'))


if __name__ == '__main__':
    main()