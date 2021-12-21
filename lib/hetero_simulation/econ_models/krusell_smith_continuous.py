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
n = 1600
mb = 256
n_epoch = 30000
n_iter = n // mb
n_forward = 10

k = 5 # number of agents (~ size of state space)
m = 64 # embedding dimension
nn_shapes = jnp.array([m, m, m, m])


@jax.jit
def neural_network(params, X, E, Z, e, x):
    X_tilde = jnp.concatenate([X.reshape(1, -1),
                               E.reshape(1, -1),
                               Z.reshape(1, -1),
                               e.reshape(1, -1),
                               x.reshape(1, -1)], axis=1)
    l1 = sigmoid(X_tilde, params['w0'], params['b0'])
    l2 = sigmoid(l1, params['w1'], params['b1'])
    # l3 = tanh(l2, params['w2'], params['b2'])
    # l4 = tanh(jnp.concatenate((l3, e[..., jnp.newaxis], x[..., jnp.newaxis])), params['w3'], params['b3'])
    return jnp.array([jnp.squeeze(x * sigmoid(l2, params['cwf'], params['cbf'])),
                      jnp.squeeze(exp(l2, params['lwf'], params['lbf']))])


@jax.jit
def fischer_burmeister(a, b):
    return a + b - jnp.sqrt(jnp.power(a, 2) + jnp.power(b, 2))


@jax.jit
def prices(config, X, Z, E):
    sumk = jnp.sum(X)
    sumexpl = jnp.sum(jnp.exp(E))
    w = (1 - config['alpha']) * jnp.exp(Z) * jnp.power(sumk, config['alpha']) * jnp.power(sumexpl, -1 * config['alpha'])
    r = 1 - config['delta'] + config['alpha'] * jnp.exp(Z) * jnp.power(sumk, config['alpha'] - 1) * jnp.power(sumexpl, 1 - config['alpha'])
    return r, w


@jax.jit
def loss(params, config, Xs, Zs, Es, key):
    Z1s, E1s = next_state(Zs, Es, config, key)
    Rs, Ws = jax.vmap(lambda X, Z, E: prices(config, X, Z, E))(Xs, Zs, Es)
    ws = jax.vmap(lambda X, E, R, W: jax.vmap(lambda x, e: (R * x) + (W * jnp.exp(e)))(X, E))(Xs, Es, Rs, Ws)
    outputs = jax.vmap(
        lambda X, Z, E, w: jax.vmap(lambda i: neural_network(params, X, E, Z, E[i], w[i]))(jnp.arange(k)))(Xs, Zs, Es, ws)
    cs = outputs[..., 0]
    lms = outputs[..., 1]
    c_rels = cs / ws
    X1s = ws - cs
    R1s, W1s = jax.vmap(lambda X, Z, E: prices(config, X, Z, E))(X1s, Z1s, E1s)
    w1s = jax.vmap(lambda X, E, R, W: jax.vmap(lambda x, e: (R * x) + (W * jnp.exp(e)))(X, E))(X1s, E1s, R1s, W1s)
    c1s = jax.vmap(
        lambda X, Z, E, w: jax.vmap(lambda i: neural_network(params, X, E, Z, E[i], w[i])[0])(jnp.arange(k)))(X1s, Z1s,
                                                                                                              E1s, w1s)

    u = lambda c: log_utility()(c)
    gs = jax.vmap(lambda R, cs: jax.vmap(lambda c: config['beta'] * R * jax.grad(u)(c))(cs))(R1s, c1s)
    ups = jax.grad(u)(cs)
    g_diff = jax.vmap(lambda g, up, lm: (g / up) - lm)(gs.reshape(-1, 1), ups.reshape(-1, 1), lms.reshape(-1, 1))

    lm_diff = jax.vmap(lambda c, lm: fischer_burmeister(1 - c, 1 - lm))(c_rels.reshape(-1, 1), lms.reshape(-1, 1))

    return g_diff, lm_diff, c_rels, X1s, Z1s, E1s


@jax.jit
def batch_loss(params, config, Xs, Zs, Es, keys):
    g_diff_1, lm_diff_1, c_rels, X1s, Z1s, E1s = loss(params, config, Xs, Zs, Es, keys[0])
    g_diff_2, lm_diff_2, c_rels, X1s, Z1s, E1s = loss(params, config, Xs, Zs, Es, keys[1])
    g2 = g_diff_1 * g_diff_2
    lm2 = lm_diff_1 * lm_diff_2

    return jnp.squeeze(jnp.mean(g2 + lm2, axis=0)), (jnp.squeeze(jnp.mean(g2, axis=0)), jnp.squeeze(jnp.mean(lm2, axis=0)), c_rels, Z1s, E1s, X1s)


@jax.jit
def next_state(Zs, Es, config, key):
    Zs_prime = config['rho_z'] * Zs + config['sigma_z'] * jax.random.normal(key, shape=(Zs.shape[0],))
    Es_prime = config['rho_e'] * Es + config['sigma_e'] * jax.random.normal(key, shape=(Es.shape[0], k))
    return Zs_prime, Es_prime


def simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward):
    for _ in range(n_forward):
        keys = jax.random.split(key, 2)
        Zs, Es, Xs = batch_loss(params, config, Xs, Zs, Es, keys)[1][-3:]
        key = keys[-1]
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
w0f = scale * jnp.ones(nn_shapes[1]).reshape(nn_shapes[1], 1)
b00 = scale * jnp.ones(nn_shapes[0]).reshape(1, nn_shapes[0])
b01 = scale * jnp.ones(nn_shapes[1]).reshape(1, nn_shapes[1])
b02 = scale * jnp.ones(nn_shapes[2]).reshape(1, nn_shapes[2])
b03 = scale * jnp.ones(nn_shapes[3]).reshape(1, nn_shapes[3])
b0f = scale * jnp.ones(1).reshape(1, 1)

params0 = {
    'w0': w00, 'w1': w01, 'w2': w02, 'w3': w03, 'cwf': w0f, 'lwf': w0f, 'b0': b00, 'b1': b01, 'b2': b02, 'b3': b03, 'cbf': b0f, 'lbf': b0f
}


def training_loop(opt_state, tol=1e-10, max_iter=10 ** 4):
    j = 0
    key = jax.random.PRNGKey(np.random.randint(1, int(1e8)))
    val_loss = jnp.inf
    grad = {'0': jnp.inf}
    opt_init, opt_update, get_params = adam(step_size=0.001)
    params = get_params(opt_state)

    Xs, Zs, Es, key = generate_random_state(params, config, key, n_forward=n_forward)

    while j < max_iter and max([jnp.max(jnp.abs(v)) for k, v in grad.items()]) > tol and jnp.abs(val_loss) > tol:
        jj = 0
        while jj < n_iter:
            keys = keys = jax.random.split(key, 2)
            key = keys[-1]
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