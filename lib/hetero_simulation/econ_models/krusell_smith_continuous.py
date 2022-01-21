import numpy as np
import pickle
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adam, unpack_optimizer_state, pack_optimizer_state
from hetero_simulation.archive.agent import log_utility
from hetero_simulation.ml.utils import *

jax.config.update('jax_platform_name', 'cpu')

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
n = 128
mb = 16
n_epoch = 10000
n_iter = 2 * (n // mb)
n_forward = 100

k = 5 # number of agents (~ size of state space)
m = 4 # embedding dimension
nn_shapes = jnp.array([m, m, m, m])


@jax.jit
def neural_network(params, X, E, Z, e, x):
    X_tilde = jnp.concatenate([X.reshape(1, -1),
                               E.reshape(1, -1),
                               Z.reshape(1, -1),
                               e.reshape(1, -1),
                               x.reshape(1, -1)], axis=1)
    X_tilde = X_tilde @ params['theta']
    # l1 = tanh(X_tilde, params['w0'], params['b0'])
    l2 = tanh(X_tilde, params['w1'], params['b1'])
    l3 = tanh(l2, params['w2'], params['b2'])
    # l4 = tanh(jnp.concatenate((l3, e[..., jnp.newaxis], x[..., jnp.newaxis])), params['w3'], params['b3'])
    return jnp.array([jnp.squeeze(x * sigmoid(l3, params['cwf'], params['cbf'])),
                      jnp.squeeze(exp(l3, params['lwf'], params['lbf']))])


@jax.jit
def fischer_burmeister(a, b):
    return a + b - jnp.sqrt(jnp.power(a, 2) + jnp.power(b, 2))


@jax.jit
def next_X(params, config, X, Z, E):
    R, W = prices(config, X, Z, E)
    w = jax.vmap(lambda x, e: (R * x) + (W * jnp.exp(e)))(X, E)
    c = jax.vmap(lambda i: neural_network(params, X, E, Z, E[i], w[i])[0])(jnp.arange(k))
    return w - c


@jax.jit
def prices(config, X, Z, E):
    sumk = jnp.sum(X)
    sumexpl = jnp.sum(jnp.exp(E))
    w = (1 - config['alpha']) * jnp.exp(Z) * jnp.power(sumk, config['alpha']) * jnp.power(sumexpl, -1 * config['alpha'])
    r = 1 - config['delta'] + config['alpha'] * jnp.exp(Z) * jnp.power(sumk, config['alpha'] - 1) * jnp.power(sumexpl, 1 - config['alpha'])
    return r, w


@jax.jit
def loss(params, config, X, Z, E, key):
    Z1, E1 = next_state(Z, E, config, key)
    R, W = prices(config, X, Z, E)
    w = jax.vmap(lambda x, e: (R * x) + (W * jnp.exp(e)))(X, E)
    outputs = jax.vmap(lambda i: neural_network(params, X, E, Z, E[i], w[i]))(jnp.arange(k))
    c = outputs[..., 0]
    lm = outputs[..., 1]
    c_rel = c / w
    X1 = w - c
    R1, W1 = prices(config, X1, Z1, E1)
    w1 = jax.vmap(lambda x, e: (R * x) + (W * jnp.exp(e)))(X1, E1)
    c1 = jax.vmap(lambda i: neural_network(params, X1, E1, Z1, E1[i], w1[i])[0])(jnp.arange(k))

    u = lambda c: log_utility()(c)
    g = jax.vmap(lambda c: config['beta'] * R1 * jax.grad(u)(c))(c1)
    up = jax.grad(u)(c)
    g_diff = jax.vmap(lambda g, up, lm: (g / up) - lm)(g.reshape(-1, 1), up.reshape(-1, 1), lm.reshape(-1, 1))
    lm_diff = jax.vmap(lambda c, lm: fischer_burmeister(1 - c, 1 - lm))(c_rel.reshape(-1, 1), lm.reshape(-1, 1))

    return g_diff, lm_diff, c_rel, X1, Z1, E1


@jax.jit
def batch_loss(params, config, Xs, Zs, Es, keys):
    g_diff_1, lm_diff_1, c_rels, X1s, Z1s, E1s = jax.vmap(loss, in_axes=(None, None, 0, 0, 0, None))(params, config, Xs, Zs, Es, keys[0])
    g_diff_2, lm_diff_2, c_rels, X1s, Z1s, E1s = jax.vmap(loss, in_axes=(None, None, 0, 0, 0, None))(params, config, Xs, Zs, Es, keys[1])
    g2 = g_diff_1 * g_diff_2
    lm2 = lm_diff_1 * lm_diff_2

    return jnp.squeeze(jnp.mean(g2 + lm2)), (jnp.squeeze(jnp.mean(g2)), jnp.squeeze(jnp.mean(lm2)), c_rels, Z1s, E1s, X1s)


@jax.jit
def next_state(Z, E, config, key):
    Z_prime = config['rho_z'] * Z + config['sigma_z'] * jax.random.normal(key,)
    E_prime = config['rho_e'] * E + config['sigma_e'] * jax.random.normal(key, shape=(k,))
    return Z_prime, E_prime


def simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward):
    keys = jax.random.split(key, Zs.shape[0] * n_forward).reshape(n_forward, Zs.shape[0], 2)
    for i in range(n_forward):
        Xs = jax.vmap(next_X, in_axes=(None, None, 0, 0, 0))(params, config, Xs, Zs, Es)
        Zs, Es = jax.vmap(next_state, in_axes=(0, 0, None, 0))(Zs, Es, config, keys[i])
    return Xs, Zs, Es, keys[-1, -1]


def generate_random_state(params, config, key, n_forward=0):
    Zs = jnp.zeros(shape=(n // 2,))
    Es = jnp.zeros(shape=(n // 2, k))
    Xs = jnp.exp(a * jax.random.normal(jax.random.PRNGKey(np.random.randint(1, int(1e8))), shape=(n // 2, k))) + b
    if n_forward > 0:
        Xs, Zs, Es, key = simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward)

    return Xs, Zs, Es, key


scale = 0.5
init_keys = jax.random.split(jax.random.PRNGKey(5), 11)
theta0 = jax.random.gamma(init_keys[0], scale, shape=(2 * k + 3, nn_shapes[0]))
w00 = scale * jax.random.normal(init_keys[1], shape=(2 * k + 3, nn_shapes[0]))
w01 = scale * jax.random.normal(init_keys[2], shape=(nn_shapes[0], nn_shapes[1]))
w02 = scale * jax.random.normal(init_keys[3], shape=(nn_shapes[1], nn_shapes[2]))
w03 = scale * jax.random.normal(init_keys[4], shape=(nn_shapes[2] + 2, nn_shapes[3]))
w0f = scale * jax.random.normal(init_keys[5], shape=(nn_shapes[1], 1))
b00 = scale * jax.random.normal(init_keys[6], shape=(1, nn_shapes[0]))
b01 = scale * jax.random.normal(init_keys[7], shape=(1, nn_shapes[1]))
b02 = scale * jax.random.normal(init_keys[8], shape=(1, nn_shapes[2]))
b03 = scale * jax.random.normal(init_keys[9], shape=(1, nn_shapes[3]))
b0f = scale * jax.random.normal(init_keys[10], shape=(1, 1))

params0 = {
    'theta': theta0, 'w0': w00, 'w1': w01, 'w2': w02, 'w3': w03, 'cwf': w0f, 'lwf': w0f, 'b0': b00, 'b1': b01, 'b2': b02, 'b3': b03, 'cbf': b0f, 'lbf': b0f
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
            keys = jax.random.split(key, 2)
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
            pickle.dump(trained_params, open(f'./share/models/ks_cont_model_{k}_{j}.pkl', 'wb'))
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
    pickle.dump(params, open(f'./share/models/ks_cont_model_{k}_final.pkl', 'wb'))


if __name__ == '__main__':
    main()