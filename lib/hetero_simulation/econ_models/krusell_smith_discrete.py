import numpy as np
import pickle
import jax
import jax.numpy as jnp
import time
from jax.example_libraries.optimizers import adamax, unpack_optimizer_state, pack_optimizer_state
from hetero_simulation.archive.agent import log_utility, ces_utility
from hetero_simulation.ml.utils import *

jax.config.update('jax_platform_name', 'cpu')

# Parameters of wealth distribution
a = 0.5
b = 0.2

# Preference and production parameters
k = 5 # number of agents (~ size of state space)
alpha = 0.36
beta = 0.96
delta = 0.025
rho_z = 0.95
rho_e = 0.9
sigma_z = 0.01
sigma_e = 0.2 * jnp.sqrt(1 - rho_e**2)
discrete = True
prefs = 2 + 0.5 * jax.random.normal(jax.random.PRNGKey(6), shape=(k,))

config = {
    'alpha': alpha, 'beta': beta, 'delta': delta,
    'sigma_z': sigma_z, 'sigma_e': sigma_e, 'rho_z': rho_z, 'rho_e': rho_e, 'discrete': discrete,
    'prefs': prefs
}

# Transition parameters
z_g_ave_dur = 8
z_b_ave_dur = 8
u_g_ave_dur = 1.5
u_b_ave_dur = 2.5
puu_rel_gb2bb = 1.25
puu_rel_bg2gg = 0.75

z_g = 1.01
z_b = 0.99
u_g = 0.04
u_b = 0.1

# ML parameters
n = 128
mb = 16
n_epoch = 30000
n_iter = 2 * (n // mb)
n_forward = 100

m = 64 # embedding dimension
nn_shapes = jnp.array([m, m, m, m])

# Credit: https://github.com/QuantEcon/krusell_smith_code/blob/master/KSfunctions.ipynb

# Build transition matrices
# probability of remaining in good state
pgg = 1 - 1 / z_g_ave_dur
# probability of remaining in bad state
pbb = 1 - 1 / z_b_ave_dur
# probability of changing from g to b
pgb = 1 - pgg
# probability of changing from b to g
pbg = 1 - pbb

# prob. of 0 to 0 cond. on g to g
p00_gg = 1 - 1 / u_g_ave_dur
# prob. of 0 to 0 cond. on b to b
p00_bb = 1 - 1 / u_b_ave_dur
# prob. of 0 to 1 cond. on g to g
p01_gg = 1 - p00_gg
# prob. of 0 to 1 cond. on b to b
p01_bb = 1 - p00_bb

# prob. of 0 to 0 cond. on g to b
p00_gb = puu_rel_gb2bb * p00_bb
# prob. of 0 to 0 cond. on b to g
p00_bg = puu_rel_bg2gg * p00_gg
# prob. of 0 to 1 cond. on g to b
p01_gb = 1 - p00_gb
# prob. of 0 to 1 cond. on b to g
p01_bg = 1 - p00_bg

# prob. of 1 to 0 cond. on  g to g
p10_gg = (u_g - u_g * p00_gg) / (1 - u_g)
# prob. of 1 to 0 cond. on b to b
p10_bb = (u_b - u_b * p00_bb) / (1 - u_b)
# prob. of 1 to 0 cond. on g to b
p10_gb = (u_b - u_g * p00_gb) / (1 - u_g)
# prob. of 1 to 0 cond on b to g
p10_bg = (u_g - u_b * p00_bg) / (1 - u_b)
# prob. of 1 to 1 cond. on  g to g
p11_gg = 1 - p10_gg
# prob. of 1 to 1 cond. on b to b
p11_bb = 1 - p10_bb
# prob. of 1 to 1 cond. on g to b
p11_gb = 1 - p10_gb
# prob. of 1 to 1 cond on b to g
p11_bg = 1 - p10_bg

#                   (g1)         (b1)          (g0)          (b0)
P = jnp.array([[pgg * p11_gg, pgb * p11_gb, pgg * p10_gg, pgb * p10_gb],
               [pbg * p11_bg, pbb * p11_bb, pbg * p10_bg, pbb * p10_bb],
               [pgg * p01_gg, pgb * p01_gb, pgg * p00_gg, pgb * p00_gb],
               [pbg * p01_bg, pbb * p01_bb, pbg * p00_bg, pbb * p00_bb]])
Pz = jnp.array([[pgg, pgb],
                [pbg, pbb]])
Peps_gg = jnp.array([[p11_gg, p10_gg],
                     [p01_gg, p00_gg]])
Peps_bb = jnp.array([[p11_bb, p10_bb],
                     [p01_bb, p00_bb]])
Peps_gb = jnp.array([[p11_gb, p10_gb],
                     [p01_gb, p00_gb]])
Peps_bg = jnp.array([[p11_bg, p10_bg],
                     [p01_bg, p00_bg]])

Peps = jnp.array([[Peps_gg, Peps_gb],
                  [Peps_bg, Peps_bb]])

# Potential states
zs = jnp.array([z_g, z_b])
es = jnp.array([1, 0])
states = jnp.array(jnp.meshgrid(es, zs)).T.reshape(-1, 2)
states = states[:, [1, 0]]


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
    alpha = config['alpha']
    delta = config['delta']

    kl = jnp.sum(X) / jnp.clip(jnp.sum(E), 1, None)
    w = (1 - alpha) * Z * jnp.power(kl, alpha)
    r = 1 - delta + (alpha * Z * jnp.power(kl, alpha - 1))
    return r, w


@jax.jit
def next_X(params, config, X, Z, E):
    R, W = prices(config, X, Z, E)
    w = jax.vmap(lambda x, e: (R * x) + (W * e))(X, E)
    c = jax.vmap(lambda i: neural_network(params, X, E, Z, E[i], w[i])[0])(jnp.arange(k))
    return w - c


@jax.jit
def loss(params, config, X, Z, E, key):
    Z1, E1 = next_state(Z, E, config, key)
    R, W = prices(config, X, Z, E)
    w = jax.vmap(lambda x, e: (R * x) + (W * e))(X, E)
    outputs = jax.vmap(lambda i: neural_network(params, X, E, Z, E[i], w[i]))(jnp.arange(k))
    c = outputs[..., 0]
    lm = outputs[..., 1]
    c_rel = c / w
    X1 = w - c
    R1, W1 = prices(config, X1, Z1, E1)
    w1 = jax.vmap(lambda x, e: (R * x) + (W * e))(X1, E1)
    c1 = jax.vmap(lambda i: neural_network(params, X1, E1, Z1, E1[i], w1[i])[0])(jnp.arange(k))

    g = jax.vmap(lambda p, c: config['beta'] * R1 * jax.grad(ces_utility(p))(c))(config['prefs'], c1)
    up = jax.vmap(lambda p, c: jax.grad(ces_utility(p))(c))(config['prefs'], c)
    # g = jax.vmap(lambda c: config['beta'] * R1 * jax.grad(log_utility())(c))(c1)
    # up = jax.vmap(lambda c: jax.grad(log_utility())(c))(c)
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
    keys = jax.random.split(key, (k + 1))
    Z_prime = jax.random.choice(keys[0], jnp.array([z_g, z_b]), p=(Z == z_g) * Pz[0] + (Z == z_b) * Pz[1])
    E_prime = jax.vmap(lambda e, k:
                       jax.random.choice(k,
                           jnp.array([1, 0]),
                           p=Peps[((Z == z_b).astype(int), (Z_prime == z_b).astype(int), (e == 0).astype(int))]))\
                       (E, keys[1:])
    return Z_prime, E_prime


def simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward):
    keys = jax.random.split(key, Zs.shape[0] * n_forward).reshape(n_forward, Zs.shape[0], 2)
    for i in range(n_forward):
        Xs = jax.vmap(next_X, in_axes=(None, None, 0, 0, 0))(params, config, Xs, Zs, Es)
        Zs, Es = jax.vmap(next_state, in_axes=(0, 0, None, 0))(Zs, Es, config, keys[i])
    return Xs, Zs, Es, keys[-1, -1]


def generate_random_state(params, config, key, n_forward=0):
    Zs = jax.random.choice(key, zs, shape=(n // 2,))
    Es = jax.random.choice(key, es, shape=(n // 2, k))
    Xs = jnp.exp(a * jax.random.normal(jax.random.PRNGKey(np.random.randint(1, int(1e8))), shape=(n // 2, k))) + b
    if n_forward > 0:
        Xs, Zs, Es, key = simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward)

    return Xs, Zs, Es, key


scale = 0.5
init_keys = jax.random.split(jax.random.PRNGKey(5), 10)
# theta0 = jax.random.gamma(jax.random.PRNGKey(1), scale, shape=(k, m))
w00 = scale * jax.random.normal(init_keys[0], shape=(2 * k + 3, nn_shapes[0]))
w01 = scale * jax.random.normal(init_keys[1], shape=(nn_shapes[0], nn_shapes[1]))
w02 = scale * jax.random.normal(init_keys[2], shape=(nn_shapes[1], nn_shapes[2]))
w03 = scale * jax.random.normal(init_keys[3], shape=(nn_shapes[2] + 2, nn_shapes[3]))
w0f = scale * jax.random.normal(init_keys[4], shape=(nn_shapes[1], 1))
b00 = scale * jax.random.normal(init_keys[5], shape=(1, nn_shapes[0]))
b01 = scale * jax.random.normal(init_keys[6], shape=(1, nn_shapes[1]))
b02 = scale * jax.random.normal(init_keys[7], shape=(1, nn_shapes[2]))
b03 = scale * jax.random.normal(init_keys[8], shape=(1, nn_shapes[3]))
b0f = scale * jax.random.normal(init_keys[9], shape=(1, 1))

params0 = {
    'w0': w00, 'w1': w01, 'w2': w02, 'w3': w03, 'cwf': w0f, 'lwf': w0f, 'b0': b00, 'b1': b01, 'b2': b02, 'b3': b03, 'cbf': b0f, 'lbf': b0f
}


def training_loop(opt_state, tol=1e-10, max_iter=10**4):
    j = 0
    jj = 0
    key = jax.random.PRNGKey(np.random.randint(1, int(1e8)))
    val_loss = jnp.inf
    grad = {'0': jnp.inf}
    opt_init, opt_update, get_params = adamax(step_size=0.001)
    params = get_params(opt_state)
    Xs, Zs, Es, key = generate_random_state(params, config, key, n_forward=1000)

    while j < max_iter and jnp.abs(val_loss) > tol:
        while jj < n_iter:
            keys = jax.random.split(key, 2)
            key = keys[-1]
            sample = jax.random.choice(jax.random.PRNGKey(j * n_iter + jj), jnp.arange(n), shape=(mb,))
            params = get_params(opt_state)
            grad = jax.grad(batch_loss, has_aux=True)(params, config, Xs[sample], Zs[sample], Es[sample], keys)[0]
            opt_state = opt_update(j * n_iter + jj, grad, opt_state)
            jj += 1

        # Start from a new random position, before moving into the current implied ergodic set
        params = get_params(opt_state)
        if n_forward > 0:
            Xs, Zs, Es, key = simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward)

        if j % 100 == 0:
            trained_params = unpack_optimizer_state(opt_state)
            pickle.dump(trained_params, open(f'./share/models/ks_disc_model_{k}_{j}.pkl', 'wb'))
            val, grad = jax.value_and_grad(batch_loss, has_aux=True)(params, config, Xs, Zs, Es, keys)
            val_loss = jnp.abs(val[0])
            c_val = val[1][0]
            kt_val = jnp.abs(val[1][1])
            print(f'Iteration: {j}\tTotal Loss: {val_loss}\tC Loss: {c_val}\tKT Loss: {kt_val}' +\
                  f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in grad.items()])}' +\
                  f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in params.items()])}')

        j += 1

    print(f'Terminating training with final statistics:\n' + \
          f'Iteration: {j}\tTotal Loss: {val_loss}\tC Loss: {c_val}\tKT Loss: {kt_val}' + \
          f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in grad.items()])}' + \
          f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in params.items()])}')
    return opt_state


def main():
    opt_init, opt_update, get_params = adamax(step_size=0.001)
    # saved_params = pickle.load(open(f'./share/models/ks_disc_model_{k}_final.pkl', 'rb'))
    # opt_state = pack_optimizer_state(saved_params)
    opt_state = opt_init(params0)

    opt_state = training_loop(opt_state, max_iter=n_epoch)
    # params = training_loop(max_iter=n_epoch)
    params = unpack_optimizer_state(opt_state)
    pickle.dump(params, open(f'./share/models/ks_disc_model_{k}_final.pkl', 'wb'))


if __name__ == '__main__':
    main()