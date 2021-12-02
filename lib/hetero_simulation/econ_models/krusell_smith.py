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
discrete = False

config = {
    'alpha': alpha, 'beta': beta, 'delta': delta,
    'sigma_z': sigma_z, 'sigma_e': sigma_e, 'rho_z': rho_z, 'rho_e': rho_e, 'discrete': discrete
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
n = 2 ** 8
mb = 2 ** 4
n_epoch = 500
n_iter = 2 * (n // mb)
n_forward = 10

k = 1000 # number of agents (~ size of state space)
m = 10 # embedding dimension
config['p0'] = 1. # weight on envelope condition
config['p1'] = 1. # weight on k-t conditions
config['p2'] = 1. # weight on prediction error
nn_shapes = jnp.array([m, 2 * m, 2 * m, m])


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

# Potential states
zs = jnp.array([z_g, z_b])
es = jnp.array([1, 0])
states = jnp.array(jnp.meshgrid(es, zs)).T.reshape(-1, 2)
states = states[:, [1, 0]]


@jax.jit
def V_hat(params, X, Z, e, x):
    X_tilde = jnp.concatenate([X.reshape(1, -1),
                               Z.reshape(1, -1),
                               e.reshape(1, -1),
                               x.reshape(1, -1)], axis=1)
    l1 = tanh(X_tilde, params['vw0'], params['vb0'])
    l2 = tanh(l1, params['vw1'], params['vb1'])
    # l3 = tanh(l2, params['vw2'], params['vb2'])
    # l4 = tanh(jnp.concatenate((l3, e[..., jnp.newaxis], x[..., jnp.newaxis])), params['vw3'], params['vb3'])
    return jnp.squeeze(linear(l2, params['vwf'], params['vbf']))


@jax.jit
def opt_control(params, X, Z, e, x):
    X_tilde = jnp.concatenate([X.reshape(1, -1),
                               Z.reshape(1, -1),
                               e.reshape(1, -1),
                               x.reshape(1, -1)], axis=1)
    l1 = tanh(X_tilde, params['cw0'], params['cb0'])
    l2 = tanh(l1, params['cw1'], params['cb1'])
    # l3 = tanh(l2, params['cw2'], params['cb2'])
    # l4 = tanh(jnp.concatenate((l3, e[..., jnp.newaxis], x[..., jnp.newaxis])), params['cw3'], params['cb3'])
    return jnp.squeeze(((tanh(l2, params['cwf'], params['cbf']) + 1) / 2))


@jax.jit
def gamma_prime(params, X, Z, E):
    X_tilde = jnp.concatenate([X.reshape(1, -1),
                               E.reshape(1, -1),
                               Z.reshape(1, -1)], axis=1)
    l1 = tanh(X_tilde, params['xw0'], params['xb0'])
    l2 = tanh(l1, params['xw1'], params['xb1'])
    # l3 = tanh(l2, params['xw2'], params['xb2'])
    # l4 = tanh (jnp.concatenate((l3, jnp.squeeze(X_tilde)), axis=0), params['xw3'], params['xb3'])
    return jnp.squeeze(softplus(l2, params['xwf'], params['xbf']))


@jax.jit
def lagrange_multiplier(params, X, Z, e, x):
    X_tilde = jnp.concatenate([X.reshape(1, -1),
                               Z.reshape(1, -1),
                               e.reshape(1, -1),
                               x.reshape(1, -1)], axis=1)
    l1 = tanh(X_tilde, params['lw0'], params['lb0'])
    l2 = tanh(l1, params['lw1'], params['lb1'])
    # l3 = tanh(l2, params['lw2'], params['lb2'])
    # l4 = tanh(jnp.concatenate((l3, e[..., jnp.newaxis], x[..., jnp.newaxis])), params['lw3'], params['lb3'])
    return jnp.squeeze(softplus(l2, params['lwf'], params['lbf']))


@jax.jit
def fischer_burmeister(a, b):
    return a + b - jnp.sqrt(jnp.power(a, 2) + jnp.power(b, 2))


# Discrete exo states
# @jax.jit
# def prices(config, X, Z, E):
#     alpha = config['alpha']
#     kl = jnp.sum(X) / jnp.clip(jnp.sum(E), 1, None)
#     w = (1 - alpha) * Z * jnp.power(kl, alpha)
#     r = alpha * Z * jnp.power(kl, alpha - 1)
#     return r, w


# Continuous exo states
@jax.jit
def prices(config, X, Z, E):
    sumk = jnp.sum(X)
    sumexpl = jnp.sum(jnp.exp(E))
    w = (1 - alpha) * jnp.exp(Z) * jnp.power(sumk, config['alpha'])
    r = 1 - config['delta'] + config['alpha'] * jnp.exp(Z) * jnp.power(sumk, alpha - 1) * sumexpl
    return r, w


# Discrete exo states
# @jax.jit
# def future(params, config, X, Z, E, i):
#     u = lambda c: log_utility()(c)
#     k = X[i]
#     e = E[i]
#
#     r, w = prices(config['alpha'], X, Z, E)
#     x = ((1 - config['delta'] + r)) * k + (w * e)
#
#     lm = lagrange_multiplier(params, X, Z, e, x)
#     c_star = opt_control(params, X, Z, e, x)
#     c_star_rel = c_star / x
#     x_prime = x - jnp.sum(c_star)
#     X_prime = gamma_prime(params, X, Z, E)
#     # X_prime = X.at[i].set(x_prime)
#
#     s = 0 * ((Z == z_g) & (e == 1)) + \
#         1 * ((Z == z_b) & (e == 1)) + \
#         2 * ((Z == z_g) & (e == 0)) + \
#         3 * ((Z == z_b) & (e == 0))
#     V_prime = lambda Y, y: config['beta'] * jnp.dot(P[s], jax.vmap(lambda state: V_hat(params, Y, state[0], state[1], y))(states))
#
#     # Consider own and aggregate effect
#     grad = jax.grad(V_prime, argnums=(0, 1))(X, x) # Use current X => Envelope theorem
#     V_grad = grad[0][i] + grad[1]
#     # only consider own effect
#     # grad = jax.grad(V_prime, argnums=(1))(X_prime, x_prime)
#     # V_grad = grad
#
#     f = jnp.squeeze(u(c_star) + V_prime(X_prime, x_prime))
#     g = V_grad
#
#     return f, lm, g, c_star_rel, x_prime


# Continuous exo states
@jax.jit
def future(params, config, X, Z, E, Z_prime, E_prime, i):
    u = lambda c: log_utility()(c)
    k = X[i]
    e = E[i]
    e_prime = E_prime[i]
    r_t, w_t = prices(config, X, Z, E)
    x = (r_t * k) + (w_t * jnp.exp(e))

    lm = lagrange_multiplier(params, X, Z, e, k)
    c_star_rel = opt_control(params, X, Z, e, x)
    c_star = c_star_rel * x
    x_prime = x * (1 - c_star_rel)
    X_prime = gamma_prime(params, X, Z, E)
    c_star_rel_1 = opt_control(params, X_prime, Z_prime, e_prime, x_prime)
    c_star_1 = x_prime * (1 - c_star_rel_1)
    r_t1, w_t1 = prices(config, X_prime, Z_prime, E_prime)

    V_prime = lambda Y, y: config['beta'] * V_hat(params, Y, Z_prime, e_prime, y)

    # grad = jax.grad(V_prime, argnums=(0, 1))(X_prime, x_prime)
    # V_grad = grad[0][i] + grad[1]

    # f = jnp.squeeze(u(c_star) + V_prime(X_prime, x_prime))
    # g = jnp.exp(V_grad)
    g = config['beta'] * r_t1 * jax.grad(u)(c_star_1)

    # return f, lm, g, c_star_rel, x_prime
    return lm, g, c_star_rel, x_prime


@jax.jit
def batch_loss(params, config, Xs, Zs, Es, keys):
    n = Xs.shape[0]
    Xs = jnp.concatenate((Xs, Xs), axis=0)
    Zs = jnp.concatenate((Zs, Zs), axis=0)
    Es = jnp.concatenate((Es, Es), axis=0)

    Zs_prime, Es_prime = next_state(Zs, Es, config, keys)

    v_hats = jax.vmap(lambda X, Z, E: jax.vmap(lambda i: V_hat(params, X, Z, E[i], X[i]))(jnp.arange(k)))(Xs, Zs, Es)
    opt_cs = jax.vmap(lambda X, Z, E: jax.vmap(lambda i: opt_control(params, X, Z, E[i], X[i]))(jnp.arange(k)))(Xs, Zs, Es)
    lm_hats = jax.vmap(lambda X, Z, E: jax.vmap(lambda i: lagrange_multiplier(params, X, Z, E[i], X[i]))(jnp.arange(k)))(Xs, Zs, Es)
    X_prime_hat = jax.vmap(gamma_prime, in_axes=(None, 0, 0, 0))(params, Xs, Zs, Es)
    # fs, lms, gs, c_star_rels, X_prime = jax.vmap(lambda X, Z, E, Z_prime, E_prime: jax.vmap(lambda i: future(params, config, X, Z, E, Z_prime, E_prime, i))(jnp.arange(k)))(Xs, Zs, Es, Zs_prime, Es_prime)
    lms, gs, c_star_rels, X_prime = jax.vmap(
        lambda X, Z, E, Z_prime, E_prime: jax.vmap(lambda i: future(params, config, X, Z, E, Z_prime, E_prime, i))(
            jnp.arange(k)))(Xs, Zs, Es, Zs_prime, Es_prime)
    # f_diff = v_hats - fs
    # f2 = jnp.mean(jax.vmap(lambda x, y: jnp.dot(x, y))(f_diff[:n], f_diff[n:]))
    # f2 = jnp.trace(jnp.dot(f_diff[:n], f_diff[n:].T)) / n

    u = lambda c: log_utility()(c)
    u_primes = jax.grad(u)(opt_cs)
    # g_diff = jax.vmap(lambda g, up, lm: (g / up) - lms)(gs, u_primes, lms)
    g_diff = ((gs / u_primes) - lms)
    g2 = jnp.mean(jax.vmap(lambda x, y: jnp.dot(x, y))(g_diff[:n], g_diff[n:]))
    # g2 = jnp.trace(jnp.dot(g_diff[:n], g_diff[n:].T)) / n

    kt_cond = jnp.mean(jax.vmap(lambda c, lm: fischer_burmeister(1-c, 1-lm)**2)(c_star_rels, lm_hats))

    x_diff = X_prime_hat - X_prime
    x2 = jnp.mean(jax.vmap(lambda x, y: jnp.dot(x, y))(x_diff[:n], x_diff[n:]))
    # x2 = jnp.trace(jnp.dot(x_diff[:n], x_diff[n:].T)) / n

    # return (jnp.mean(f2), jnp.mean(g2), jnp.mean(x2), X_prime)
    # return f2 + config['p0'] * g2 + config['p1'] * kt_cond + config['p2'] * x2,\
    #        (f2, g2, kt_cond, x2, c_star_rels, Zs_prime[:n], Es_prime[:n], X_prime[:n])
    return config['p0'] * g2 + config['p1'] * kt_cond + config['p2'] * x2, \
           (g2, kt_cond, x2, c_star_rels, Zs_prime[:n], Es_prime[:n], X_prime[:n])



@jax.jit
def next_state(Zs, Es, config, keys):
    # Zs_prime = jax.vmap(
    #     lambda z, key: jax.random.choice(key, jnp.array([z_g, z_b]), p=(z == z_g) * Pz[0] + (z == z_b) * Pz[1]))(Zs, keys[:, 0])
    # Es_prime = jax.vmap(lambda z, z_prime, E, keys,:
    #                     jax.vmap(lambda e, key:
    #                              jax.random.choice(key,
    #                                                jnp.array([1, 0]),
    #                                                p=(((z == z_g) & (z_prime == z_g)) * Peps_gg[1 - e] +\
    #                                                   ((z == z_g) & (z_prime == z_b)) * Peps_gb[1 - e] +\
    #                                                   ((z == z_b) & (z_prime == z_g)) * Peps_bg[1 - e] +\
    #                                                   ((z == z_b) & (z_prime == z_b)) * Peps_bb[1 - e])))(E, keys))(Zs, Zs_prime, Es, keys)
    Zs_prime = jax.vmap(lambda z, k: config['rho_z'] * z + config['sigma_z'] * jax.random.normal(k))(Zs, keys[:, 0])
    Es_prime = jax.vmap(lambda e, k: config['rho_e'] * e + config['sigma_e'] * jax.random.normal(k))(Es.reshape(-1), keys[:, 1:].reshape(-1, 2)).reshape(Es.shape)
    return Zs_prime, Es_prime


def simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward):
    for _ in range(n_forward):
        keys = jax.random.split(key, n * (k + 1)).reshape(n, (k + 1), 2)
        Zs, Es, Xs = batch_loss(params, config, Xs, Zs, Es, keys)[1][-3:]
        # Xs = jax.vmap(gamma_prime, in_axes=(None, 0, 0, 0))(params, Xs, Zs, Es)
        # Zs, Es = next_state(Zs, Es, config, keys[:(n // 2)])
        key = keys[-1, -1]
    return Xs, Zs, Es, key


def generate_random_state(params, config, key, n_forward=0, discrete=True):
    if discrete:
        Zs = jax.random.choice(jax.random.PRNGKey(np.random.randint(1, int(1e8))), jnp.array([z_g, z_b]), shape=(n // 2,))
        Ps = jax.vmap(lambda z: (z == z_g) * jnp.array([u_g, 1 - u_g]) + (z == z_b) * jnp.array([u_b, 1 - u_b]))(Zs)
        Es = jax.vmap(lambda p, k: jax.vmap(lambda k: jax.random.choice(jax.random.PRNGKey(k), jnp.array([0, 1]), p=p))(k))(Ps, np.random.randint(1, int(1e8), size=(n // 2, k)))
    else:
        Zs = jnp.zeros(shape=(n // 2,))
        Es = jnp.zeros(shape=(n // 2, k))

    Xs = jnp.exp(a * jax.random.normal(jax.random.PRNGKey(np.random.randint(1, int(1e8))), shape=(n // 2, k))) + b

    if n_forward > 0:
        Xs, Zs, Es, key = simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward)

    return Xs, Zs, Es, key


scale = 0.05
# theta0 = jax.random.gamma(jax.random.PRNGKey(1), scale, shape=(k, m))
vw00 = scale * jnp.ones(nn_shapes[0] * (k + 3)).reshape(k + 3, nn_shapes[0])
cw00 = scale * jnp.ones(nn_shapes[0] * (k + 3)).reshape(k + 3, nn_shapes[0])
xw00 = scale * jnp.ones(nn_shapes[0] * (2 * k + 1)).reshape(2 * k + 1, nn_shapes[0])
w01 = scale * jnp.ones(nn_shapes[0] * nn_shapes[1]).reshape(nn_shapes[0], nn_shapes[1])
w02 = scale * jnp.ones(nn_shapes[1] * nn_shapes[2]).reshape(nn_shapes[1], nn_shapes[2])
vw03 = scale * jnp.ones((nn_shapes[2] + 2) * nn_shapes[3]).reshape(nn_shapes[2] + 2, nn_shapes[3])
cw03 = scale * jnp.ones((nn_shapes[2] + 2) * nn_shapes[3]).reshape(nn_shapes[2] + 2, nn_shapes[3])
xw03 = scale * jnp.ones((nn_shapes[2] + 2 * k + 1) * nn_shapes[3]).reshape(nn_shapes[2] + 2 * k + 1, nn_shapes[3])
vw0f = scale * jnp.ones(nn_shapes[1]).reshape(nn_shapes[1], 1)
cw0f = scale * jnp.ones(nn_shapes[1]).reshape(nn_shapes[1], 1)
xw0f = scale * jnp.ones(nn_shapes[1] * k).reshape(nn_shapes[1], k)
b00 = scale * jnp.ones(nn_shapes[0]).reshape(1, nn_shapes[0])
b01 = scale * jnp.ones(nn_shapes[1]).reshape(1, nn_shapes[1])
b02 = scale * jnp.ones(nn_shapes[2]).reshape(1, nn_shapes[2])
b03 = scale * jnp.ones(nn_shapes[3]).reshape(1, nn_shapes[3])
b0f = scale * jnp.ones(1).reshape(1, 1)

v_params0 = {
    'vw0': vw00, 'vw1': w01, 'vw2': w02, 'vw3': vw03, 'vwf': vw0f, 'vb0': b00, 'vb1': b01, 'vb2': b02, 'vb3':b03, 'vbf': b0f
}

c_params0 = {
    'cw0': cw00, 'cw1': w01, 'cw2': w02, 'cw3': cw03, 'cwf': cw0f, 'cb0': b00, 'cb1': b01, 'cb2': b02, 'cb3': b03, 'cbf': b0f
}

l_params0 = {
    'lw0': cw00, 'lw1': w01, 'lw2': w02, 'lw3': cw03, 'lwf': cw0f, 'lb0': b00, 'lb1': b01, 'lb2': b02, 'lb3': b03, 'lbf': b0f
}

x_params0 = {
    'xw0': xw00, 'xw1': w01, 'xw2': w02, 'xw3': xw03, 'xwf': xw0f, 'xb0': b00, 'xb1': b01, 'xb2': b02, 'xb3': b03, 'xbf': b0f
}

params0 = {
    **v_params0, **c_params0, **l_params0, **x_params0
}


def training_loop(opt_state, tol=1e-10, max_iter=10 ** 4):
    j = 0
    key = jax.random.PRNGKey(np.random.randint(1, int(1e8)))
    # v_val = jnp.inf
    # c_val = jnp.inf
    # x_val = jnp.inf
    val_loss = jnp.inf

    # v_grad = {'0': jnp.inf}
    # c_grad = {'0': jnp.inf}
    # x_grad = {'0': jnp.inf}
    grad = {'0': jnp.inf}

    opt_init, opt_update, get_params = adam(step_size=0.01)
    params = get_params(opt_state)

    # v_opt_state = opt_init(v_params0)
    # c_opt_state = opt_init(c_params0)
    # x_opt_state = opt_init(x_params0)
    # opt_state = opt_init(params0)

    Xs, Zs, Es, key = generate_random_state(params, config, key, n_forward=n_forward, discrete=config['discrete'])

    while j < max_iter and max([jnp.max(jnp.abs(v)) for k, v in grad.items()]) > tol and jnp.abs(val_loss) > tol:
        jj = 0
        while jj < n_iter:
            keys = jax.random.split(key, mb * (k + 1)).reshape(mb, k + 1, 2)
            key = keys[-1, -1]
            # v_params = get_params(v_opt_state)
            # x_params = get_params(x_opt_state)
            # c_params = get_params(c_opt_state)
            params = get_params(opt_state)

            sample = jax.random.choice(jax.random.PRNGKey(np.random.randint(1, int(1e8))), jnp.arange(n // 2), shape=(mb // 2,))

            # v_loss = lambda v_params, other_params: batch_loss({**v_params, **other_params}, config, Xs[sample], Zs[sample], Es[sample])[0]
            # c_loss = lambda c_params, other_params: batch_loss({**c_params, **other_params}, config, Xs[sample], Zs[sample], Es[sample])[1]
            # x_loss = lambda x_params, other_params: batch_loss({**x_params, **other_params}, config, Xs[sample], Zs[sample], Es[sample])[2]

            # v_val, v_grad = jax.value_and_grad(v_loss)(v_params, {**c_params, **x_params})
            # c_val, c_grad = jax.value_and_grad(c_loss)(c_params, {**v_params, **x_params})
            # x_val, x_grad = jax.value_and_grad(x_loss)(x_params, {**v_params, **c_params})
            val, grad = jax.value_and_grad(batch_loss, has_aux=True)(params, config, Xs[sample], Zs[sample], Es[sample], keys)
            # val_loss = v_val + c_val + x_val
            val_loss = jnp.abs(val[0])
            c_star_rel = val[1][4]
            X_primes = val[1][-1]
            assert (c_star_rel < 1).all()
            assert (X_primes > 0).all()
            if jnp.isnan(val_loss):
                raise ValueError('Loss is nan')

            # v_val = jnp.abs(val[1][0])
            c_val = jnp.abs(val[1][0])
            kt_val = jnp.abs(val[1][1])
            x_val = jnp.abs(val[1][2])

            # v_opt_state = opt_update(j * n_iter + jj, v_grad, v_opt_state)
            # c_opt_state = opt_update(j * n_iter + jj, c_grad, c_opt_state)
            # x_opt_state = opt_update(j * n_iter + jj, x_grad, x_opt_state)
            opt_state = opt_update(j * n_iter + jj, grad, opt_state)

            jj += 1

        # Start from a new random position, before moving into the current implied ergodic set
        params = get_params(opt_state)
        if n_forward > 0:
            Xs, Zs, Es, key = simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward)
        # Xs, Zs, Es, key = generate_random_state(params, config, key, n_forward=n_forward, discrete=config['discrete'])
        # params = {**get_params(v_opt_state), **get_params(c_opt_state), **get_params(x_opt_state)}
        # grad = {**v_grad, **c_grad, **x_grad}

        if j % 10 == 0:
            trained_params = unpack_optimizer_state(opt_state)
            pickle.dump(trained_params, open(f'./share/models/ks_model_{k}_{j}.pkl', 'wb'))
            # pickle.dump(params, open(f'./share/models/ks_model_{k}_{j}.pkl', 'wb'))
            print(f'Iteration: {j}\tTotal Loss: {val_loss}\tC Loss: {c_val}\tK-T Loss: {kt_val}\tX Loss: {x_val}' +\
                  f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in grad.items()])}' +\
                  f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in params.items()])}')
        j += 1

        # update weights
        # config['p0'] = jnp.float32(10 ** (- jnp.round(jnp.log10(jnp.abs(c_val)))))
        # config['p1'] = jnp.float32(10 ** (- jnp.round(jnp.log10(jnp.abs(kt_val)))))
        # config['p2'] = jnp.float32(10 ** (- jnp.round(jnp.log10(jnp.abs(x_val)))))


    print(f'Terminating training with final statistics:\n' + \
          f'Iteration: {j}\tTotal Loss: {val_loss} \tV Loss: {v_val}\tC Loss: {c_val}\tK-T Loss: {kt_val}\tX Loss: {x_val}' + \
          f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in grad.items()])}' + \
          f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in params.items()])}')
    return opt_state
    # return params

def main():
    opt_init, opt_update, get_params = adam(step_size=0.01)
    # saved_params = pickle.load(open(f'./share/models/ks_model_{k}_final.pkl', 'rb'))
    # opt_state = pack_optimizer_state(saved_params)
    opt_state = opt_init(params0)

    opt_state = training_loop(opt_state, max_iter=n_epoch)
    # params = training_loop(max_iter=n_epoch)
    params = unpack_optimizer_state(opt_state)
    pickle.dump(params, open(f'./share/models/ks_model_{k}_final.pkl', 'wb'))


if __name__ == '__main__':
    main()