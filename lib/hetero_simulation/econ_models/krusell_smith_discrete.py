import numpy as np
import pickle
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adamax, unpack_optimizer_state, pack_optimizer_state
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
n = 2**14
mb = 2**11
n_epoch = 2500
n_iter = n // mb
n_forward = 10

k = 5 # number of agents (~ size of state space)
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
    kl = jnp.sum(X) / jnp.clip(jnp.sum(E), 1, None)
    w = (1 - alpha) * Z * jnp.power(kl, alpha)
    r = alpha * Z * jnp.power(kl, alpha - 1)
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
    keys = jax.random.split(key, Es.shape[0] * (k + 1)).reshape(Es.shape[0], k + 1, 2)
    Zs_prime = jax.vmap(
        lambda z, key: jax.random.choice(key, jnp.array([z_g, z_b]), p=(z == z_g) * Pz[0] + (z == z_b) * Pz[1]))(Zs, keys[:, 0])
    Es_prime = jax.vmap(lambda z, z_prime, E, keys:
                        jax.vmap(lambda e, key:
                                 jax.random.choice(key,
                                                   jnp.array([1, 0]),
                                                   p=(jnp.float32((z == z_g) & (z_prime == z_g) & (e == 1)) * Peps_gg[0] +\
                                                      jnp.float32((z == z_g) & (z_prime == z_g) & (e == 0)) * Peps_gg[1] +\
                                                      jnp.float32((z == z_g) & (z_prime == z_b) & (e == 1)) * Peps_gb[0] +\
                                                      jnp.float32((z == z_g) & (z_prime == z_b) & (e == 0)) * Peps_gb[1] +\
                                                      jnp.float32((z == z_b) & (z_prime == z_g) & (e == 1)) * Peps_bg[0] +\
                                                      jnp.float32((z == z_b) & (z_prime == z_g) & (e == 0)) * Peps_bg[1] +\
                                                      jnp.float32((z == z_b) & (z_prime == z_b) & (e == 1)) * Peps_bb[0] +\
                                                      jnp.float32((z == z_b) & (z_prime == z_b) & (e == 0)) * Peps_bb[1]))) \
                                 (E, keys)) \
                        (Zs, Zs_prime, Es, keys[:, 1:])
    return Zs_prime, Es_prime


def simulate_state_forward(params, config, Xs, Zs, Es, key, n_forward):
    for _ in range(n_forward):
        keys = jax.random.split(key, 2)
        Zs, Es, Xs = batch_loss(params, config, Xs, Zs, Es, keys)[1][-3:]
        key = keys[-1]
    return Xs, Zs, Es, key


def generate_random_state(params, config, key, n_forward=0):
    Zs = jax.random.choice(key, zs, shape=(n // 2,))
    Es = jax.random.choice(key, es, shape=(n // 2, k))
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
    opt_init, opt_update, get_params = adamax(step_size=0.001)
    params = get_params(opt_state)

    # with jax.disable_jit():
    Xs, Zs, Es, key = generate_random_state(params, config, key, n_forward=n_forward)

    while j < max_iter and max([jnp.max(jnp.abs(v)) for k, v in grad.items()]) > tol and jnp.abs(val_loss) > tol:
        jj = 0
        while jj < n_iter:
            keys = keys = jax.random.split(key, 2)
            key = keys[-1]
            params = get_params(opt_state)

            sample = jax.random.choice(jax.random.PRNGKey(np.random.randint(1, int(1e8))), jnp.arange(n // 2), shape=(mb // 2,))
            # with jax.disable_jit():
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
            pickle.dump(trained_params, open(f'../models/ks_disc_model_{k}_{j}.pkl', 'wb'))
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
    opt_init, opt_update, get_params = adamax(step_size=0.001)
    # saved_params = pickle.load(open(f'./share/models/ks_model_{k}_final.pkl', 'rb'))
    # opt_state = pack_optimizer_state(saved_params)
    opt_state = opt_init(params0)

    opt_state = training_loop(opt_state, max_iter=n_epoch)
    # params = training_loop(max_iter=n_epoch)
    params = unpack_optimizer_state(opt_state)
    pickle.dump(params, open(f'../models/ks_model_{k}_final.pkl', 'wb'))


if __name__ == '__main__':
    main()