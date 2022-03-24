import numpy as np
import pickle
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adam, unpack_optimizer_state, pack_optimizer_state
from hetero_simulation.archive.agent import log_utility
from hetero_simulation.ml.utils import *
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update('jax_disable_jit', True)
# jax.config.update('jax_debug_nans', True)

# Model parameters
econ_config = {
    'phi': 0.2,
    'beta': 0.97,
    'pa': 1,
    'rh': 2,
    'p': 1,
    'w': 0.5,
    'ibar': 0.8,
    'sigma_z': 0, # 0.01,
    'sigma_i': 0, #0.01,
    'rho_z': 0, #0.95,
    'rho_i': 0, #0.5,
    'T': 100,
    'bc_penalty_coef': 1e5
}

# ML parameters
ml_config = {
    'mb': 2 ** 12,
    'n_epoch': 10 ** 5,
    'save_interval': 10 ** 2,
    'report_interval': 10,
    'step_size': 1e-2,
    'nn_shapes': jnp.array([64, 64, 64, 64]),
    'init_seed': 1234, # np.random.randint(1, 10000000)
    'init_scale': 0.001,
    'train_new': False,
    'start_from': "latest"
}


# @jax.jit
# def renter_policy(params, z0, i0, m0, t0):
#     live_next = jnp.int32(t0 < econ_config['T'])
#     X_tilde = jnp.concatenate([z0.reshape(1, -1),
#                                i0.reshape(1, -1),
#                                m0.reshape(1, -1),
#                                t0.reshape(1, -1)], axis=1)
#     l1 = relu(X_tilde, params['w0'], params['b0'])
#     l2 = relu(l1, params['w1'], params['b1'])
#     lf = relu(l2, params['w2'], params['b2'])
#     # lf = relu(l3, params['w3'], params['b3'])
#
#     fixed_income = m0
#     n0 = scaled_sigmoid(lf, params['nwf'], params['nbf'])
#     wealth = fixed_income + econ_config['w'] * n0
#
#     s0 = scaled_sigmoid(lf, params['swf'], params['sbf'])
#     savings = wealth * s0 # * live_next
#     spending = wealth - savings
#
#     rel_cons = scaled_sigmoid(lf, params['cwf'], params['cbf'])
#     c0 = rel_cons * spending
#     h0 = ((1 - rel_cons) * spending) / econ_config['rh']
#
#     a1 = 0.
#     b1 = 0.
#     m1 = savings
#
#     v0 = custom_value_fn(lf, params['vwf'], params['vbf']) * live_next
#     l0 = 1 / c0 # exp(lf, params['lwf'], params['lbf'])
#
#     bc = (econ_config['w'] * n0 - econ_config['rh'] * h0 + m0) - (c0 + m1)
#     return (v0, c0, h0, n0, m1, a1, b1, 0., bc, l0)


@jax.jit
def renter_policy(params, z0, i0, m0, a0, b0, t0):
    live_next = jnp.int32(t0 < econ_config['T'])
    X_tilde = jnp.concatenate([z0.reshape(1, -1),
                               i0.reshape(1, -1),
                               m0.reshape(1, -1),
                               a0.reshape(1, -1),
                               b0.reshape(1, -1),
                               t0.reshape(1, -1)], axis=1)
    l1 = relu(X_tilde, params['w0'], params['b0'])
    l2 = relu(l1, params['w1'], params['b1'])
    lf = relu(l2, params['w2'], params['b2'])
    # l4 = relu(l3, params['w3'], params['b3'])

    fixed_income = (econ_config['pa'] + econ_config['rh']) * a0 + (1 + i0) * b0 + m0
    n0 = scaled_sigmoid(lf, params['nwf'], params['nbf'])
    wealth = fixed_income + econ_config['w'] * n0

    s0 = scaled_sigmoid(lf, params['swf'], params['sbf'])
    savings = wealth * s0 * live_next
    spending = wealth - savings

    rel_cons = scaled_sigmoid(lf, params['cwf'], params['cbf'])
    c0 = rel_cons * spending
    h0 = ((1 - rel_cons) * spending) / econ_config['rh']

    rel_illiquid = 0 # scaled_sigmoid(lf, params['iwf'], params['ibf'])
    rel_bond = scaled_sigmoid(lf, params['bwf'], params['bbf'])
    a1 = (rel_illiquid * (1 - rel_bond) * savings) / econ_config['pa']
    b1 = rel_illiquid * rel_bond * savings
    m1 = savings - (a1 * econ_config['pa']) - b1
    # rel_bond = scaled_sigmoid(lf, params['bwf'], params['bbf'])
    # a1 = ((1 - rel_bond) * savings) / (econ_config['pa'])
    # b1 = rel_bond * savings
    # m1 = 0.

    v0 = custom_value_fn(lf, params['vwf'], params['vbf']) * live_next
    l0 = 1 / c0 # exp(lf, params['lwf'], params['lbf'])

    bc = (econ_config['w'] * n0 + econ_config['rh'] * (a0 - h0) + (1 + i0) * b0 + m0) - (c0 + (econ_config['pa']) * (a1 - a0) + b1 + m1)
    return (v0, c0, h0, n0, m1, a1, b1, 1., bc, l0)


@jax.jit
def owner_policy(params, z0, i0, m0, a0, b0, t0):
    live_next = jnp.int32(t0 < econ_config['T'])
    X_tilde = jnp.concatenate([z0.reshape(1, -1),
                               i0.reshape(1, -1),
                               m0.reshape(1, -1),
                               a0.reshape(1, -1),
                               b0.reshape(1, -1),
                               t0.reshape(1, -1)], axis=1)
    l1 = relu(X_tilde, params['w0'], params['b0'])
    l2 = relu(l1, params['w1'], params['b1'])
    lf = relu(l2, params['w2'], params['b2'])
    # l4 = relu(l3, params['w3'], params['b3'])

    fixed_income = (econ_config['pa'] + econ_config['rh']) * a0 + (1 + i0) * b0 + m0
    n0 = scaled_sigmoid(lf, params['nwf'], params['nbf'])
    wealth = fixed_income + econ_config['w'] * n0

    s0 = scaled_sigmoid(lf, params['swf'], params['sbf'])
    savings = wealth * s0 * live_next
    spending = wealth - savings

    rel_cons = scaled_sigmoid(lf, params['cwf'], params['cbf'])
    c0 = rel_cons * spending
    h0 = ((1 - rel_cons) * spending) / econ_config['rh']

    rel_illiquid = 1 # scaled_sigmoid(lf, params['iwf'], params['ibf'])
    rel_bond = scaled_sigmoid(lf, params['bwf'], params['bbf'])
    a1 = (rel_illiquid * (1 - rel_bond) * savings) / econ_config['pa']
    b1 = rel_illiquid * rel_bond * savings
    m1 = savings - (a1 * econ_config['pa']) - b1
    # rel_bond = scaled_sigmoid(lf, params['bwf'], params['bbf'])
    # a1 = ((1 - rel_bond) * savings) / (econ_config['pa'])
    # b1 = rel_bond * savings
    # m1 = 0.

    v0 = custom_value_fn(lf, params['vwf'], params['vbf']) * live_next
    l0 = 1 / c0 # exp(lf, params['lwf'], params['lbf'])

    bc = (econ_config['w'] * n0 + econ_config['rh'] * (a0 - h0) + (1 + i0) * b0 + m0) - (c0 + (econ_config['pa']) * (a1 - a0) + b1 + m1)
    return (v0, c0, h0, n0, m1, a1, b1, 1., bc, l0)


@jax.jit
def neural_network(params, z0, i0, m0, a0, b0, o0, t0):
    # This is just because inputs are drawn independently from grid
    # If data were generated via simulation this would be unnecessary
    o0 = jnp.int32(t0 > 0) * o0
    a0 = o0 * a0
    b0 = o0 * b0

    ov0, oc0, oh0, on0, om1, oa1, ob1, oo1, obc, ol0 = owner_policy(params['owner'], z0, i0, m0, a0, b0, t0)
    # rv0, rc0, rh0, rn0, rm1, ra1, rb1, ro1, rbc, rl0 = renter_policy(params['renter'], z0, i0, m0, t0)
    rv0, rc0, rh0, rn0, rm1, ra1, rb1, ro1, rbc, rl0 = renter_policy(params['renter'], z0, i0, m0, a0, b0, t0)

    o1 = jnp.minimum(jnp.int32(m0 > econ_config['phi'] * econ_config['pa']) + o0, 1)
    return (o1 * ov0 + (1 - o1) * rv0,
            o1 * oc0 + (1 - o1) * rc0,
            o1 * oh0 + (1 - o1) * rh0,
            o1 * on0 + (1 - o1) * rn0,
            o1 * om1 + (1 - o1) * rm1,
            o1 * oa1 + (1 - o1) * ra1,
            o1 * ob1 + (1 - o1) * rb1,
            o1,
            o1 * obc + (1 - o1) * rbc,
            o1 * ol0 + (1 - o1) * rl0)

    # X_tilde = jnp.concatenate([z0.reshape(1, -1),
    #                            i0.reshape(1, -1),
    #                            m0.reshape(1, -1),
    #                            a0.reshape(1, -1),
    #                            b0.reshape(1, -1),
    #                            o0.reshape(1, -1),
    #                            t0.reshape(1, -1)], axis=1)
    # l1 = relu(X_tilde, params['w0'], params['b0'])
    # l2 = relu(l1, params['w1'], params['b1'])
    # lf = relu(l2, params['w2'], params['b2'])
    # # l4 = relu(l3, params['w3'], params['b3'])
    #
    # o1 = jnp.minimum(o0 + jnp.int32(m0 > econ_config['phi'] * econ_config['pa']), 1.)
    # fixed_income = (econ_config['pa'] + econ_config['rh']) * a0 + (1 + i0) * b0 + m0
    # n0 = scaled_sigmoid(lf, params['nwf'], params['nbf'])
    # wealth = fixed_income + econ_config['w'] * n0
    #
    # s0 = scaled_sigmoid(lf, params['swf'], params['sbf'])
    # savings = live_next * wealth * s0
    # spending = wealth - savings
    #
    # rel_cons = scaled_sigmoid(lf, params['cwf'], params['cbf'])
    # c0 = rel_cons * spending
    # h0 = ((1 - rel_cons) * spending) / econ_config['rh']
    #
    # rel_illiquid = o1 * scaled_sigmoid(lf, params['iwf'], params['ibf'])
    # rel_bond = scaled_sigmoid(lf, params['bwf'], params['bbf'])
    # a1u = (rel_illiquid * (1 - rel_bond) * savings) / econ_config['pa']
    # b1u = rel_illiquid * rel_bond * savings
    # a1 = o1 * a1u
    # b1 = o1 * b1u
    # m1 = savings - (a1 * econ_config['pa']) - b1
    #
    # v0 = live_next * log(lf, params['vwf'], params['vbf'])
    #
    # bc = (econ_config['w'] * n0 + econ_config['rh'] * (a0 - h0) + (1+i0) * b0 + m0) - (c0 + econ_config['pa'] * (a1 - a0) + b1 + m1)
    # return (v0, c0, h0, n0, m1, a1, b1, a1u, b1u, o1, bc)


@jax.jit
def loss(params, m0, a0, b0, o0, t0, z0, i0, key):
    z1, i1 = draw_states(z0, i0, key)
    v0, c0, h0, n0, m1, a1, b1, o1, bc, l0 = neural_network(params, z0, i0, m0, a0, b0, o0, t0)

    # Calculate some values we will need
    u = lambda c, h, n: log_utility()(c) + log_utility()(h) + log_utility()(1 - n)
    utility = u(c0, h0, n0)
    u_grad = jax.grad(u, (0, 1, 2))(c0, h0, n0)
    uc = u_grad[0]
    uh = u_grad[1]
    un = u_grad[2]

    v0f = lambda params, z0, i0, m0, a0, b0, o0, t0: neural_network(params, z0, i0, m0, a0, b0, o0, t0)[0]
    v0_grad = jax.grad(v0f, (3, 4, 5))(params, z0, i0, m0, a0, b0, o0, t0)
    v0m = v0_grad[0]
    v0a = v0_grad[1]
    v0b = v0_grad[2]

    v1f = lambda params, z1, i1, m1, a1, b1, o1, t1: neural_network(params, z1, i1, m1, a1, b1, o1, t1)[0]
    v1 = v1f(params, z1, i1, m1, a1, b1, o1, t0 + 1)
    v1_grad = jax.grad(v1f, (3, 4, 5))(params, z1, i1, m1, a1, b1, o1, t0 + 1)
    v1m = v1_grad[0]
    v1a = v1_grad[1]
    v1b = v1_grad[2]

    l1f = lambda params, z1, i1, m1, a1, b1, o1, t1: neural_network(params, z1, i1, m1, a1, b1, o1, t1)[-1]
    l1 = l1f(params, z1, i1, m1, a1, b1, o1, t0 + 1)

    # Calculate the counterfactual value of being a homeowner if the agent is a renter
    ov0, _, _, _, om1, oa1, ob1, _, _, _ = neural_network(params, z0, i0, m0, 0., 0., 1., t0)
    rv0, _, _, _, rm1, ra1, rb1, _, _, _ = neural_network(params, z0, i0, m0, 0., 0., 0., t0)

    ov1 = neural_network(params, z0, i0, om1, oa1, ob1, 1., t0 + 1)[0]
    rv1 = neural_network(params, z0, i0, rm1, ra1, rb1, 0., t0 + 1)[0]

    v_diff1 = jnp.abs(ov1 - rv1)

    # Bellman error
    loss_bellman = utility + econ_config['beta'] * v1 - v0

    # Value
    loss_value = utility + econ_config['beta'] * v1

    # FOC c:
    loss_focc = uc - l0

    # FOC h:
    loss_foch = uh - econ_config['rh'] * l0

    # FOC n:
    loss_focn = un + econ_config['w'] * l0

    # FOC m1:
    loss_focm1 = (econ_config['beta'] * v1m - l0)

    # FOC a1
    loss_foca1 = o1 * (econ_config['beta'] * v1a - econ_config['pa'] * l0)

    # FOC b1
    loss_focb1 = o1 * (econ_config['beta'] * v1b - l0)

    # FOC m0 (Envelope)
    loss_focm0 = (v0m - l0) # - (1 - o1) * econ_config['beta'] * jnp.isclose(m0, econ_config['phi'] * econ_config['pa']) * v_diff1)

    # FOC a0 (Envelope)
    loss_foca0 = o0 * (v0a - (econ_config['pa'] + econ_config['rh']) * l0)

    # FOC b0 (Envelope)
    loss_focb0 = o0 * (v0b - (1 + i0) * l0)

    # FOC Euler m:
    loss_focem = ((econ_config['beta'] * l1) - l0)

    # FOC Euler a:
    loss_focea = (econ_config['beta'] * l1 * (econ_config['rh'] + econ_config['pa']) - (econ_config['pa'] * l0)) * o0

    # FOC Euler b:
    loss_foceb = (econ_config['beta'] * l1 * (1 + i1) - l0) * o0

    # KKT
    # loss_kkt = l0 * jnp.abs(m1)
    # loss_kktu = l0 * jnp.abs(m1u)
    #
    # Market Clearing
    # loss_bc = c_error + h_error + n_error
    # loss_mc = jnp.exp(-1 * jnp.minimum(m1, 0)) - 1
    # loss_mcu = jnp.exp(-1 * jnp.minimum(m1u, 0)) - 1

    return loss_bellman, loss_value, loss_focc, loss_foch, loss_focn, loss_focm1, loss_foca1, loss_focb1, loss_focm0, loss_foca0, loss_focb0, loss_focem, loss_focea, loss_foceb, v_diff1


@jax.jit
def batch_loss(params, m0, a0, b0, o0, t0, z0, i0, keys):
    loss_bellman0, loss_value0, loss_focc0, loss_foch0, loss_focn0, loss_focm10, loss_foca10, loss_focb10, loss_focm00, loss_foca00, loss_focb00, loss_focem0, loss_focea0, loss_foceb0, v_diff10 = jax.vmap(loss, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, None))(params, m0, a0, b0, o0, t0, z0, i0, keys[0])
    # loss_bellman1, loss_focc1, loss_foch1, loss_focn1, loss_focm11, loss_foca11, loss_focb11, loss_focm01, loss_foca01, loss_focb01 = jax.vmap(loss, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, None))(params, m0, a0, b0, o0, t0, z0, i0, keys[1])

    batch_size = m0.shape[0]
    # return (1 / batch_size) * (jnp.dot(loss_bellman0, loss_bellman1) +
    #                            jnp.dot(loss_focc0, loss_focc1) +
    #                            jnp.dot(loss_foch0, loss_foch1) +
    #                            jnp.dot(loss_focn0, loss_focn1) +
    #                            jnp.dot(loss_focm00, loss_focm01) +
    #                            jnp.dot(loss_foca00, loss_foca01) +
    #                            jnp.dot(loss_focb00, loss_focb01) +
    #                            jnp.dot(loss_focm10, loss_focm11) +
    #                            jnp.dot(loss_foca10, loss_foca11) +
    #                            jnp.dot(loss_focb10, loss_focb11))
    return jnp.log(jnp.mean(loss_bellman0**2 - loss_value0**2 + loss_focc0**2 + loss_foch0**2 + loss_focn0**2 + loss_focm10**2 + loss_foca10**2 + loss_focb10**2 + loss_focm00**2 + loss_foca00**2 + loss_focb00**2 + loss_focem0**2 + loss_focea0**2 + loss_foceb0**2)), \
           {'loss_bellman': jnp.mean(loss_bellman0 ** 2), 'loss_value': jnp.mean(loss_value0 ** 2), 'loss_focc': jnp.mean(loss_focc0 ** 2),
            'loss_foch': jnp.mean(loss_foch0 ** 2), 'loss_focn': jnp.mean(loss_focn0 ** 2), 'loss_focm1': jnp.mean(loss_focm10 ** 2),
            'loss_foca1': jnp.mean(loss_foca10 ** 2), 'loss_focb1': jnp.mean(loss_focb10 ** 2), 'loss_focm0': jnp.mean(loss_focm00 ** 2),
            'loss_foca0': jnp.mean(loss_foca00 ** 2), 'loss_focb0': jnp.mean(loss_focb00 ** 2), 'loss_euler_m': jnp.mean(loss_focem0 ** 2),
            'loss_euler_a': jnp.mean(loss_focea0 ** 2), 'loss_euler_b': jnp.mean(loss_foceb0 ** 2), 'v_diff1': jnp.mean(v_diff10)}
           #(jnp.mean(loss_bellman0**2), jnp.mean(loss_value0**2), jnp.mean(loss_focc0**2), jnp.mean(loss_foch0**2), jnp.mean(loss_focn0**2), jnp.mean(loss_focm10**2), jnp.mean(loss_foca10**2), jnp.mean(loss_focb10**2), jnp.mean(loss_focm00**2), jnp.mean(loss_foca00**2), jnp.mean(loss_focb00**2), jnp.mean(loss_focem0**2), jnp.mean(loss_focea0**2), jnp.mean(loss_foceb0**2))


@jax.jit
def draw_states(z0, i0, key):
    keys = jax.random.split(key, 2)
    z_shock = econ_config['sigma_z'] * jax.random.normal(keys[0])
    i_shock = econ_config['sigma_i'] * jax.random.normal(keys[1])
    z = jnp.exp(econ_config['rho_z'] * jnp.log(z0) + z_shock)
    i = econ_config['rho_i'] * (i0 - econ_config['ibar']) + i_shock + econ_config['ibar']
    return z, i


# def simulate_state_forward(params, econ_config, Xs, Zs, Es, key, n_forward):
#     keys = jax.random.split(key, Zs.shape[0] * n_forward).reshape(n_forward, Zs.shape[0], 2)
#     for i in range(n_forward):
#         Xs = jax.vmap(next_X, in_axes=(None, None, 0, 0, 0))(params, econ_config, Xs, Zs, Es)
#         Zs, Es = jax.vmap(next_state, in_axes=(0, 0, None, 0))(Zs, Es, econ_config, keys[i])
#     return Xs, Zs, Es, keys[-1, -1]


# def generate_random_state(params, econ_config, key, n_forward=0):
#     Zs = jnp.zeros(shape=(n // 2,))
#     Es = jnp.zeros(shape=(n // 2, k))
#     Xs = jnp.exp(a * jax.random.normal(jax.random.PRNGKey(np.random.randint(1, int(1e8))), shape=(n // 2, k))) + b
#     if n_forward > 0:
#         Xs, Zs, Es, key = simulate_state_forward(params, econ_config, Xs, Zs, Es, key, n_forward)

#     return Xs, Zs, Es, key


scale = ml_config['init_scale']
init_keys = jax.random.split(jax.random.PRNGKey(ml_config['init_seed']), 11)
w00o = scale * jax.random.normal(init_keys[1], shape=(6, ml_config['nn_shapes'][0]))
w00r = scale * jax.random.normal(init_keys[1], shape=(6, ml_config['nn_shapes'][0]))
w01 = scale * jax.random.normal(init_keys[2], shape=(ml_config['nn_shapes'][0], ml_config['nn_shapes'][1]))
w02 = scale * jax.random.normal(init_keys[3], shape=(ml_config['nn_shapes'][1], ml_config['nn_shapes'][2]))
w03 = scale * jax.random.normal(init_keys[3], shape=(ml_config['nn_shapes'][2], ml_config['nn_shapes'][3]))
w0f = scale * jax.random.normal(init_keys[4], shape=(ml_config['nn_shapes'][1], 1))
b00 = scale * jax.random.normal(init_keys[5], shape=(1, ml_config['nn_shapes'][0]))
b01 = scale * jax.random.normal(init_keys[6], shape=(1, ml_config['nn_shapes'][1]))
b02 = scale * jax.random.normal(init_keys[7], shape=(1, ml_config['nn_shapes'][2]))
b03 = scale * jax.random.normal(init_keys[7], shape=(1, ml_config['nn_shapes'][3]))
b0f = scale * jax.random.normal(init_keys[8], shape=(1, 1))
params0 = {
    'owner': {
        'w0': w00o, 'w1': w01, 'w2': w02, 'w3': w03, 'cwf': w0f, 'nwf': w0f, 'swf': w0f, 'dwf': w0f, 'bwf': w0f, 'iwf': w0f, 'vwf': w0f, 'lwf': w0f,
        'b0': b00, 'b1': b01, 'b2': b02, 'b3': b03, 'cbf': b0f, 'nbf': b0f, 'sbf': b0f, 'dbf': b0f, 'bbf': b0f, 'ibf': b0f, 'vbf': b0f, 'lbf': b0f
    },
    'renter': {
        'w0': w00r, 'w1': w01, 'w2': w02, 'w3': w03, 'cwf': w0f, 'nwf': w0f, 'swf': w0f, 'dwf': w0f, 'bwf': w0f, 'iwf': w0f, 'vwf': w0f, 'lwf': w0f,
        'b0': b00, 'b1': b01, 'b2': b02, 'b3': b03, 'cbf': b0f, 'nbf': b0f, 'sbf': b0f, 'dbf': b0f, 'bbf': b0f, 'ibf': b0f, 'vbf': b0f, 'lbf': b0f
    }
}


def training_loop(opt_state, tol=1e-5, max_iter=10 ** 4, batch_size=32):
    j = 0
    key = jax.random.PRNGKey(np.random.randint(1, int(1e8)))
    val_loss = jnp.inf
    grad = {'0': jnp.inf}
    opt_init, opt_update, get_params = adam(step_size=ml_config['step_size'])
    params = get_params(opt_state)

    m = jnp.concatenate((jnp.linspace(econ_config['phi'] * econ_config['pa'] - 1e-10, econ_config['phi'] * econ_config['pa'], 10), jnp.linspace(0, 2, 91)))
    a = jnp.linspace(0, 1, 101)
    b = jnp.linspace(0, 1, 101)
    z = jnp.ones(101) # jnp.linspace(0.5, 1.5, 100)
    i = econ_config['ibar'] * jnp.ones(101) # jnp.linspace(0.01, 0.1, 100)
    o = jnp.concatenate((jnp.zeros(50), jnp.ones(51)))
    t = jnp.concatenate((jnp.repeat(jnp.arange(econ_config['T'] + 1), 100 // (econ_config['T'])), econ_config['T'] * jnp.ones(1)))

    while j < max_iter:
        sample = jax.random.choice(key, jnp.arange(101), shape=(7, batch_size))
        keys = jax.random.split(key, 2)
        key = keys[-1]
        val, grad = jax.value_and_grad(batch_loss, has_aux=True)(get_params(opt_state), m[sample[0]], a[sample[1]], b[sample[2]], o[sample[3]], t[sample[4]], z[sample[5]], i[sample[6]], keys)
        val_loss = val[0]
        if any([jnp.isnan(val).any() for val in grad["owner"].values() for key in grad.keys()] + [jnp.isnan(val).any() for val in grad["renter"].values() for key in grad.keys()]):
            print('Warning: grad was nan, continuing ...')
            continue
        opt_state = opt_update(j, grad, opt_state)
        if j % ml_config['report_interval'] == 0:
            print(f'Iteration: {j}\tTotal Loss: {val_loss}' +\
                  f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in (grad["owner"] | grad["renter"]).items()])}' +\
                  f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in (params["owner"] | params["renter"]).items()])}' +\
                  f'\nLoss Components: {val[1]}')
        if j % ml_config['save_interval'] == 0:
            os = unpack_optimizer_state(opt_state)
            # pickle.dump(os, open(f'./share/models/housing_model_{j}.pkl', 'wb'))
            pickle.dump(os, open(f'./share/models/housing_model_latest.pkl', 'wb'))
        j += 1

    print(f'Terminating training with final statistics:\n' + \
          f'Iteration: {j}\tTotal Loss: {val_loss}' + \
          f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in (grad["owner"] | grad["renter"]).items()])}' + \
          f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in (params["owner"] | params["renter"]).items()])}')
    return opt_state


def main():
    opt_init, opt_update, get_params = adam(step_size=ml_config['step_size'])
    if ml_config['train_new']:
        opt_state = opt_init(params0)
    else:
        saved_params = pickle.load(open(f'./share/models/housing_model_{ml_config["start_from"]}.pkl', 'rb'))
        opt_state = pack_optimizer_state(saved_params)

    opt_state = training_loop(opt_state, max_iter=ml_config['n_epoch'], batch_size=ml_config['mb'])
    params = unpack_optimizer_state(opt_state)
    pickle.dump(params, open(f'./share/models/housing_model.pkl', 'wb'))


if __name__ == '__main__':
    main()