import numpy as np
import pandas as pd
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
    'beta': 0.8,
    'p': 1.,
    'T': 2,
}

# ML parameters
ml_config = {
    'mb': 2 ** 4,
    'n_epoch': 10 ** 5,
    'save_interval': 10 ** 2,
    'report_interval': 10,
    'step_size': 1e-2,
    'nn_shapes': jnp.array([16, 16, 16, 16]),
    'init_seed': 1234, # np.random.randint(1, 10000000)
    'init_scale': 1e-1,
    'train_new': False,
    'start_from': "latest"
}

@jax.jit
def neural_network(params, m0, t0):
    live_next = jnp.int32(t0 < econ_config['T'])
    alive = jnp.int32(t0 < econ_config['T'] + 1)
    cparams = params['cparams']
    vparams = params['vparams']
    lparams = params['lparams']

    X_tilde = jnp.concatenate([m0.reshape(1, -1),
                               t0.reshape(1, -1)], axis=1)
    cl1 = relu(X_tilde, cparams['w0'], cparams['b0'])
    clf = relu(cl1, cparams['w1'], cparams['b1'])
    # clf = relu(cl2, params['cw2'], params['cb2'])

    vl1 = relu(X_tilde, vparams['w0'], vparams['b0'])
    vlf = relu(vl1, vparams['w1'], vparams['b1'])
    # vlf = relu(vl2, params['vw2'], params['vb2'])

    # c0, m1 = live_next * m0 * softmax(clf, cparams['wf'], cparams['bf']) + (1 - live_next) * jnp.array([m0 - 1e-10, 1e-10])
    # l0 = live_next * 1 / c0
    # c0, m1 = m0 * softmax(clf, cparams['wf'], cparams['bf'])
    c0 = m0 * scaled_sigmoid(clf, cparams['wf'], cparams['bf'])
    # c0 = jnp.maximum(c0, 1e-10) # We can't let c0 be equal to zero or the multiplier is inf
    m1 = m0 - c0
    l0 = exp(clf, lparams['wf'], lparams['bf'])

    v0 = custom_value_fn(vlf, vparams['wf'], vparams['bf']) * alive
    bc = m0 - (c0 + m1)
    return (v0, c0, m1, bc, l0)


@jax.jit
def loss(params, m0, t0, key):
    v0, c0, m1, bc0, l0 = neural_network(params, m0, t0)
    v1, c1, m2, bc1, l1 = neural_network(params, m1, t0 + 1)

    utility = lambda c: jnp.log(c)
    u = utility(c0)
    uc = jax.grad(utility)(c0)

    vf = lambda m, t: neural_network(params, m, t)[0]
    v0m = jax.grad(vf, 0)(m0, t0)
    v1m = jax.grad(vf, 1)(m1, t0 + 1)

    live_next = jnp.int32(t0 < econ_config['T'])
    loss_euler = (((econ_config['beta'] * l1) / l0) - 1) * live_next
    loss_bellman = ((u + econ_config['beta'] * v1) - v0)
    loss_focm1 = (((econ_config['beta'] * v1m) / l0) - 1)
    loss_focm0 = ((l0 / v0m) - 1)
    loss_value = 0 # u + econ_config['beta'] * v1
    loss_focc = (uc / l0) - 1

    return loss_bellman, loss_value, loss_focc, loss_focm1, loss_focm0, loss_euler


@jax.jit
def batch_loss(params, m0, t0, keys):
    loss_bellman, loss_value, loss_focc, loss_focm1, loss_focm0, loss_euler = jax.vmap(loss, in_axes=(None, 0, 0, None))(params, m0, t0, keys[0])
    return jnp.log(jnp.mean(loss_bellman**2 - loss_value**2 + loss_focc**2 + loss_focm1**2 + loss_focm0**2 + loss_euler**2)), \
           {'loss_bellman': jnp.mean(loss_bellman ** 2), 'loss_value': jnp.mean(loss_value ** 2), 'loss_focc': jnp.mean(loss_focc ** 2),
            'loss_focm1': jnp.mean(loss_focm1 ** 2), 'loss_focm0': jnp.mean(loss_focm0 ** 2), 'loss_euler': jnp.mean(loss_euler ** 2)}


def flatten(dict):
    df = pd.json_normalize(dict, sep='_')
    return df.to_dict(orient='records')[0]


scale = ml_config['init_scale']
init_keys = jax.random.split(jax.random.PRNGKey(ml_config['init_seed']), 11)
w00 = scale * jax.random.normal(init_keys[1], shape=(2, ml_config['nn_shapes'][0]))
w01 = scale * jax.random.normal(init_keys[2], shape=(ml_config['nn_shapes'][0], ml_config['nn_shapes'][1]))
w02 = scale * jax.random.normal(init_keys[3], shape=(ml_config['nn_shapes'][1], ml_config['nn_shapes'][2]))
w03 = scale * jax.random.normal(init_keys[3], shape=(ml_config['nn_shapes'][2], ml_config['nn_shapes'][3]))
w0f = scale * jax.random.normal(init_keys[4], shape=(ml_config['nn_shapes'][1], 1))
cw0f = scale * jax.random.normal(init_keys[4], shape=(ml_config['nn_shapes'][1], 1))
b00 = scale * jax.random.normal(init_keys[5], shape=(1, ml_config['nn_shapes'][0]))
b01 = scale * jax.random.normal(init_keys[6], shape=(1, ml_config['nn_shapes'][1]))
b02 = scale * jax.random.normal(init_keys[7], shape=(1, ml_config['nn_shapes'][2]))
b03 = scale * jax.random.normal(init_keys[7], shape=(1, ml_config['nn_shapes'][3]))
b0f = scale * jax.random.normal(init_keys[8], shape=(1, 1))
cb0f = scale * jax.random.normal(init_keys[8], shape=(1, 1))

params0 = {
    'cparams': {
        'w0': w00, 'w1': w01, 'w2': w02, 'w3': w03, 'wf': cw0f,
        'b0': b00, 'b1': b01, 'b2': b02, 'b3': b03, 'bf': cb0f,
    },
    'vparams': {
        'w0': w00, 'w1': w01, 'w2': w02, 'w3': w03, 'wf': w0f,
        'b0': b00, 'b1': b01, 'b2': b02, 'b3': b03, 'bf': b0f,
    },
    'lparams': {
        'wf': w0f,
        'bf': b0f
    }
}


def training_loop(opt_state, tol=1e-5, max_iter=10 ** 4, batch_size=32):
    j = 0
    key = jax.random.PRNGKey(np.random.randint(1, int(1e8)))
    val_loss = jnp.inf
    grad = {'0': jnp.inf}
    opt_init, opt_update, get_params = adam(step_size=ml_config['step_size'])
    params = get_params(opt_state)

    m = jnp.linspace(1e-10, 2, 101)
    t = jnp.concatenate((jnp.zeros(33), jnp.ones(34), 2 * jnp.ones(33)))

    while j < max_iter:
        sample = jax.random.choice(key, jnp.arange(101), shape=(2, batch_size))
        keys = jax.random.split(key, 2)
        key = keys[-1]

        val, grad = jax.value_and_grad(batch_loss, has_aux=True)(get_params(opt_state), m[sample[0]], t[sample[1]], keys)
        val_loss = val[0]
        if any([jnp.isnan(val).any() for val in flatten(grad).values() for key in flatten(grad).keys()]):
            print('Warning: grad was nan, will apply nan_to_num')
            # for z in grad.keys():
            #     grad[z] = {a: jnp.nan_to_num(b) for a, b in grad[z].items()}
            continue
        opt_state = opt_update(j, grad, opt_state)

        if j % ml_config['report_interval'] == 0:
            print(f'Iteration: {j}\tTotal Loss: {val_loss}' +\
                  f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in flatten(grad).items()])}' +\
                  f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in flatten(params).items()])}' +\
                  f'\nLoss Components: {val[1]}')
        if j % ml_config['save_interval'] == 0:
            os = unpack_optimizer_state(opt_state)
            # pickle.dump(os, open(f'./share/models/renter_model_{j}.pkl', 'wb'))
            pickle.dump(os, open(f'./share/models/very_simple_model_latest.pkl', 'wb'))
        j += 1

    print(f'Terminating training with final statistics:\n' + \
          f'Iteration: {j}\tTotal Loss: {val_loss}' +\
          f'\tMax Grad: {max([jnp.max(jnp.abs(v)) for k, v in flatten(grad).items()])}' + \
          f'\tMax Param: {max([jnp.max(jnp.abs(v)) for k, v in flatten(params).items()])}')
    return opt_state


def main():
    opt_init, opt_update, get_params = adam(step_size=ml_config['step_size'])
    if ml_config['train_new']:
        opt_state = opt_init(params0)
    else:
        saved_params = pickle.load(open(f'./share/models/very_simple_model_{ml_config["start_from"]}.pkl', 'rb'))
        opt_state = pack_optimizer_state(saved_params)

    opt_state = training_loop(opt_state, max_iter=ml_config['n_epoch'], batch_size=ml_config['mb'])
    params = unpack_optimizer_state(opt_state)
    pickle.dump(params, open(f'./share/models/very_simple_model.pkl', 'wb'))


if __name__ == '__main__':
    main()