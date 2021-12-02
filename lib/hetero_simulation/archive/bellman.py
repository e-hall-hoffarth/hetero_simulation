import time
import pickle
import jax
import jax.numpy as jnp
from jax.scipy import optimize
from jax.experimental.optimizers import adam, unpack_optimizer_state, pack_optimizer_state
from agent import log_utility, ces_utility


def V_hat(params):
    @jax.jit
    def f(X):
        X_tilde = jnp.dot(X, params['theta'])
        l1 = jnp.clip(jnp.dot(X_tilde, params['w0']) + params['b0'], 0)
        l2 = jnp.clip(jnp.dot(l1, params['w1']) + params['b1'], 0)
        l3 = jnp.clip(jnp.dot(l2, params['w2']) + params['b2'], 0)
        return jnp.squeeze(jnp.dot(l3, params['wf']) + params['bf'])

    return f


@jax.tree_util.Partial(jax.jit, static_argnums=(2,))
def future(params, beta, c_shape, X):
    v_hat = V_hat(params)
    u = ces_utility(2.)

    f = lambda c: - (u(c) + beta * v_hat(X - jnp.sum(c)))
    c0 = jnp.ones(c_shape)

    c_star = optimize.minimize(f, c0, method='BFGS', options={'line_search_maxiter': 10000, 'gtol': 1e-2}).x

    return u(c_star) + beta * v_hat(X - jnp.sum(c_star))


@jax.tree_util.Partial(jax.jit, static_argnums=(2,))
def epsilon(params, beta, c_shape, X):
    v_hat = V_hat(params)

    f = future(params, beta, c_shape, X)
    v = v_hat(X)

    return (v - f) ** 2


def main():
    c_shape = 1
    k = 1000
    m = 10
    nn_shapes = jnp.array([10, 20, 10])
    beta = 0.95
    e = lambda params: epsilon(params, beta, c_shape, X)

    X = jnp.exp(jax.random.normal(jax.random.PRNGKey(123), shape=(1, k)))
    theta0 = jax.random.normal(jax.random.PRNGKey(129), shape=(k, m))
    w00 = jax.random.normal(jax.random.PRNGKey(6), shape=(m, nn_shapes[0]))
    w01 = jax.random.normal(jax.random.PRNGKey(7), shape=(nn_shapes[0], nn_shapes[1]))
    w02 = jax.random.normal(jax.random.PRNGKey(8), shape=(nn_shapes[1], nn_shapes[2]))
    w0f = jax.random.normal(jax.random.PRNGKey(9), shape=(nn_shapes[2], 1))
    b00 = jax.random.normal(jax.random.PRNGKey(52), shape=(1, nn_shapes[0]))
    b01 = jax.random.normal(jax.random.PRNGKey(51), shape=(1, nn_shapes[1]))
    b02 = jax.random.normal(jax.random.PRNGKey(58), shape=(1, nn_shapes[2]))
    b0f = jax.random.normal(jax.random.PRNGKey(48), shape=(1, 1))
    params0 = {'theta': theta0, 'w0': w00, 'w1': w01, 'w2': w02, 'wf': w0f, 'b0': b00, 'b1': b01, 'b2': b02, 'bf': b0f}

    opt_init, opt_update, get_params = adam(step_size=0.01)
    opt_state = opt_init(params0)

    i = 0
    tol = 1e-8
    err = jnp.inf
    st = time.time()

    while err > tol:
        params = get_params(opt_state)
        grad = jax.jacfwd(e)(params)
        err = e(params)
        if any([jnp.isnan(v).any() for k, v in grad.items()]):
            print(f'Grad in iteration {i} is nan, terminating')
            break
        opt_state = opt_update(i, grad, opt_state)
        if i % 10 == 0:
            print(f'iteration: {i}\nerror: {err}\ntime elapsed: {time.time() - st}')

        i += 1

    trained_params = unpack_optimizer_state(opt_state)
    pickle.dump(trained_params, open('trained_params.pkl', 'wb'))


if __name__ == '__main__':
    main()