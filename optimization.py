import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from firm import *
from agent import *


def error(firms, agents, n_goods, T):
    @jax.jit
    def ret_fn(log_prices):
        prices = jnp.exp(log_prices)
        s = supply(prices, firms)
        R = s[1]
        s0 = s[0][0]
        d0 = demand(R, prices, agents, n_goods, T)
#         a = jnp.sum(agents, axis=1)
#         d = jnp.concatenate((d, a))
        return jnp.linalg.norm(d0 - s0)**2
    return ret_fn


def find_equilibrium_prices(n_products, n_assets, n_firms, n_agents, theta, alpha, beta, scale, T,
                            max_iter, step_size, tol, key):
    n_goods = n_products - n_assets
    i = 0
    err = jnp.inf
    grad = jnp.inf
    opt_init, opt_update, get_params = adam(step_size=step_size)

    firms = (jax.random.gamma(key, theta, shape=(n_firms,)),
             jax.random.beta(key, alpha, beta, shape=(n_firms,)),
             jnp.zeros((n_firms, 1)).astype(jnp.int32))

    assets = jnp.exp(scale * jax.random.normal(key, (n_agents, n_assets)))
    sigmas = 2 + 0.1 * jax.random.normal(key, (n_agents, 1))
    agents = jnp.concatenate((assets, sigmas), axis=1)

    log_prices = jnp.zeros((n_products,))
    e = error(firms, agents, n_goods, T)

    opt_state = opt_init(log_prices)

    while i < max_iter and err > tol:
        log_prices = get_params(opt_state)
        grad = jax.jacfwd(e)(log_prices)
        err = e(log_prices)
        if jnp.isnan(grad).any():
            print(f'Grad in iteration {i} is nan, terminating')
            break
        opt_state = opt_update(i, grad, opt_state)
        if i % 10 == 0:
            print(f'iteration: {i}\nlog_prices: {jnp.exp(log_prices)}\ngradient: {grad}\nerror: {e(log_prices)}')

        i += 1

    print('Final results')
    print(f'iteration: {i}\nprices: {jnp.exp(log_prices)}\ngradient: {grad}\nerror: {e(log_prices)}')

    return log_prices