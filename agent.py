import jax
import jax.numpy as jnp
from jax.scipy import optimize


def log_utility(B):
    return lambda c, t: jnp.multiply(jnp.power(B, t), jnp.log(c[0]))


def disc_log_utility(B):
    return lambda c, t: jnp.multiply(jnp.power(B, t), jnp.sum(jnp.log(c)))


@jax.tree_util.Partial(jax.jit, static_argnums=(1,))
def jit_bellman(c, T, R, cp):
    u = disc_log_utility(1/R)
    cv = jnp.dot(c.reshape(T, -1), cp)
    R_t = jnp.power(R, jnp.arange(T)).reshape(-1, 1)
    cpv = cv / R_t
    ts = jnp.arange(T)
    utility = jnp.sum(jax.vmap(u)(cpv, ts))
    return -utility


def agent_optimal_choice(n, T, a, R):
    @jax.jit
    def ret_fn(ap, cp):
        av = jnp.dot(ap, a)
        r = R - 1
        apv = ((1-(r**T))/(1-r)) * av
        c = jnp.ones((T * n,))
        optim_result_scaled = optimize.minimize(jit_bellman, c, (T, R, cp), method='BFGS')
        result_scaled = optim_result_scaled.x.reshape(T, -1)

        result = (result_scaled * (apv / jnp.sum(jnp.dot(result_scaled, cp))))
        return result
    return ret_fn


@jax.tree_util.Partial(jax.jit, static_argnums=(3, 4))
def demand(R, prices, agents, n_goods, T):
    demand =  jax.vmap(lambda a, b, c, d: agent_optimal_choice(a, b, c, d)(prices[n_goods:], prices[:n_goods]), in_axes=(None, None, 0, None))(n_goods, T, agents, R)
    agg_demand = jnp.sum(demand[:, 0, :], axis=0)
    return agg_demand