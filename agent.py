import jax
import jax.numpy as jnp
from jax.scipy import optimize


def disc_log_utility(B):
    return lambda c, t: jnp.multiply(jnp.power(B, t), jnp.sum(jnp.log(c)))


def disc_ces_utility(B, sigma):
    return lambda c, t: jnp.multiply(jnp.power(B, t),
                                        jnp.power(
                                            jnp.sum(jnp.power(c, ((sigma - 1)/sigma))),
                                            (sigma/(sigma - 1))
                                     ))


@jax.tree_util.Partial(jax.jit, static_argnums=(2,))
def jit_bellman(c, sigma, T, R, cp):
    u = disc_ces_utility(1/R, sigma)
    cv = jnp.dot(c.reshape(T, -1), cp)
    R_t = jnp.power(R, jnp.arange(T)).reshape(-1, 1)
    cpv = cv / R_t
    ts = jnp.arange(T)
    utility = jnp.sum(jax.vmap(u)(cpv, ts))
    return -utility


def agent_optimal_choice(n, T, agent, R):
    @jax.jit
    def ret_fn(ap, cp):
        av = jnp.dot(ap, agent[:-1])
        r = R - 1
        apv = ((1-(r**T))/(1-r)) * av
        c = jnp.ones((T * n,))
        optim_result_scaled = optimize.minimize(jit_bellman, c, (agent[-1], T, R, cp), method='BFGS')
        result_scaled = optim_result_scaled.x.reshape(T, -1)

        result = (result_scaled * (apv / jnp.sum(jnp.dot(result_scaled, cp))))
        return result
    return ret_fn


@jax.tree_util.Partial(jax.jit, static_argnums=(3, 4))
def demand(R, prices, agents, n_goods, T):
    demand = jax.vmap(lambda a, b, c, d: agent_optimal_choice(a, b, c, d)(prices[n_goods:], prices[:n_goods]),
                      in_axes=(None, None, 0, None))(n_goods, T, agents, R)
    agg_demand = jnp.sum(demand[:, 0, :], axis=0)
    return agg_demand