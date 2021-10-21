import jax
import jax.numpy as jnp
from jax.scipy import optimize


def cobb_douglas(A, alpha, rts=1):
    return lambda inputs: jnp.prod(
        jnp.concatenate([jnp.array([A]), jnp.power(inputs, jnp.array([alpha * rts, (1 - alpha) * rts]))]))


def hd_symmetric_cd(A, rts=1):
    return lambda inputs: jnp.prod(jnp.concatenate(
        [jnp.array([A]), jnp.power(inputs, jnp.repeat(jnp.array([rts / inputs.shape[0]]), inputs.shape[0]))]))


@jax.jit
def normalize(x):
    return x / jnp.sum(x)


def firm_optimal_production(firm):
    @jax.jit
    def ret_fn(prices):
        A = firm[0]
        rts = firm[1]
        output_idx = jnp.int32(firm[2])

        production_function = hd_symmetric_cd(A, rts)
        profit = lambda inputs, input_prices, output_prices: \
            jnp.multiply(
                jnp.subtract(jnp.dot(output_prices, production_function(inputs)),
                             jnp.dot(input_prices, inputs)),
                -1)
        input_prices = prices[
            jnp.where(~jnp.isin(jnp.arange(prices.shape[0]), output_idx), size=(prices.shape[0] - 1))[0]]
        output_prices = prices[output_idx]
        inputs = jnp.ones(input_prices.shape[0]) / input_prices.shape[0]

        inputs = optimize.minimize(profit, inputs, (input_prices, output_prices), method='BFGS').x
        outputs = production_function(inputs).reshape(-1)
        exp = jnp.multiply(input_prices, inputs)
        pi = jnp.multiply(-1, profit(inputs, input_prices, output_prices))
        pi = jnp.clip(pi, jnp.zeros(pi.shape), None)

        io_vec = jnp.squeeze(jnp.multiply(jnp.concatenate((outputs, inputs)), (pi > 0).astype(jnp.int8).reshape(-1, 1)))
        returns = ((jnp.multiply(pi, normalize(exp)) / exp))
        return io_vec, returns

    return ret_fn


@jax.jit
def supply(prices, firms):
    io_vec, rets = jax.vmap(lambda firm: firm_optimal_production(firm)(prices))(firms)
    agg_io = jnp.sum(io_vec, axis=0)
    R = jnp.sum(jnp.mean(rets, axis=0))
    return agg_io, R