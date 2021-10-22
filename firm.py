import jax
import jax.numpy as jnp
from jax.scipy import optimize


def cobb_douglas(A, alpha, rts):
    return lambda inputs: jnp.prod(
        jnp.concatenate([jnp.array([A]), jnp.power(inputs, jnp.array([alpha * rts, (1 - alpha) * rts]))]))


def hd_symmetric_cd(A, rts):
    return lambda inputs: jnp.nan_to_num(jnp.prod(jnp.concatenate(
        [jnp.array([A]), jnp.power(inputs, jnp.repeat(jnp.array([rts / inputs.shape[0]]), inputs.shape[0]))])))


@jax.jit
def normalize(x):
    return jnp.nan_to_num(x / jnp.sum(x), nan=jnp.ones(x.shape[0])/x.shape[0])


def firm_optimal_production(firm):
    @jax.jit
    def ret_fn(prices):
        A = firm[0]
        rts = firm[1]
        fc = firm[2]
        output_idx = jnp.int32(firm[3])

        production_function = hd_symmetric_cd(A, rts)
        profit = lambda ln_inputs, input_prices, output_prices: \
            jnp.multiply(
                jnp.subtract(
                    jnp.subtract(jnp.dot(output_prices, production_function(jnp.exp(ln_inputs))),
                                 jnp.dot(input_prices, jnp.exp(ln_inputs))),
                    fc),
                -1)
        input_prices = prices[
            jnp.where(~jnp.isin(jnp.arange(prices.shape[0]), output_idx), size=(prices.shape[0] - 1))[0]]
        output_prices = prices[output_idx]
        ln_inputs = jnp.ones(input_prices.shape[0]) / input_prices.shape[0]

        ln_inputs = optimize.minimize(profit, ln_inputs, (input_prices, output_prices),
                                      method='BFGS', options={'line_search_maxiter': 10000, 'gtol': 1e-2}).x
        outputs = production_function(ln_inputs).reshape(-1)

        pi = jnp.multiply(-1, profit(ln_inputs, input_prices, output_prices))
        pi = jnp.clip(pi, jnp.zeros(pi.shape), None)

        # Firm exits production if its optimal production results in a loss
        profitable = (pi > 0)
        inputs = jnp.multiply(jnp.exp(ln_inputs), profitable)
        outputs = jnp.multiply(outputs, profitable)
        exp = jnp.multiply(jnp.multiply(input_prices, inputs) + fc, profitable)

        io_vec = jnp.squeeze(jnp.concatenate((outputs, inputs)))
        returns = jnp.multiply(jnp.nan_to_num(pi / jnp.sum(exp)) + 1, normalize(exp))
        return io_vec, returns

    return ret_fn


@jax.jit
def supply(prices, firms):
    io_vec, rets = jax.vmap(lambda firm: firm_optimal_production(firm)(prices))(firms)
    agg_io = jnp.sum(io_vec, axis=0)
    R = jnp.mean(jnp.sum(rets, axis=1))
    return agg_io, R