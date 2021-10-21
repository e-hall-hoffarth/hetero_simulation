import argparse
import jax
import jax.numpy as jnp
from optimization import *
import time
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--key', type=int, default=1, help='Random seed')
parser.add_argument('--theta', type=float, default=10., help='Parameter of gamma distribution from which firm tfp is drawn')
parser.add_argument('--alpha', type=float, default=5., help='Alpha parameter of beta distribution from which firm rts is drawn')
parser.add_argument('--beta', type=float, default=5., help='Beta parameter of beta distribution from which firm rts is drawn')
parser.add_argument('--mean_sigma', type=float, default=2., help='Average sigma parameter of agents')
parser.add_argument('--var_sigma', type=float, default=0.1, help='Variance of sigma parameter')
parser.add_argument('--scale', type=float, default=3., help='Scale parameter of log-normal distribution from which agnet wealth is drawn')
parser.add_argument('--n_firms', type=int, default=2, help='Number of firms')
parser.add_argument('--n_agents', type=int, default=1000, help='Number of agents')
parser.add_argument('--max_iter', type=int, default=1000, help='Max iterations')
parser.add_argument('--n_products', type=int, default=3, help='Total number of products')
parser.add_argument('--n_assets', type=int, default=2, help='Number of products which are assets owned by agents')
parser.add_argument('--T', type=int, default=100, help='Time horizon of agents')
parser.add_argument('--step_size', type=float, default=1e-2, help='Step size in SGD')
parser.add_argument('--tol', type=float, default=1e-8, help='Tolerance in SGD')
args = parser.parse_args()


def main():
    start_time = time.time()
    key = jax.random.PRNGKey(args.key)
    eql_log_prices = find_equilibrium_prices(args.n_products, args.n_assets, args.n_firms, args.n_agents, args.theta,
                                             args.alpha, args.beta, args.mean_sigma, args.var_sigma, args.scale, args.T,
                                             args.max_iter, args.step_size, args.tol, key)
    end_time = time.time()
    print(f'Time elapsed: {timedelta(seconds=(end_time - start_time))}')
    return jnp.exp(eql_log_prices)


if __name__ == '__main__':
    main()