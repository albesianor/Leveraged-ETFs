'''
GBM simulation and dataset generation functions
'''

import numpy as np
import pandas as pd

def simulate_gbm(S0=1, mu=0, sigma=0.1, T=5, N=None, n_paths=10_000, leverage=2, return_mean_std=False):
    '''
    Simulate P&L of n_paths GBM portfolios of 1 long leveraged ETF and 1 short underlying

    Input:
    S0: starting price of underlying (default=1)
    mu: drift (default=0)
    sigma: volatility (default=0.4)
    T: time in days (default=5)
    N: number of updates (default=T)
    n_paths: number of simulated paths (default=10_000)
    leverage: leverage (default=2)
    return_mean_std: if true, returns mean and std, otherwise returns distribution

    Output:
    Distribution of the portfolio at time T: L_end - S_end
    '''

    if N is None:
        N = T
    dt = T / N
    times = np.linspace(0, T, N + 1)

    Z = np.random.randn(n_paths, N)

    drift = (mu - (1 / 2) * (sigma**2)) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    logS = np.zeros((n_paths, N + 1))
    logS[:, 0] = np.log(S0)
    logS[:, 1:] = np.cumsum(drift + diffusion, axis=1) + logS[:, :1]

    # leveraged log-returns
    logL = np.zeros((n_paths, N + 1))
    logL[:, 0] = logS[:, 0]
    logL[:, 1:] = (
        np.cumsum(np.log((np.exp(drift + diffusion) - 1) * leverage + 1), axis=1)
        + logL[:, :1]
    )

    S_end = np.exp(logS[:, N])
    L_end = np.exp(logL[:, N])
    L_end[np.isnan(L_end)] = 0

    PnL = L_end - S_end

    if return_mean_std:
        return PnL.mean(), PnL.std()

    return PnL


def mean_diff(S0=1, mu=0, sigma=0.1, T=5, N=None, n_paths=10_000, leverage=2):
    '''
    Simulate mean return of n_paths GBM portfolios of 1 long leveraged ETF and 1 short underlying

    Input:
    S0: starting price of underlying (default=1)
    mu: drift (default=0)
    sigma: volatility (default=0.4)
    T: time in days (default=5)
    N: number of updates (default=T)
    n_paths: number of simulated paths (default=10_000)
    leverage: leverage (default=2)

    Output:
    Mean return of the portfolio at time T: L_end - S_end
    '''
    return simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, N=N, n_paths=n_paths, leverage=leverage).mean()


def std_diff(S0=1, mu=0, sigma=0.1, T=5, N=None, n_paths=10_000, leverage=2):
    '''
    Simulate P&L std of n_paths GBM portfolios of 1 long leveraged ETF and 1 short underlying

    Input:
    S0: starting price of underlying (default=1)
    mu: drift (default=0)
    sigma: volatility (default=0.4)
    T: time in days (default=5)
    N: number of updates (default=T)
    n_paths: number of simulated paths (default=10_000)
    leverage: leverage (default=2)

    Output:
    Standard deviation of returns of the portfolio at time T: L_end - S_end
    '''
    return simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, N=N, n_paths=n_paths, leverage=leverage).std()


def lev_sigma_mean_std(
    mu=0,
    sigmas=np.linspace(0.01, 0.4, 100),
    T=5,
    freq=1,
    n_paths=10_000,
    leverages=np.linspace(1.01, 3, 100),
):
    '''
    Simulate P&L mean and std of n_paths of GBM portfolios of 1 long leveraged ETF and 1 short underlying,
    with volatility ranging in sigmas and leverage ranging in leverages.

    Input:
    mu: drift (default=0)
    sigmas: range of volatilities (default=np.linspace(0.01, 0.4, 100))
    T: time in days (default=5)
    freq: updates/day (default=1)
    n_paths: number of simulated paths (default=10_000)
    leverages: range of leverage factors (default=np.linspace(1.01, 3, 100))

    Output:
    Pandas dataframe with columns leverage, sigma, mean of P&L, std of P&L
    '''

    leverage, sigma = np.meshgrid(leverages, sigmas, indexing="ij")

    N = None
    if freq != 1:
        N = int(T*freq)

    mean, std = np.vectorize(simulate_gbm)(S0=1, mu=mu, sigma=sigma, T=T, N=N, n_paths=n_paths, leverage=leverage, return_mean_std=True)

    return pd.DataFrame(
        {"leverage": leverage.ravel(), "sigma": sigma.ravel(), "mean": mean.ravel(), "std": std.ravel()}
    )