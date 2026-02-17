import numpy as np

def simulate_gbm(S0, mu, sigma, T, N, n_paths, leverage, S_power=1):
    # simulate n_paths many paths from 0 to T of GBM with parameters mu and sigma
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

    S_end = np.exp(S_power * logS[:, N])
    L_end = np.exp(logL[:, N])
    L_end[np.isnan(L_end)] = 0

    return L_end - S_end