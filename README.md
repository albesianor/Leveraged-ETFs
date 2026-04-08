# Leveraged ETFs

This repository is an exploratory research project on the short-horizon behavior of leveraged ETFs compared to their underlying asset. The core object of study is a portfolio that is long one leveraged ETF and short one unit of the underlying, with the goal of understanding how leverage, volatility, rebalancing frequency, and holding horizon affect the portfolio's return distribution and, in particular, its standard deviation.

In this project we conduct:

- geometric Brownian motion (GBM) simulation,
- exploratory analysis,
- interpretable modeling, and
- historical backtesting on ETF price data from `yfinance`.

## Main idea

Across the synthetic experiments and the historical backtests, the clearest and most stable signal is in the standard deviation of portfolio P&L rather than in mean P&L. The modeling work in this repo eventually focuses on a simple relationship of the form


$$\sigma_{\text{portfolio}} \approx a (L - 1) \sigma \sqrt{d},$$

where:

- `d` is the holding horizon in trading days,
- `sigma` is the annualized volatility of the underlying,
- `L` is the leverage factor, and
- `a` is a fitted constant.

## Note on path dependence

We are aware of Avellaneda--Zhang's paper, [*Path-Dependence of Leveraged ETF Returns*](https://nyuscholars.nyu.edu/en/publications/path-dependence-of-leveraged-etf-returns) (SIAM Journal on Financial Mathematics, 2010, DOI: `10.1137/090760805`), but this repository does **not** attempt to reproduce or directly follow that paper's framework.

Instead, this project takes a simpler simulation-plus-backtesting approach centered on short-horizon volatility estimation. That said, the paper's core message is very much relevant here: leveraged ETF returns are path-dependent, and the mismatch between "a fixed multiple of index return" intuition and realized performance becomes more problematic as variance and holding time increase. In that sense, the paper supports the practical design choice used throughout this repo: keep the holding horizon short. Most of the analysis here is deliberately concentrated on one to ten trading day holding horizons rather than long buy-and-hold windows.

## Repository layout

### Notebooks

- `00_preliminary_exploration.ipynb`: introduces leveraged ETF path dependence with toy examples and first comparisons across leverage factors and holding horizons.
- `01_frequency_impact.ipynb`: studies the role of rebalancing frequency in GBM simulation and connects the behavior to continuous-time formulas.
- `02_leverage_horizon_analysis.ipynb`: explores how leverage and holding horizon affect mean and standard deviation of the portfolio P&L.
- `03_drift_volatility_analysis.ipynb`: checks how drift and volatility change the dispersion of outcomes, with volatility emerging as the dominant factor.
- `04_leverage_volatility_analysis.ipynb`: varies leverage and volatility, visualizing the resulting surface of P&L standard deviation, and fits an initial polynomial approximation.
- `05_model_exploration.ipynb`: compares polynomial, transformed-target, and weighted approaches for modeling portfolio standard deviation.
- `06_data_creation.ipynb`: generates synthetic training and test datasets over leverage/volatility/horizon grids.
- `07_model_selection.ipynb`: evaluates candidate models and selects a simple theory-motivated specification.
- `08_historical_data_retrieval.ipynb`: downloads underlying ETF prices with `yfinance`, estimates realized volatility, and constructs historical short-horizon leveraged-vs-underlying P&L samples.
- `09_backtesting.ipynb`: fits the selected model on historical data and evaluates out-of-sample performance, including a set of extra tickers.

### Python modules

- `utils.py`: simulation utilities for GBM paths and synthetic dataset generation.
- `models.py`: scikit-learn model builders, including polynomial baselines and theory-driven models.

### Data and outputs

- `output/train/` and `output/test/`: synthetic datasets used in model selection.
- `output/historical_data/`: parquet datasets created from historical ETF prices for training and backtesting.
- `output/lev_sigma_std*.csv`: intermediate simulation outputs used in exploratory modeling.

### Notes

- `roadmap.md`: project goals, follow-up ideas, and future extensions.

## Methodology in brief

1. Simulate the underlying with GBM.
2. Construct leveraged returns by applying the leverage factor to the underlying return increments.
3. Study the P&L of a long leveraged position against a short underlying position.
4. Sweep leverage, volatility, rebalancing frequency, and holding horizon to understand the shape of the return distribution.
5. Generate synthetic datasets and fit simple predictive models for portfolio standard deviation.
6. Backtest the preferred specification on historical ETF data.

For the historical section, the repo does not download real leveraged ETF NAV series and fit directly on those. Instead, it starts from underlying ETF prices, computes realized volatility over a lookback window, and then constructs synthetic leveraged return series from the underlying returns. This keeps the historical exercise aligned with the same modeling assumptions used in the simulation notebooks.

## Setup

A typical local setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

If you want to run the parquet-based historical-data notebooks from scratch, you may also need `pyarrow`, since those notebooks read and write parquet files.

## Scope and limitations

- This is an exploratory research repo, not a production-ready trading system.
- The synthetic analysis relies heavily on GBM assumptions.
- The historical backtests use synthetic leveraged return construction from underlying ETF prices rather than observed LETF fund data.
- The focus is on short holding periods, not long-term portfolio construction.
- The current framing emphasizes volatility of P&L more than expected return forecasting.

## Next directions

Natural follow-ups include:

- testing the model on additional unseen tickers,
- building a simple calculator that maps target volatility to leverage factor and horizon,
- extending the framework to include expected return, Sharpe ratio, or drift assumptions,
- exploring pricing and hedging
