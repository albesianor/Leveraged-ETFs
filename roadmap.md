# Roadmap
- simulate GBM and compute comparisons between stock and lETF (mean, median, std, VaR)
- plot mean, median, std, VaR as functions of factor and window size (can control), and infer some relationship
- plot mean, median, std, VaR as functions of drift and volatility (cannot control in real-world), and infer some relationship
- knowing drift and vol, can we adjust window size and factor to get similar distributions
- try to figure out a theoretical dependence
- price lETF
- hedge lETF

## March 12 2026
- test model on unseen tickers
- make a calculator that outputs leverage factor (and possibly window span) to achieve desired portfolio volatility
- add documentation to make it readable and followable
- rearrange notebooks 08 and 09 to adapt it to eventual portfolio lifespan variation
- go back and run modelling for different portfolio lifespans (instead of just 5 days) - keep it in the range 1-10 days

## Future
- taking into consideration market drift, we can model expected returns as well as portfolio volatility