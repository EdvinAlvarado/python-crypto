# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
from crypto import CoinGeckoCoins as cg
import crypto
import yaml
from pprint import pprint

# wrapped = cg("wrapped-nxm", "nxm", 300)
# wrapped = cg("convex-crv", "curve-dao-token", 300)
# wrapped.compare_plot()

with open("portfolio.yaml", "r") as file:
    n = yaml.safe_load(file)
pprint(n)


crypto_portfolio = crypto.Portfolio("portfolio.yaml")
# print(crypto_portfolio.portfolio)
crypto_portfolio.pie()
full_portfolio = crypto_portfolio.filtered_assets()

cash = full_portfolio["cash"]
notcash = sum(full_portfolio.values()) - cash
print(f'cash\t{cash}')
print(f'crypto\t{notcash}')

pprint(crypto_portfolio.accounts)
pprint(full_portfolio)
