from __future__ import annotations
from typing import Dict, List, Union, Generic, Optional
import requests
import json
from pycoingecko import CoinGeckoAPI
import numpy as np
import matplotlib.pyplot as plt
import datetime as d
import statsmodels.api as sm
import pandas as pd
from option import Option
import yaml

AssetDict = Dict[str, Union[str, float, List[str]]]
NetworkDict = Dict[str, Union[str, List[AssetDict]]]
AccountDict = Dict[str,Union[str,List[NetworkDict]]]
PortfolioDict = Dict[str, List[AccountDict]]

class Asset:
    def __init__(self, asset: AssetDict) -> None:
        self.amount: float = asset["amount"] #type: ignore
        self.value: float = asset["value"] if "value" in asset else coingecko_coin_price(asset["name"]) * self.amount
        self.name: str = asset["name"] #type: ignore
        self.lp: Option[List[str]] = Option.maybe(asset["lp"] if "lp" in asset else None)
    def name(self) -> List[str]:
        return self.lp.unwrap() if self.lp.is_some else [self.name]
        
class Network:
    def __init__(self, network: NetworkDict) -> None:
        self.name: str = network["name"]
        self.items: List[Asset] = [Asset(asset) for asset in network["items"]]

class Account:
    def __init__(self, account: AccountDict) -> None:
        self.name = account["name"]
        self.networks = [Network(network) for network in account["network"]]

def etherscan():
    global apikeys 
    apikeys = "8TVTQ2IVSFWM99VZV2JZC7CDB82BCWXQYV"
    
    with open("ether_contracts.json", 'r') as ether_contracts_file:
        global ERC20_contracts
        ERC20_contracts = json.load(ether_contracts_file)
    with open("user_address.txt", 'r') as user_address_file:
        global user_address
        user_address = user_address_file.read()

    def TokenTotalSupply(token: str):
        link: str = "https://api.etherscan.io/api?module=stats&action=tokensupply&contractaddress=" + ERC20_contracts[token] + "&apikey=" + apikeys
        response: Dict = requests.get(link).json()

        if response["status"] == "0":
            raise KeyError(response["result"])
        return response["result"]

    # Assuming 18 decimals
    def TokenAccountBalance(token: str, address: str):
        link: str = "https://api.etherscan.io/api?module=account&action=tokenbalance&contractaddress=" + ERC20_contracts[token] + "&address=" + address + "&tag=latest&apikey=" + apikeys
        response: Dict = requests.get(link).json()

        # print(response)
        if response["status"] == "0":
            raise KeyError(response["result"])
        elif response["result"] == '0':
            return 0
        return int(response["result"]) / 10**18

    def AccountBalance(address, amountOver):
        portfolio = {}
        for token in ERC20_contracts.keys():
            balance = TokenAccountBalance(token, address)
            if balance > amountOver:
                portfolio[token] = balance
        return portfolio

# print(ETH_AccountBalance(user_address, 0.0001))

class CoinGeckoCoinTrend:
    # Shared between all classes
    api_provider = 'CoinGecko'
    api = CoinGeckoAPI()

    def __init__(self, coin, days) -> None:
        self.coin = coin
        self.days = days
        self.coin_data = self.__get_coin_data(self.coin, self.days)

    def __get_coin_data(self, coin, days) -> Dict:
        data = self.api.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=days)
        unixtimestamp_list, price_list = list(zip(*data["prices"]))
        datetime_list = [d.datetime.fromtimestamp(unixtimestamp / 1000) for unixtimestamp in unixtimestamp_list]

        return {"datetime": datetime_list, "price": price_list}

class CoinGeckoCoins(CoinGeckoCoinTrend):
    def __init__(self, coin1, coin2, days) -> None:
        self.coin1 = coin1
        self.coin2 = coin2
        self.days = days
        self.coin1_data = self._CoinGeckoCoinTrend__get_coin_data(self.coin1, self.days)
        self.coin2_data = self._CoinGeckoCoinTrend__get_coin_data(self.coin2, self.days)
        self.coin_comparison = self.__converge_data()
  
    # nearest date is chosen by a difference of less than 60 seconds and whichever is first in the list.
    # For whatever reason the actual difference is greater than 60 seconds but it is gets the nearest datetime.
    def __converge_data(self) -> pd.DataFrame:
        result = {"date1": self.coin1_data["datetime"], "price1": self.coin1_data["price"], "date2": [], "price2": [], "price compahttps://github.com/EdvinAlvarado/python-option.gitrison": []}
        for i in range(len(self.coin1_data["datetime"])):
            nearest_match_coin2_date = next(d for d in self.coin2_data["datetime"] if (self.coin1_data["datetime"][i] - d).total_seconds() < 3600 * 24)
            result["date2"].append(nearest_match_coin2_date)
            result["price2"].append(self.coin2_data["price"][self.coin2_data["datetime"].index(nearest_match_coin2_date)])
            # print("{}\t{}".format(self.coin1_data[0][i], nearest_match_self.coin2_data_date))
        result["price comparison"] = [price1 / price2 for price1, price2 in zip(result["price1"], result["price2"])]
        return pd.DataFrame(result)

    def __check_coin(self, coin) -> None:
        if coin not in (self.coin1, self.coin2):
            raise KeyError("Coin not included in object")

    def plot(self, coin) -> None:
        self.__check_coin(coin)
        if coin == self.coin1:
            plt.plot(self.coin1_data["datetime"], self.coin1_data["price"])
        elif coin == self.coin2:
            plt.plot(self.coin2_data["datetime"], self.coin2_data["price"])

    def compare_plot(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.coin_comparison["date1"], self.coin_comparison["price1"], color='blue')
        ax.set_ylabel("USD", color='blue')
        ax2 = ax.twinx()
        ax2.plot(self.coin_comparison["date1"], self.coin_comparison["price comparison"], color='red')
        ax2.set_ylabel(self.coin2, color='red')
        plt.title(self.coin1)
        plt.show()


class CoinGeckoCoinsStats(CoinGeckoCoins):
    
    def __init__(self, coin1, coin2, days) -> None:
        super().__init__(coin1, coin2, days)
        self.model = self.__coin_correlation()

    def __coin_correlation(self):
        x = self.coin_comparison["price2"]
        y = self.coin_comparison["price1"]
        model = sm.OLS(y, x).fit()
        predictions = model.predict(x)
        # print(model.summary())
        return model

    def corrcoef(self) -> float:
        return np.corrcoef([self.coin_comparison["price1"], self.coin_comparison["price2"]])[0][1]

    def cumret(self, coin) -> float:
        self.__check_coin(coin)
        c = "price1" if coin == self.coin1 else "price2"
        return (self.coin_comparison[c].tail(1).item() - self.coin_comparison[c][0]) / self.coin_comparison[c][0] - 1

    def days_of_data(self):
        return (self.coin_comparison["date1"].tail(1).item() - self.coin_comparison["date1"][0]).days

    def annhmlret(self, coin) -> float:
        self.__check_coin(coin)
        return (1 + self.cumret(coin))**(365 / self.days_of_data()) - 1

    # FIXME
    def sharpe_ratio(self, coin, rf) -> float:
        self.__check_coin(coin)
        return (self.annret(coin) - rf) / np.std(self.coin_comparison["price1" if coin == self.coin1 else "price2"])


def coingecko_coin_price(coin):
    api = CoinGeckoAPI()
    
    with open('coin-list.json', 'r') as cl:
        coin_list = json.loads(cl.read())   
    
    def find_coin_id(coin_name) -> str:
        for coin in coin_list:
            if coin_name.lower() in (coin["symbol"].lower(), coin["name"].lower()):
                return coin["id"]
        print("coin not found")
        return ""

    def coin_price(coin):
        id = find_coin_id(coin)
        return api.get_price(ids=id, vs_currencies='usd')[id]['usd']

    return coin_price(coin)


class Portfolio:
    def __init__(self, yaml_file: str):
        with open(yaml_file, "r") as file:
            n = yaml.safe_load(file)
        self.accounts = [Account(account) for account in n["accounts"]]

    def assets(self, sel_acc="all", sel_net="all") -> List[Asset]:
        l = []
        for account in filter(lambda a: a.name == sel_acc or sel_net == "all", self.accounts):
            for network in filter(lambda n: n.name == sel_net or sel_net == "all", account.networks):
                l.extend(network.items)
        return l
    
    def flatten_assets(self, sel_acc="all", sel_net="all", simplify=True) -> Dict[str, float]:
        cash = ["USDC", "DAI", "USDT", "BUSD", "EUR", "GUSD", "TUSD"]
        d = {"cash": 0}
        l = self.assets(sel_acc=sel_acc, sel_net=sel_net)
        for asset in l:
            if asset.lp.is_some:
                val = asset.value / len(asset.lp.unwrap())
                for ass in asset.lp.unwrap():
                    if ass in cash:
                        d["cash"] += val
                    elif ass not in d:
                        d[ass] = val
                    else:
                        d[ass] += val
            elif asset.name in cash:
                d["cash"] += asset.value
            elif asset.name not in d:
                d[asset.name] = asset.value
            else:
                d[asset.name] += asset.value
        return d

    def filtered_assets(self, sel_acc="all", sel_net="all", simplify=True, filter=0.01) -> Dict[str, float]:
        l = self.flatten_assets(sel_acc=sel_acc, sel_net=sel_net, simplify=simplify)
        total_value = sum(l.values())
        d = {"others": 0}
        for coin,val in l.items():
            if val <= filter * total_value:
                d["others"] += val
            else:
                d[coin] = val
        return d

    def pie(self, sel_acc="all", sel_net="all", simplify=True, filter=0.01):
        d = self.flatten_assets(sel_acc=sel_acc, sel_net=sel_net, simplify=simplify) if filter == 0 else self.filtered_assets(sel_acc=sel_acc, sel_net=sel_net, simplify=simplify, filter=filter)
        print(d)
        coins = d.keys()
        values = d.values()

        fig1, ax1 = plt.subplots()
        ax1.pie(values, labels=coins, autopct='%1.2f%%')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f"{sel_acc} {sel_net}")
        plt.show()
