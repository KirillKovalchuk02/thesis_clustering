
import yfinance as yf
from yfinance import EquityQuery

from datetime import datetime
from dateutil.relativedelta import relativedelta
import random

import pandas as pd
import numpy as np

from typing import List, Dict







#Here we first create a filter with EquityQuery and then use it in the yf.screen() function
def get_tickers_stocks(min_dayvolume, exchanges, n):

    #markets = ['region'] + markets
    exchanges = ['exchange'] + exchanges
    
    q = EquityQuery('and', [            #EquityQuery('is-in', markets), #remove the notion of a region
                  EquityQuery('is-in', exchanges),
                  EquityQuery('gt', ['dayvolume', min_dayvolume])
    ])


    response = yf.screen(q, sortField = 'lastclosemarketcap.lasttwelvemonths', sortAsc = False, size=n) #select top 100 companies by market cap


    selected_stocks = {}
    ticker_list = []
    for stock in response['quotes']:
        ticker = stock['symbol']
        ticker_list.append(stock['symbol'])
        selected_stocks[ticker] = {} #initialize the new sub dictionary
        try:
            selected_stocks[ticker]['name'] = stock['shortName']
            selected_stocks[ticker]['type'] = stock['quoteType']
            selected_stocks[ticker]['exchange'] = stock['fullExchangeName']
        except:
            continue

    return selected_stocks, ticker_list





def get_close_prices(ticker_list, period = 2, start = '2022-01-01'):
    
    date_obj = datetime.strptime(start, '%Y-%m-%d')
    end = date_obj + relativedelta(years=period)
    enddate = end.strftime('%Y-%m-%d')

    df = yf.download(ticker_list, start=start, end=enddate)
    df_close = df['Close']

    df_return = df_close.dropna(how='all').copy()

    #Drop tickers that have more that 10% missing data
    df_return = df_return.drop(columns=df_return.columns[df_return.isna().mean() >= 0.1])
        
    #Impute missing data with the previous price (for eu and asia mostly)
    df_cleaned = df_return.ffill().copy()

    
    return df_cleaned





def double_listed_stocks(full_stocks_dict):
    
    company_names = []
    duplicated_tickers = []
    for ticker, sub_dict in full_stocks_dict.items():

        try:
            name = sub_dict['name']
        except:
            print(f'There is no name found in dict for {ticker}')

        if name not in company_names:
            company_names.append(name)
        else:
            duplicated_tickers.append(ticker)
    
    return duplicated_tickers




#Sharpe Ratio calculation

def sharpe_ratio_calculation(df, rf_rate_annual = 0.02, ):
    df_pct_change = df.pct_change()

    avg_return = df_pct_change.mean()
    sigma = df_pct_change.std()

    return_annual = avg_return * 252
    sigma_annual = sigma * np.sqrt(252)

    sharpe_ratio = (return_annual - rf_rate_annual) / sigma_annual

    return sharpe_ratio




def generate_rand_portfolios(n_reps:int, n_stocks:int, tickers:list):
    random_portfolios = {}
    for i in range(0, n_reps):
        stocks_indices = list()
        stocks_indices = random.sample(tickers, n_stocks)
        random_portfolios[f'portfolio_{i}'] = stocks_indices
        
    return random_portfolios 




def select_top_five(portfolios: List[Dict], metric: pd.Series) -> List[Dict]: #sharpe_ratio = sharpe_ratio_calculation(df_all_stocks, rf_rate_annual = 0.02)
    top_five_dict = {}
    for name, port in portfolios.items():
        portfolio = port
        
        metric = metric.apply(lambda x: float(x))

        dict_portfolio = {k: float(v) for k, v in dict(metric[portfolio]).items()} 

        sorted_dict = dict(sorted(dict_portfolio.items(), key=lambda x:x[1], reverse=True))
        top_five = dict(list(sorted_dict.items())[:5])
        top_five_dict[name] = top_five
    return top_five_dict