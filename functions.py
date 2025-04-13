
import yfinance as yf
from yfinance import EquityQuery

from datetime import datetime
from dateutil.relativedelta import relativedelta
import random

import pandas as pd
import numpy as np

from typing import List, Dict

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

import cvxpy as cp

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans


import warnings




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






def join_stocks_crypto(crypto_df, stocks_df, mode = 'crypto_left'):
    if mode == 'crypto_left':
        joined_df = pd.merge(crypto_df, stocks_df, how='left', left_index=True, right_index=True)
        joined_df = joined_df.ffill()
        joined_df = joined_df.bfill()
    elif mode == 'stocks_left':
        joined_df = pd.merge(stocks_df, crypto_df, how='left', left_index=True, right_index=True)
    else:
        print('Unknown mode')
        
    return joined_df




#####################################################################
############DATA PROCESSING AND PORTFOLIO GENERATION#################
#####################################################################
def optimize_portfolio(mu, S, top_five:dict):

    ef = EfficientFrontier(mu, S, solver=cp.CPLEX, weight_bounds=(0,1))

    for ticker in top_five.keys():
        ef.add_constraint(lambda w: w[ef.tickers.index(ticker)] >= 0.01)

    booleans = cp.Variable(len(ef.tickers), boolean=True)
    ef.add_constraint(lambda x: x <= booleans)
    ef.add_constraint(lambda x: cp.sum(booleans) == 15)

    weights = ef.min_volatility()
    
    selected = {ticker: weights[ticker] for ticker in ef.tickers if weights[ticker] >= 0.01}

    return selected



def run_min_variance(df_price, top_five, risk_model='sample_cov'):
    mu = expected_returns.mean_historical_return(df_price)  # Expected returns
    
    if risk_model == 'sample_cov':
        S = risk_models.sample_cov(df_price)  # Covariance matrix
    elif risk_model == 'ledoit_wolf':
        S = risk_models.CovarianceShrinkage(df_price).ledoit_wolf()
    else:
        print('Model not recognised')



    results = dict()
    for index, port in top_five.items():

        result = optimize_portfolio(mu, S, port)
        results[index] = result
    
    return results




#####################################################################
################################CLUSTERING###########################
#####################################################################

def run_kmeans_dtw(df_all_stocks, n_clus=3):
    df_returns = df_all_stocks.pct_change().dropna()
    data_kmeans = df_returns.T.values

    tickers = list(df_all_stocks.columns)

    data_scaled = TimeSeriesScalerMeanVariance().fit_transform(data_kmeans)

    model = TimeSeriesKMeans(n_clusters=n_clus, metric="dtw", random_state=0)
    labels = model.fit_predict(data_scaled)

    tickers_with_lables = {k: int(v) for k, v in zip(tickers, labels)}

    warnings.simplefilter(action='ignore', category=FutureWarning) #supress warnings for cleanliness

    return tickers_with_lables






from tslearn.metrics import cdist_dtw

def dtw_matrix_calc(df):
    df_returns = df.pct_change().dropna()
    X = df_returns.T.values

    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)
    
    distance_matrix = cdist_dtw(X_scaled)

    tickers = df_returns.columns  # or wherever your tickers are stored
    dtw_matrix_df = pd.DataFrame(distance_matrix, index=tickers, columns=tickers)

    return dtw_matrix_df