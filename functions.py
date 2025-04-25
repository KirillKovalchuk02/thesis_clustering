
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
from tslearn.metrics import cdist_dtw
from tslearn.clustering import KShape
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import subprocess
import warnings





#Here we first create a filter with EquityQuery and then use it in the yf.screen() function
def get_tickers_stocks(min_dayvolume, exchanges, n):

    #markets = ['region'] + markets
    exchanges = ['exchange'] + exchanges
    
    q = EquityQuery('and', [         
                  EquityQuery('is-in', exchanges),
                  EquityQuery('gt', ['dayvolume', min_dayvolume])
    ])


    response = yf.screen(q, sortField = 'lastclosemarketcap.lasttwelvemonths', sortAsc = False, size=n) #select top N companies by market cap


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





def double_listed_stocks(full_stocks_dict): #finds the doubly listed tickers in the dataframe
    
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



#OLD:
# def generate_rand_portfolios(n_reps:int, n_stocks:int, tickers:list):
#     random_portfolios = {}
#     for i in range(0, n_reps):
#         stocks_indices = dict()
#         stocks_indices = random.sample(tickers, n_stocks)
#         random_portfolios[f'portfolio_{i}'] = stocks_indices
        
#     return random_portfolios 

#NEW:
def generate_rand_portfolios(n_reps:int, n_stocks:int, tickers:list):
    random_portfolios = {}
    for i in range(0, n_reps):
        stocks_indices = dict()
        stocks_indices = random.sample(tickers, n_stocks)
        weights = np.random.random(n_stocks)
        weights /= np.sum(weights)

        tickers_w_weights_dict = dict()
        for k, v in zip(stocks_indices, weights):
            tickers_w_weights_dict[k] = float(v)

        random_portfolios[f'portfolio_{i}'] = tickers_w_weights_dict
        
    return random_portfolios




def select_top_five(portfolios, metric: pd.Series) -> List[Dict]: #gets the top 5 stocks with the highest sharpe ratio
    top_five_dict = {}
    for name, port in portfolios.items():
        portfolio = list(port.keys())
        
        metric = metric.apply(lambda x: float(x))

        dict_portfolio = {k: float(v) for k, v in dict(metric[portfolio]).items()} 

        sorted_dict = dict(sorted(dict_portfolio.items(), key=lambda x:x[1], reverse=True))
        top_five = dict(list(sorted_dict.items())[:5])
        top_five_dict[name] = top_five
    return top_five_dict






def join_stocks_crypto(crypto_df, stocks_df, mode = 'crypto_left'): #joins the full stock dataset with full cryptos dataset joining on the date
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

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def optimize_portfolio(mu, S, top_five:dict, min_weight_for_top_five=0.01):
    """ runs the portfolio optimization with the min variance as an objective 
    
    parameters:
    -----------
        mu: Expected returns
        S: Covariance Matrix
        top_five: dict with the preselected stocks that must be included in the optimizaed portfolio 
                    (their weights must be at least X% of the portfolio)
        min_weight_for_top_five: minimum weight that top five preselected stocks must have
    returns:
    --------
        DTW matrix in a dataframe format
    """

    log('calculating frontier')
    ef = EfficientFrontier(mu, S, solver=cp.CPLEX, weight_bounds=(0,1))

    for ticker in top_five.keys():
        ef.add_constraint(lambda w: w[ef.tickers.index(ticker)] >= min_weight_for_top_five)

    booleans = cp.Variable(len(ef.tickers), boolean=True)
    ef.add_constraint(lambda x: x <= booleans)
    ef.add_constraint(lambda x: cp.sum(booleans) == 15)

    log('finding min_volatility')
    weights = ef.min_volatility()
    
    selected = {ticker: weights[ticker] for ticker in ef.tickers if weights[ticker] >= 0.01}

    return selected



def run_min_variance(df_price, top_five, risk_model='sample_cov', min_weight_for_top_five=0.01):
    """ Actually runs the optimization
    
    parameters:
    -----------
        df_price: dataframe of asset prices
        top_five: dict of dict per portfolio with the preselected stocks that must be included in the optimizaed portfolio 
                    (their weights must be at least X% of the portfolio)
        risk_model: what kind of covariance matrix calculation to use - from pyportopt
        min_weight_for_top_five: minimum weight that top five preselected stocks must have
    returns:
    --------
        DTW matrix in a dataframe format
    """
    mu = expected_returns.mean_historical_return(df_price)  # Expected returns
    print('calculating the covariance matrix')
    if risk_model == 'sample_cov':
        S = risk_models.sample_cov(df_price)  # Covariance matrix
    elif risk_model == 'ledoit_wolf':
        S = risk_models.CovarianceShrinkage(df_price).ledoit_wolf()
    else:
        print('Model not recognised')



    results = dict()
    for index, port in top_five.items():

        result = optimize_portfolio(mu, S, port, min_weight_for_top_five)
        results[index] = result
    
    return results




#####################################################################
################################CLUSTERING###########################
#####################################################################


def distance_matrix_calc(df, return_mode='arithmetic', method='kmeans'):
    """ calculates the DTW matrix for the 
    
    parameters:
    -----------
        df: dataframe with the stock prices
    returns:
    --------
        DTW matrix in a dataframe format
    """
    tickers = df.columns


    if return_mode == 'arithmetic':
        df_returns = df.pct_change().dropna()
    elif return_mode == 'geometric':
        df_returns = np.log(df / df.shift(1)).dropna()
    
    if method == 'kmeans' or method == 'ahc':
        X = df_returns.T.values

        scaler = TimeSeriesScalerMeanVariance()
        X_scaled = scaler.fit_transform(X)
    
        distance_matrix = cdist_dtw(X_scaled)

        distance_matrix_df = pd.DataFrame(distance_matrix, index=tickers, columns=tickers)
        
    elif method == 'kshape':
        df_returns.to_csv('data_for_R/dataframe_returns.csv') #the first row with NAs is already dropped
        retcode = subprocess.call(['C:/Program Files/R/R-4.5.0/bin/Rscript', '--vanilla', 'sbd.r'], shell=True) #EXECUTING R script, which saves the SBD matrix to the data_for_R folder
        if retcode == 0:
            pass
        else:
            raise RuntimeError("Something went wrong with the R file")
        
        distance_matrix_df = pd.read_csv('data_for_R/sbd_matrix.csv')
        distance_matrix_df.columns = tickers
        distance_matrix_df.index = tickers

    
    return distance_matrix_df#, distance_matrix



def run_clustering_model(df, n_clus=3, model_name='kmeans', linkage='single', return_mode='arithmetic', n_init=1): 
    """ runs the clustering for one of the 3 possible models
    
    parameters:
    -----------
        df: dataframe with the stock prices 
        n_clus: how many clusters to generate
        model_name: kmeans, kshape or ahc (aggregated hierarchical clustering)
        linkage: only for ahc, what kind of linkage to use
        
    returns:
    --------
        array of labels and dictionary mapping the cluster label to the ticker
    """

    if return_mode == 'arithmetic':
        df_returns = df.pct_change().dropna()
    elif return_mode == 'geometric':
        df_returns = np.log(df / df.shift(1)).dropna()

    data_clustering = df_returns.T.values

    tickers = list(df.columns)
    warnings.simplefilter(action='ignore', category=FutureWarning) #supress warnings for cleanliness

    scaler = TimeSeriesScalerMeanVariance()
    data_scaled = scaler.fit_transform(data_clustering)


    if model_name == 'ahc':
        dtw_matrix = distance_matrix_calc(df, return_mode=return_mode, method=model_name)
        model = AgglomerativeClustering(n_clusters=n_clus, metric='precomputed', linkage=linkage)
        labels = model.fit_predict(dtw_matrix)
        inertia = None
    elif model_name == 'kmeans':
        model = TimeSeriesKMeans(n_clusters=n_clus, metric="dtw", n_init=n_init, init='random')
        labels = model.fit_predict(data_scaled)
        inertia = model.inertia_
    elif model_name == 'kshape':
        model = KShape(n_clusters=n_clus, n_init=n_init)
        labels = model.fit_predict(data_scaled)
        inertia = model.inertia_

    tickers_with_labels = {k: int(v) for k, v in zip(tickers, labels)}
    try:
        cluster_centers = model.cluster_centers_
    except:
        cluster_centers = None
    return labels, tickers_with_labels, inertia, cluster_centers









def test_for_silhouette_score(df, n_clusters_list, method='kmeans', linkage_list=None, return_mode='arithmetic', n_init=1):

    """ runs the clustering for one of the 3 possible models and the Silhouette score calculation
    parameters:
    -----------
        df: dataframe with the stock prices
        n_clusters_list: list of number of clusters like [3,5,7] etc. that we want to find out the silhouette score for
        model_name: kmeans, kshape or ahc (aggregated hierarchical clustering)
        linkage_list: only for ahc, what kind of linkages to use
    returns:
    --------
        DataFrame of the silhouette scores per N clusters and type of Linkage
    """

    distance_matrix_df = distance_matrix_calc(df, return_mode=return_mode, method=method)

    silhouettes = []

    if method == 'ahc':
        if linkage_list is None:
            raise ValueError("You must provide a list of linkages when using method='ahc'")
        
        for linkage in linkage_list:
            for n in n_clusters_list:
                labels, _, _, _ = run_clustering_model(df, n_clus=n, model_name=method, linkage=linkage, return_mode=return_mode)
                score = silhouette_score(distance_matrix_df, labels, metric='precomputed')
                silhouettes.append({
                    'clusters': n,
                    'silhouette_score': float(score),
                    'method': method,
                    'linkage': linkage
                })
    
    elif method == 'kmeans':
        for n in n_clusters_list:
            labels, _, inertia, _ = run_clustering_model(df, n_clus=n, model_name=method, return_mode=return_mode, n_init=n_init)
            score = silhouette_score(distance_matrix_df, labels, metric='precomputed')
            silhouettes.append({
                'clusters': n,
                'silhouette_score': float(score),
                'inertia': float(inertia),
                'method': method
            })
    
    else: 
        #print('For now, we use inertia for KShape, as calculating SBD matrix is not feasible (maybe use R for that?)')
        for n in n_clusters_list:
            labels, _, inertia, _ = run_clustering_model(df, n_clus=n, model_name=method, return_mode=return_mode)

            score = silhouette_score(distance_matrix_df, labels, metric='precomputed') #TESTTTTT

            silhouettes.append({
                'clusters': n,
                'silhouette_score': float(score),
                'inertia': float(inertia),
                'method': method
            })

    return pd.DataFrame(silhouettes)