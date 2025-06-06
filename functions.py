
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
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from scipy.stats import entropy

import subprocess
import warnings





#Here we first create a filter with EquityQuery and then use it in the yf.screen() function
def get_tickers_stocks(min_dayvolume, exchanges, n):

    #markets = ['region'] + markets
    exchanges = ['exchange'] + exchanges
    
    q = EquityQuery('and', [         
                  EquityQuery('is-in', exchanges),
                  EquityQuery('gt', ['avgdailyvol3m', min_dayvolume]),

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





def get_close_prices(ticker_list:list, period = 2, start = '2022-01-01'):
    
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



def sharpe_ratio_calculation(df, rf_rate_annual = 0.02):
    df_pct_change = df.pct_change()

    avg_return = df_pct_change.mean()
    sigma = df_pct_change.std()

    return_annual = avg_return * 252
    sigma_annual = sigma * np.sqrt(252)

    sharpe_ratio = (return_annual - rf_rate_annual) / sigma_annual

    return sharpe_ratio


def generate_rand_portfolios(n_reps:int, n_stocks:int, tickers:list, weight_calc='random'):
    random_portfolios = {}
    for i in range(0, n_reps):
        stocks_indices = dict()
        stocks_indices = random.sample(tickers, n_stocks)
        if weight_calc == 'random':
            weights = np.random.random(n_stocks)
            weights /= np.sum(weights)
        elif weight_calc == 'equal':
            weights = np.ones(n_stocks) / n_stocks

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

def optimize_portfolio(mu, S, top_five:dict, clusters:dict=None, min_weight_for_top_five=0.01, verbose=False, min_stocks_per_cluster=2):
    """
    Optimizes a portfolio with the following constraints:
    - Minimum weights for top five stocks
    - Exactly 15 stocks in total
    - At least 2 stocks from each cluster
    
    Parameters:
    -----------
    mu : pd.Series
        Expected returns for each ticker
    S : pd.DataFrame
        Covariance matrix of returns
    top_five : dict
        Dictionary of top five tickers and their scores
    clusters : dict
        Dictionary mapping tickers to their cluster labels
    min_weight_for_top_five : float, optional
        Minimum weight for each of the top five stocks
        
    Returns:
    --------
    dict
        Selected tickers and their optimized weights
    """
    ef = EfficientFrontier(mu, S, solver=cp.CPLEX, weight_bounds=(0,1))
    
    for ticker in top_five.keys():
        ef.add_constraint(lambda w, t=ticker: w[ef.tickers.index(t)] >= min_weight_for_top_five)
    
    booleans = cp.Variable(len(ef.tickers), boolean=True)

    ef.add_constraint(lambda x: x <= booleans)
    ef.add_constraint(lambda x: cp.sum(booleans) == 15)
    



    if clusters:
        unique_clusters = set(clusters.values())
    
        # Add cluster constraints - at least 2 stocks from each cluster
        for cluster_label in unique_clusters:
            # Get indices of tickers in this cluster that are in ef.tickers
            cluster_tickers = [ticker for ticker in ef.tickers if ticker in clusters and clusters[ticker] == cluster_label]
            cluster_indices = [ef.tickers.index(ticker) for ticker in cluster_tickers]

            if verbose:
                print(f"Cluster {cluster_label}: found {len(cluster_indices)} stocks")

            # Add constraint to select at least 2 stocks from this cluster
            if cluster_indices:  # Only add constraint if there are stocks in this cluster
                cluster_sum = cp.sum([booleans[i] for i in cluster_indices])
                ef.add_constraint(lambda w, cs=cluster_sum: cs >= min_stocks_per_cluster)
    
    # Find the minimum volatility portfolio
    weights = ef.min_volatility()
    
    if verbose and clusters:
        print(f"Found {len(unique_clusters)} unique clusters: {unique_clusters}")
        # Check which stocks were selected and their weights
        selected_stocks = [ticker for ticker in ef.tickers if weights[ticker] > 1e-5]
        print(f"Selected {len(selected_stocks)} stocks in total")

        # Print cluster representation
        for cluster_label in unique_clusters:
            cluster_stocks = [t for t in selected_stocks if t in clusters and clusters[t] == cluster_label]
            print(f"Cluster {cluster_label}: {len(cluster_stocks)} stocks selected - {cluster_stocks}")
    
    # Don't filter by minimum weight here to ensure we get all 15 stocks
    selected = {ticker: weights[ticker] for ticker in ef.tickers if weights[ticker] > 1e-5}
    
    return selected



def run_min_variance(df_price, top_five:dict, risk_model='sample_cov', min_weight_for_top_five=0.01, clusters=None, min_stocks_per_cluster=2):
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

        result = optimize_portfolio(mu, S, port, min_weight_for_top_five=min_weight_for_top_five, clusters=None, min_stocks_per_cluster=min_stocks_per_cluster)
        results[index] = result
    
    return results




#####################################################################
################################CLUSTERING###########################
#####################################################################


def distance_matrix_calc(df, return_mode='arithmetic', method='kmeans'):
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

    
    return distance_matrix_df



def run_clustering_model(df, n_clus=3, model_name='kmeans', linkage='single', return_mode='geometric', n_init=3): 
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
        model = TimeSeriesKMeans(n_clusters=n_clus, metric="dtw", n_init=n_init, init='random', random_state=42)
        labels = model.fit_predict(data_scaled)
        inertia = model.inertia_
    elif model_name == 'kshape':
        model = KShape(n_clusters=n_clus, n_init=n_init, random_state=42)
        labels = model.fit_predict(data_scaled)
        inertia = model.inertia_

    tickers_with_labels = {k: int(v) for k, v in zip(tickers, labels)}
    try:
        cluster_centers = model.cluster_centers_
    except:
        cluster_centers = None

    return labels, tickers_with_labels, inertia, cluster_centers




#TEST THE COMBINED FUNCTION -  delete the separate ones

def test_clustering_metrics(df_dict, n_clusters_list, method='kmeans', linkage_list=None, 
                           return_mode='arithmetic', window=1, n_init=1):
    """
    Calculates both silhouette scores and label balance metrics for clustering models.
    
    Parameters:
    -----------
        df_dict: dict with dataframe name as key and stock prices dataframe as value
        n_clusters_list: list of number of clusters like [3,5,7] etc. that we want to find out the scores for
        method: 'kmeans', 'kshape' or 'ahc' (aggregated hierarchical clustering)
        linkage_list: only for ahc, what kind of linkages to use
        return_mode: usually 'arithmetic' 
        window: rolling window size for smoothing the data
        n_init: number of k-means initializations
        
    Returns:
    --------
        Combined DataFrame with silhouette scores and balance metrics
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    # Get the dataframe from dict
    df_name = list(df_dict.keys())[0]
    df = df_dict[df_name]
    
    # Apply rolling window smoothing
    df_smooth = df.rolling(window=window, center=False).mean().dropna()
    
    # Calculate distance matrix
    distance_matrix_df = distance_matrix_calc(df_smooth, return_mode=return_mode, method=method)
    
    # Store results
    results = []
    

    if linkage_list is None:
        linkage_list = ['not_applicable']
    
    for linkage in linkage_list:
        for n in n_clusters_list:
            # Get clustering results
            labels, tickers_with_labels, inertia, _ = run_clustering_model(
                df_smooth, n_clus=n, model_name=method, linkage=linkage, 
                return_mode=return_mode, n_init=n_init
            )
            
            # Calculate metrics
            sil_score, balance_metrics = _calculate_metrics_new(df_smooth, labels, tickers_with_labels, distance_matrix_df) #CAREFUL WITH THE NEW FUNCTION
            
            results.append({
                'clusters': n,
                'silhouette_score': float(sil_score),
                'method': method,
                'linkage': linkage,
                'return_mode': return_mode,
                'window_size': window,
                'df_mode': df_name,
                'inertia': inertia,
                **balance_metrics
            })
    
    return pd.DataFrame(results)


def _calculate_metrics_new(df_smooth, labels, tickers_with_labels, distance_matrix_df):
    """Helper function to calculate both metrics for a given clustering result"""

    # Silhouette score
    sil_score = silhouette_score(distance_matrix_df, labels, metric='precomputed')

    # Cluster label counts
    balance_df = pd.DataFrame(list(tickers_with_labels.items()), columns=['ticker', 'label'])
    cluster_counts = balance_df['label'].value_counts().sort_index()
    
    #Convert to proportions
    proportions = cluster_counts / cluster_counts.sum()

    #Entropy
    cluster_entropy = entropy(proportions, base=2)

    balance_metrics = {
        'entropy': round(cluster_entropy, 4),
    }

    return sil_score, balance_metrics




def evaluate_clustering_stability(df, 
                                  n_clusters,
                                   method='kmeans', 
                                   return_mode='geometric', 
                                   window_size=252,     # e.g. 1 year of daily data
                                   step_size=63,        # e.g. ~1 quarter
                                   linkage='average',
                                   n_init=3,
                                   agg_level=1,          # 1 = daily, 3 = every 3 days, 5 = weekly
                                   smoothing_window=None):  # e.g. 3-day moving average
    """
    Evaluates clustering stability over time using rolling windows,
    applying aggregation and optional smoothing within each window.

    Parameters:
    -----------
    df : DataFrame
        Time series data (rows = time, columns = tickers)
    agg_level : int
        Aggregation step size (1 = daily, 3 = every 3rd day, 5 = weekly, etc.)
    smoothing_window : int or None
        If set, applies a rolling mean of this window size (in aggregated points)
    """
    
    label_dicts = []
    time_indices = []

    # Adjust effective window and step sizes after aggregation
    effective_window = window_size
    effective_step = step_size

    for start in range(0, len(df) - effective_window + 1, effective_step):
        df_window = df.iloc[start:start + effective_window]
        time_indices.append(df.index[start])
        
        # Apply aggregation
        # df_agg = df_window.iloc[::agg_level].copy()
        if agg_level == 3:
            df_agg = df_window.resample('3D').last() #try aggregating on a weekly level
        elif agg_level == 5:
            df_agg = df_window.resample('W').last()
        elif agg_level == 1:
            df_agg = df_window.copy()
        else:
            raise ValueError("This aggregation level is not available")

        # Apply smoothing if specified
        if smoothing_window is not None:
            df_agg = df_agg.rolling(window=smoothing_window, min_periods=1).mean().dropna()

        if df_agg.shape[0] < 40: 
            print('The data is way too aggregated to make sense')
            return None
        try:
            # print((df <= 0).sum().sum())
            # if (df <= 0).sum().sum() > 0:
            #     return df_agg
            _, ticker_label_map, _, _ = run_clustering_model(
                df_agg,
                n_clus=n_clusters,
                model_name=method,
                linkage=linkage,
                return_mode=return_mode,
                n_init=n_init
            )
            label_dicts.append(ticker_label_map)
        except Exception as e:
            print(f"Clustering failed at window starting {df.index[start]}: {e}")
            label_dicts.append(None)
    
    # Compute stability scores between consecutive windows
    stability_scores = []
    
    for i in range(len(label_dicts) - 1):
        d1, d2 = label_dicts[i], label_dicts[i + 1]
        if d1 is None or d2 is None:
            continue
        
        common_tickers = list(set(d1) & set(d2))
        if len(common_tickers) < 5:
            continue
        
        labels1 = [d1[t] for t in common_tickers]
        labels2 = [d2[t] for t in common_tickers]
        
        ari = adjusted_rand_score(labels1, labels2)
        nmi = normalized_mutual_info_score(labels1, labels2)
        
        stability_scores.append({
            'window_pair': f"{time_indices[i].date()} → {time_indices[i+1].date()}",
            'ari': ari,
            'nmi': nmi,
            'common_tickers': len(common_tickers)
        })

    return pd.DataFrame(stability_scores)