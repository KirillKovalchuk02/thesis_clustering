import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import normaltest
from scipy.stats import mstats

import scikit_posthocs as sp

from functions import sharpe_ratio_calculation

from scipy import stats
from scipy import optimize



def select_cryptos_from_subclusters(existing_stocks, crypto_candidates, crypto_cluster_assignments, df_prices, 
                                   n_cryptos=3, selection_method='random', rf_rate=0.04):
    
    crypto_by_subcluster = {}
    for crypto in crypto_candidates:
        if crypto in crypto_cluster_assignments:
            cluster = crypto_cluster_assignments[crypto]
            if cluster not in crypto_by_subcluster:
                crypto_by_subcluster[cluster] = []
            crypto_by_subcluster[cluster].append(crypto)
    
    available_subclusters = list(crypto_by_subcluster.keys())
    
    n_cryptos = min(n_cryptos, len(available_subclusters))
    
    selected_cryptos = []
    
    if selection_method == 'random':
        for i, cluster in enumerate(available_subclusters[:n_cryptos]):
            selected_crypto = random.choice(crypto_by_subcluster[cluster])
            selected_cryptos.append(selected_crypto)
            
    elif selection_method == 'correlation':
        returns = df_prices.pct_change().dropna()
        available_stocks = [stock for stock in existing_stocks if stock in returns.columns]
        
        if not available_stocks:
            for cluster in available_subclusters[:n_cryptos]:
                selected_cryptos.append(random.choice(crypto_by_subcluster[cluster]))
        else:
            portfolio_returns = returns[available_stocks].mean(axis=1)

            for cluster in available_subclusters[:n_cryptos]:
                cluster_cryptos = crypto_by_subcluster[cluster]

                crypto_correlations = {}
                
                for crypto in cluster_cryptos:
                    if crypto in returns.columns:

                        aligned_portfolio, aligned_crypto = portfolio_returns.align(returns[crypto], join='inner')
                        
                        if len(aligned_portfolio) > 30:  # Minimum data points for reliable correlation
                            corr = aligned_portfolio.corr(aligned_crypto)
                            crypto_correlations[crypto] = abs(corr) if not pd.isna(corr) else 1.0
                        else:
                            crypto_correlations[crypto] = 1.0
                    else:
                        crypto_correlations[crypto] = 1.0  
             
                if crypto_correlations:
                    best_crypto = min(crypto_correlations.keys(), key=lambda crypto: crypto_correlations[crypto])
                    selected_cryptos.append(best_crypto)
                else:

                    selected_cryptos.append(random.choice(cluster_cryptos))
    

    full_portfolio = existing_stocks + selected_cryptos
    total_assets = len(full_portfolio)
    equal_weight = 1.0 / total_assets
    
    return {ticker: equal_weight for ticker in full_portfolio}



def select_cryptos_by_correlation(existing_stocks, crypto_candidates, df_prices, n_cryptos=3, selection_metric='sharpe', rf_rate=0.04):
    returns = df_prices.pct_change().dropna()
    

    available_stocks = [stock for stock in existing_stocks if stock in returns.columns]
    if not available_stocks:

        selected_cryptos = random.sample(crypto_candidates, n_cryptos)
        crypto_supplemented_port = existing_stocks + selected_cryptos
        total_assets = len(crypto_supplemented_port)
        equal_weight = 1.0 / total_assets
        return {ticker: equal_weight for ticker in crypto_supplemented_port}
    
    portfolio_returns = returns[available_stocks].mean(axis=1) 
    

    if selection_metric == 'sharpe':
        metric_data = dict(sharpe_ratio_calculation(df_prices, rf_rate_annual=rf_rate))
    else:
        metric_data = {asset: returns[asset].mean() * 252 for asset in crypto_candidates if asset in returns.columns}
    

    crypto_correlations = {}
    
    for crypto in crypto_candidates:
        if crypto in returns.columns:

            aligned_portfolio, aligned_crypto = portfolio_returns.align(returns[crypto], join='inner')
            
            if len(aligned_portfolio) > 30:  
                corr = aligned_portfolio.corr(aligned_crypto)
                crypto_correlations[crypto] = corr if not pd.isna(corr) else 1.0
            else:
                crypto_correlations[crypto] = 1.0  
        else:
            crypto_correlations[crypto] = 1.0
    
    valid_cryptos = [crypto for crypto in crypto_candidates if crypto in metric_data]

    sorted_cryptos = sorted(
        valid_cryptos,
        key=lambda crypto: (crypto_correlations[crypto], -metric_data.get(crypto, -999))
    )
    
    selected_cryptos = sorted_cryptos[:n_cryptos]
    crypto_supplemented_port = existing_stocks + selected_cryptos

    total_assets = len(crypto_supplemented_port)
    equal_weight = 1.0 / total_assets
    weighted_portfolio = {ticker: equal_weight for ticker in crypto_supplemented_port}
    
    return weighted_portfolio


def supplement_set_with_cryptos(portfolio_set: dict, cryptos_list, tickers_with_labels, df_prices,
                                selection_metric='sharpe', 
                               selection_method='clustering', selection_method_clusters='random', 
                               n_cryptos=3, seed=30, rf_rate=0.04):

    random.seed(seed)
    new_portfolios_w_cryptos = dict()
    
    for key, portfolio in portfolio_set.items():
        existing_stocks = list(portfolio.keys())
        
        if selection_method == 'random':
            # Random selection
            indices = random.sample(range(len(cryptos_list)), n_cryptos)
            cryptos = [cryptos_list[i] for i in indices]
            crypto_supplemented_port = existing_stocks + cryptos
            total_assets = len(crypto_supplemented_port)
            equal_weight = 1.0 / total_assets
            portfolio_to_return = {ticker: equal_weight for ticker in crypto_supplemented_port}

        elif selection_method == 'clustering':
            portfolio_to_return = select_cryptos_from_subclusters(
                existing_stocks=existing_stocks, 
                crypto_candidates=cryptos_list, 
                crypto_cluster_assignments=tickers_with_labels, 
                df_prices=df_prices, 
                n_cryptos=n_cryptos, 
                selection_method=selection_method_clusters,
                rf_rate=rf_rate
            )
            
        elif selection_method == 'correlation':
            # Correlation-based selection
            portfolio_to_return = select_cryptos_by_correlation(
                existing_stocks=existing_stocks,
                crypto_candidates=cryptos_list,
                df_prices=df_prices,
                n_cryptos=n_cryptos,
                selection_metric=selection_metric,
                rf_rate=rf_rate
            )
            
        
        new_portfolios_w_cryptos[key] = portfolio_to_return

    return new_portfolios_w_cryptos






def reoptimize_weights(df_prices, portfolio_set, how='max_sharpe', min_weight=0.01, rf_rate=0.02):
    new_set = dict()
    for key, portfolio in portfolio_set.items():
        if isinstance(portfolio, dict):
            tickers = list(portfolio.keys())
        elif isinstance(portfolio, list):
            tickers = portfolio

        if how == 'equal_weights':
            weight = 1/len(tickers)
            new_portfolio = {ticker: weight for ticker in tickers}
        
        elif how == 'max_sharpe':
            prices_df = df_prices[tickers]
        
            mu = expected_returns.mean_historical_return(prices_df)  
            S = risk_models.sample_cov(prices_df)  
            try:

                ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, 1))
                weights = ef.max_sharpe(risk_free_rate=rf_rate)
                new_portfolio = ef.clean_weights()
            except:
                ef = EfficientFrontier(mu, S, weight_bounds=(min_weight - 0.01, 1))
                weights = ef.max_sharpe(risk_free_rate=rf_rate)
                new_portfolio = ef.clean_weights()

        new_portfolio = dict(new_portfolio)
        new_set[key] = new_portfolio

    return new_set
    






def run_simulation(portfolio_dict:dict, returns_for_portfolio:pd.DataFrame, n_sims=100, t=100, distribution_model='multivar_norm', 
                   plot=False, initialPortfolio=100, winsorize=False, winsorize_limits=(0.01, 0.01)):
    returns_for_portfolio = returns_for_portfolio[list(portfolio_dict.keys())]
    
    if winsorize:

        winsorized_returns = returns_for_portfolio.copy()
        
        for col in winsorized_returns.columns:
            winsorized_returns[col] = mstats.winsorize(winsorized_returns[col], limits=winsorize_limits)
        

        returns_for_portfolio = winsorized_returns


    mean_returns = returns_for_portfolio.mean()
    cov_matrix = returns_for_portfolio.cov()

    weights = [v for _, v in portfolio_dict.items()]

    meanM = np.tile(mean_returns, (t, 1))  # Shape: (T, n_assets)

    portfolio_sims = np.zeros((t, n_sims))

    L = np.linalg.cholesky(cov_matrix)

    for sim in range(n_sims):

        if distribution_model == 'bootstrap':
            sampled_returns = returns_for_portfolio.bfill().sample(n=t, replace=True).values
            portfolio_returns = sampled_returns @ weights
        elif distribution_model in ['multivar_norm', 'multivar_t']:

            if distribution_model == 'multivar_norm':
                Z = np.random.normal(size=(t, len(portfolio_dict))) 
                daily_returns = meanM + Z @ L.T 
            elif distribution_model == 'multivar_t':
                df = 25  # degrees of freedom
                Z = np.random.normal(size=(t, len(portfolio_dict)))
                chi2 = np.random.chisquare(df, size=(t, 1))
                Z_t = Z / np.sqrt(chi2 / df) 

                daily_returns = meanM + Z_t @ L.T

            portfolio_returns = daily_returns @ weights 
        
        else:
            break
        
        portfolio_sims[:, sim] = np.cumprod(1 + portfolio_returns) * initialPortfolio

    return portfolio_sims


def calculate_cumulative_returns(daily_returns):
    growth_factors = daily_returns + 1
    cumulative_growth = np.cumprod(growth_factors, axis=0)
    cumulative_returns = cumulative_growth[-1, :] - 1
    
    return cumulative_returns




def calculate_var_cvar(daily_returns, confidence_level=0.05, initial_value=100):

        sorted_returns = np.sort(daily_returns.flatten())
        
        var_index = int(confidence_level * len(sorted_returns))
        var = abs(sorted_returns[var_index])
        
        cvar = abs(np.mean(sorted_returns[:var_index]))
        
        var_value = var 
        cvar_value = cvar
        
        return var_value, cvar_value




#Simulations for the whole subset

def simulate_evaluate_portfolio_subset(portfolios_subset:dict, return_df, n_sims=100, t=100, distribution_model='multivar_norm', rf_annual=0.04, seed=30,
                                       winsorize=False, winsorize_limits=(0.01, 0.01)):
    simulations_results_dict = dict()
    subset_statistics_df = pd.DataFrame()

    random.seed(seed)

    for i, portfolio_dict in portfolios_subset.items():
        returns_portfolio = return_df[list(portfolio_dict.keys())]

        total = sum(portfolio_dict.values())
        portfolio_dict = {k: v / total for k, v in portfolio_dict.items()}

        portfolio_sims = run_simulation(portfolio_dict, returns_portfolio, n_sims=n_sims, t=t, distribution_model=distribution_model, plot=False, winsorize=winsorize, winsorize_limits=winsorize_limits)

        simulations_results_dict[i] = portfolio_sims


        #CALCULATE STATISTICS PER PORTFOLIO:
        daily_returns = (portfolio_sims[1:, :] - portfolio_sims[:-1, :]) / portfolio_sims[:-1, :]
        cumulative_returns_per_simulation = calculate_cumulative_returns(daily_returns)
        mean_daily_return_for_portfolio = np.mean(daily_returns)
        #STD
        std_daily_return = np.std(daily_returns) 


        final_portfolio_values = portfolio_sims[-1, :]  
        initial_value = 100  
        holding_period_years = t / 252

        # Calculate annualized returns 
        return_for_period = (final_portfolio_values - initial_value) / initial_value
        annualised_return = (final_portfolio_values / initial_value) ** (1 / holding_period_years) - 1
        mean_return = np.mean(return_for_period)
        mean_annual_return = np.mean(annualised_return)




        #sharpe ratio
        rf_daily = rf_annual / 252

        sharpe_daily = (mean_daily_return_for_portfolio - rf_daily) / std_daily_return
        sharpe_annual = sharpe_daily * np.sqrt(252)

        #VaR:
        last_period_returns = portfolio_sims[-1:]
        initial_portfolio_value = 100
        portfolio_returns = (last_period_returns - initial_portfolio_value) / initial_portfolio_value
        VaR = np.percentile(portfolio_returns, 5)
        VaR_final = abs(VaR) * initial_portfolio_value
        #CVaR
        worst_losses = portfolio_returns[portfolio_returns <= VaR]
        CVaR_final = abs(worst_losses.mean()) * initial_portfolio_value

        VaR_final, CVaR_final = calculate_var_cvar(daily_returns, 
                                                 confidence_level=0.05, 
                                                 initial_value=initial_portfolio_value)


        stat_results = pd.DataFrame({'annualised_return': [mean_annual_return],
                                     'mean_period_return': [mean_return],
                                     'sharpe_annualized': [sharpe_annual],
                                     'VaR': [VaR_final],
                                     })

        subset_statistics_df = pd.concat([subset_statistics_df, stat_results])

    subset_statistics_df = subset_statistics_df.reset_index(drop=True)

    #RUN THE NORMALITY TEST
    results_normality_test = {}
    for col in subset_statistics_df.columns:
        stat, p_value = normaltest(subset_statistics_df[col])
        results_normality_test[col] = {'statistic': stat, 'p_value': p_value}

    normality_results_df = pd.DataFrame(results_normality_test).T
    normality_results_df['normal'] = normality_results_df['p_value'] > 0.05


    return simulations_results_dict, subset_statistics_df, normality_results_df







def kruskal_anova_test(subset_stats_dfs:dict, metrics='all', test='anova'):
    if metrics == 'all':
        subset_stats_dfs_list = [x for x in subset_stats_dfs.values()]
        metrics = list(subset_stats_dfs_list[0].columns)

    tests_results = dict()
    for metric in metrics:
        groups = [subset_df[metric] for k, subset_df in subset_stats_dfs.items()]
    
        if test == 'anova':
            test_stat, test_p = f_oneway(*groups)
        elif test == 'kruskal':
            test_stat, test_p = kruskal(*groups)

        tests_results[metric] = {'test_stat': round(float(test_stat), 4), 'test_p': round(float(test_p), 4)}
    
    return pd.DataFrame(tests_results).T




def dunn_bonferroni(subset_stats_dfs:dict, metrics='all'):
    if metrics == 'all':
        subset_stats_dfs_list = [x for x in subset_stats_dfs.values()]
        metrics = list(subset_stats_dfs_list[0].columns)

    dunn_tables_results = dict()
    for metric in metrics:

        group_list = list()
        group_labels = list()
        for i, subset_dict_name in enumerate(subset_stats_dfs):
            group = subset_stats_dfs[subset_dict_name][metric]
            group_list.append(group)
            group_labels.extend([subset_dict_name.replace(' Stats', '')] * len(group))
        

        data = pd.concat(group_list, ignore_index=True)


        df = pd.DataFrame({'value': data, 'group': group_labels})
        result = sp.posthoc_dunn(df, val_col='value', group_col='group', p_adjust='bonferroni')    
        result = result.astype(float).round(4)

        dunn_tables_results[metric] = result

    
    return dunn_tables_results




def estimate_t_df_for_portfolio(returns_df):
    """
    Estimate degrees of freedom for each asset, then take the average
    """
    
    asset_dfs = []
    for column in returns_df.columns:
        returns = returns_df[column].values
        standardized_returns = (returns - np.mean(returns)) / np.std(returns)

        def neg_ll(df):
            return -np.sum(stats.t.logpdf(standardized_returns, df))

        result = optimize.minimize_scalar(neg_ll, bounds=(1, 30), method='bounded')
        asset_dfs.append(result.x)
    

    avg_df = np.mean(asset_dfs)
    min_df = np.min(asset_dfs)
    
    print(f"Average df: {avg_df:.2f}, Min df: {min_df:.2f}")
    
    return avg_df  