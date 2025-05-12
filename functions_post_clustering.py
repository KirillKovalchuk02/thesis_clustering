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



#CLAUDE's algorithm for selecting complementing cryptos:
#After this we want to reoptimize the whole portfolio for max sharpe or min var

def select_complementary_cryptos(existing_stocks, crypto_candidates, cluster_assignments, df_prices, n_cryptos=3, verbose=False, rf_rate=0.02):
    """
    Select cryptocurrencies to complement an existing stock portfolio based on
    cluster diversification, with a special case for when all cryptos are in the same cluster.
    
    Parameters:
    - existing_stocks: List of stock tickers in the current portfolio
    - crypto_candidates: List of potential crypto assets to choose from
    - cluster_assignments: Dictionary mapping each asset to its cluster ID
    - returns_data: Dictionary mapping each asset to its return metric
    - n_cryptos: Number of cryptocurrencies to select (default: 3)
    
    Returns:
    - List of selected crypto assets
    """
    if verbose:
        clusters_dict = {i: cluster_assignments[i] for i in existing_stocks}
        print('Cluster Distribution in the original portfolio: \n')
        print(pd.DataFrame(columns = ['ticker', 'cluster'], data=clusters_dict.items()).groupby('cluster').count())

    returns_data = dict(sharpe_ratio_calculation(df_prices, rf_rate_annual = rf_rate))
    # Step 1: Identify clusters already represented in the portfolio
    stock_clusters = set(cluster_assignments[stock] for stock in existing_stocks)
    
    # Step 2: Check crypto cluster diversity
    crypto_clusters = set(cluster_assignments[crypto] for crypto in crypto_candidates)
    
    # Special case: All cryptos are in the same cluster
    if len(crypto_clusters) == 1:
        sorted_by_return = sorted(
            crypto_candidates,
            key=lambda crypto: returns_data[crypto],
            reverse=True
        )
        return sorted_by_return[:n_cryptos]
    
    # Step 3: Group crypto candidates by their cluster
    crypto_by_cluster = {}
    for crypto in crypto_candidates:
        cluster = cluster_assignments[crypto]
        if cluster not in crypto_by_cluster:
            crypto_by_cluster[cluster] = []
        crypto_by_cluster[cluster].append(crypto)
    
    # Step 4: Select cryptos from unrepresented clusters first
    selected_cryptos = []
    unrepresented_clusters = set(crypto_by_cluster.keys()) - stock_clusters
    
    # Sort unrepresented clusters by the best return in each cluster
    cluster_best_returns = {
        cluster: max(returns_data[crypto] for crypto in cryptos)
        for cluster, cryptos in crypto_by_cluster.items()
        if cluster in unrepresented_clusters
    }
    
    sorted_unrepresented_clusters = sorted(
        unrepresented_clusters, 
        key=lambda cluster: cluster_best_returns[cluster],
        reverse=True
    )
    
    # For each unrepresented cluster, select the crypto with the best return
    for cluster in sorted_unrepresented_clusters:
        if len(selected_cryptos) >= n_cryptos:
            break
            
        # Choose the crypto with the best return from this cluster
        best_crypto = max(
            crypto_by_cluster[cluster],
            key=lambda crypto: returns_data[crypto]
        )
        selected_cryptos.append(best_crypto)
    
    # Step 5: If we still need more cryptos, use return metrics for selection
    if len(selected_cryptos) < n_cryptos:
        remaining_cryptos = [
            crypto for crypto in crypto_candidates 
            if crypto not in selected_cryptos
        ]
        
        sorted_remaining = sorted(
            remaining_cryptos,
            key=lambda crypto: returns_data[crypto],
            reverse=True
        )
        
        needed = n_cryptos - len(selected_cryptos)
        selected_cryptos.extend(sorted_remaining[:needed])
        
    full_new_portfolio = existing_stocks + selected_cryptos
    return selected_cryptos, full_new_portfolio




def supplement_set_with_cryptos(portfolio_set:dict, cryptos_list, tickers_with_labels, df_prices, n_cryptos=3, seed=30, random_alloc=False):
    """
    Loop based function based on "select_complementary_cryptos"
    """
    random.seed(seed)
    new_portfolios_w_cryptos = dict()
    
    for key, portfolio in portfolio_set.items():
        if random_alloc:
            existing_stocks = list(portfolio.keys())
            indices = random.sample(range(len(cryptos_list)), n_cryptos)
            cryptos = [cryptos_list[i] for i in indices]
            crypto_supplemented_port = existing_stocks + cryptos
        else:
            existing_stocks = list(portfolio.keys())
            cryptos, crypto_supplemented_port = select_complementary_cryptos(existing_stocks=existing_stocks, crypto_candidates=cryptos_list, cluster_assignments=tickers_with_labels, 
                                                                              df_prices = df_prices, n_cryptos=n_cryptos, verbose=False)
        
        new_portfolios_w_cryptos[key] = crypto_supplemented_port

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
        
            mu = expected_returns.mean_historical_return(prices_df)  # Expected returns
            S = risk_models.sample_cov(prices_df)  # Covariance matrix
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

    L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition

    for sim in range(n_sims):

        if distribution_model == 'bootstrap':
            sampled_returns = returns_for_portfolio.bfill().sample(n=t, replace=True).values
            portfolio_returns = sampled_returns @ weights
        elif distribution_model in ['multivar_norm', 'multivar_t']:

            if distribution_model == 'multivar_norm':
                Z = np.random.normal(size=(t, len(portfolio_dict)))  # Shape: (T, n_assets)
                daily_returns = meanM + Z @ L.T  # Shape: (T, n_assets)
            elif distribution_model == 'multivar_t':
                df = 25  # degrees of freedom
                Z = np.random.normal(size=(t, len(portfolio_dict)))
                chi2 = np.random.chisquare(df, size=(t, 1))
                Z_t = Z / np.sqrt(chi2 / df)  # now Z_t has t-distributed marginals

                daily_returns = meanM + Z_t @ L.T

            portfolio_returns = daily_returns @ weights  # Shape: (T,)
        
        else:
            break
            
        
        
        
        portfolio_sims[:, sim] = np.cumprod(1 + portfolio_returns) * initialPortfolio


    if plot:
        plt.plot(portfolio_sims)
        plt.title("Monte Carlo Simulated Portfolio Paths")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        plt.show()

    return portfolio_sims


def calculate_cumulative_returns(daily_returns):
    # Add 1 to convert returns to growth factors
    growth_factors = daily_returns + 1
    
    # Calculate the cumulative product along the time axis
    cumulative_growth = np.cumprod(growth_factors, axis=0)
    
    # Calculate the cumulative return (final value - 1)
    cumulative_returns = cumulative_growth[-1, :] - 1
    
    return cumulative_returns




def calculate_var_cvar(daily_returns, confidence_level=0.05, initial_value=100):
        # Sort returns from worst to best
        sorted_returns = np.sort(daily_returns.flatten())
        
        # Calculate VaR
        var_index = int(confidence_level * len(sorted_returns))
        var = abs(sorted_returns[var_index])
        
        # Calculate CVaR (Expected Shortfall)
        cvar = abs(np.mean(sorted_returns[:var_index]))
        
        # Convert to monetary values - NOPE
        var_value = var #* initial_value
        cvar_value = cvar #* initial_value
        
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
        daily_returns = (portfolio_sims[1:, :] - portfolio_sims[:-1, :]) / portfolio_sims[:-1, :] #DO WE WANT TO ADD FIRST ROW OF 100 VALUES SO THAT WE HAVE A FULL THING?
        #average cumulative return 
        cumulative_returns_per_simulation = calculate_cumulative_returns(daily_returns)
        #mean_cumulative_return_for_portfolio = np.mean(cumulative_returns_per_simulation)
        #average daily return
        mean_daily_return_for_portfolio = np.mean(daily_returns)
        #STD
        std_daily_return = np.std(daily_returns) #return variability per day

        #annualised_return = (1 + mean_cumulative_return_for_portfolio)**(252/t) - 1
        #annualised_volatility = std_cumulative_return * np.sqrt(252/t)




        final_portfolio_values = portfolio_sims[-1, :]  # Final values across all simulations
        initial_value = 100  # Your initial portfolio value
        holding_period_years = t / 252  # Convert days to years

        # Calculate annualized returns 
        return_for_period = (final_portfolio_values - initial_value) / initial_value
        annualised_return = (final_portfolio_values / initial_value) ** (1 / holding_period_years) - 1
        #annualised_return = (1 + return_for_period) ** (1/holding_period_years) - 1
        mean_return = np.mean(return_for_period)
        mean_annual_return = np.mean(annualised_return)




        #sharpe ratio
        rf_daily = rf_annual / 252
        #rf_cumulative = (1 + rf_annual) ** (daily_returns.shape[0] / 252) - 1
        sharpe_daily = (mean_daily_return_for_portfolio - rf_daily) / std_daily_return
        #sharpe_cumulative = (mean_cumulative_return_for_portfolio - rf_cumulative)  / std_cumulative_return
        sharpe_annual = sharpe_daily * np.sqrt(252)
        #sharpe_cumulative_annual = sharpe_cumulative * np.sqrt(252)

        #Sortino ratio
        # downside_returns = np.minimum(0, daily_returns - rf_daily)

        # with np.errstate(over='ignore'):  # Suppress the warning
        #     squared_returns = np.square(downside_returns, dtype=np.float64)
        #     downside_std = np.sqrt(np.mean(squared_returns))

        # if np.isclose(downside_std, 0) or np.isnan(downside_std):
        #     sortino_ratio = np.nan  # or set to a default value
        # else:
        #     sortino_ratio = (mean_daily_return_for_portfolio - rf_daily) / downside_std
        
        #sortino_annual = sortino_ratio * np.sqrt(252)

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
                                     #'annualised_volatility': [annualised_volatility],
                                     #'std_daily_return': [std_daily_return],
                                     #'sharpe_daily': [sharpe_daily],
                                     #'sharpe_cumulative': [sharpe_cumulative],
                                     'sharpe_annualized': [sharpe_annual],
                                     'VaR': [VaR_final],
                                     #'CVaR': [CVaR_final],
                                     #'sortino': [sortino_ratio],
                                     #'sortino_annualized': [sortino_annual]
                                     })

        subset_statistics_df = pd.concat([subset_statistics_df, stat_results])

    subset_statistics_df = subset_statistics_df.reset_index(drop=True)

    #RUN THE NORMALITY TEST
    results_normality_test = {}
    for col in subset_statistics_df.columns:
        stat, p_value = normaltest(subset_statistics_df[col])
        results_normality_test[col] = {'statistic': stat, 'p_value': p_value}

    normality_results_df = pd.DataFrame(results_normality_test).T
    normality_results_df['normal'] = normality_results_df['p_value'] > 0.05  # True if data is likely normal

    #print('Normality Test results: \n')
    #print(normality_results_df)    





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
    from scipy import stats
    import numpy as np
    from scipy import optimize
    
    asset_dfs = []
    
    for column in returns_df.columns:
        returns = returns_df[column].values
        
        # Standardize the returns
        standardized_returns = (returns - np.mean(returns)) / np.std(returns)
        
        # Define negative log-likelihood function for t-distribution
        def neg_ll(df):
            return -np.sum(stats.t.logpdf(standardized_returns, df))
        
        # Find optimal degrees of freedom
        result = optimize.minimize_scalar(neg_ll, bounds=(1, 30), method='bounded')
        
        asset_dfs.append(result.x)
    
    # Return the average df (or minimum for more conservative estimate)
    avg_df = np.mean(asset_dfs)
    min_df = np.min(asset_dfs)
    
    print(f"Average df: {avg_df:.2f}, Min df: {min_df:.2f}")
    
    return avg_df  # or return min_df