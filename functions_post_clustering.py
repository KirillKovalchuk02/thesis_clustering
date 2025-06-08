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
def select_cryptos_from_subclusters(existing_stocks, crypto_candidates, crypto_cluster_assignments, df_prices, 
                                   n_cryptos=3, selection_method='random', rf_rate=0.04):
    """
    Select cryptocurrencies from different crypto subclusters.
    
    Parameters:
    - existing_stocks: List of stock tickers in current portfolio
    - crypto_candidates: List of crypto tickers
    - crypto_cluster_assignments: Dictionary mapping crypto tickers to their subcluster IDs
    - df_prices: DataFrame with price data
    - n_cryptos: Number of cryptos to select (should match number of subclusters)
    - selection_method: 'random' or 'correlation'
    - rf_rate: Risk-free rate for calculations
    
    Returns:
    - Dictionary with equal weights for full portfolio
    """
    
    # Group cryptos by their subclusters
    crypto_by_subcluster = {}
    for crypto in crypto_candidates:
        if crypto in crypto_cluster_assignments:
            cluster = crypto_cluster_assignments[crypto]
            if cluster not in crypto_by_subcluster:
                crypto_by_subcluster[cluster] = []
            crypto_by_subcluster[cluster].append(crypto)
    
    available_subclusters = list(crypto_by_subcluster.keys())
    
    # Ensure we don't try to select more cryptos than subclusters
    n_cryptos = min(n_cryptos, len(available_subclusters))
    
    selected_cryptos = []
    
    if selection_method == 'random':
        # Randomly select one crypto from each subcluster
        for i, cluster in enumerate(available_subclusters[:n_cryptos]):
            selected_crypto = random.choice(crypto_by_subcluster[cluster])
            selected_cryptos.append(selected_crypto)
            
    elif selection_method == 'correlation':
        # Calculate portfolio returns for correlation analysis (SAME AS select_cryptos_by_correlation)
        returns = df_prices.pct_change().dropna()
        available_stocks = [stock for stock in existing_stocks if stock in returns.columns]
        
        if not available_stocks:
            # Fallback to random if no stock data available
            for cluster in available_subclusters[:n_cryptos]:
                selected_cryptos.append(random.choice(crypto_by_subcluster[cluster]))
        else:
            # Create equal-weighted portfolio returns (SAME LOGIC)
            portfolio_returns = returns[available_stocks].mean(axis=1)
            
            # For each subcluster, find crypto with lowest correlation to portfolio
            for cluster in available_subclusters[:n_cryptos]:
                cluster_cryptos = crypto_by_subcluster[cluster]
                
                # Calculate correlations for all cryptos in this cluster (SAME LOGIC)
                crypto_correlations = {}
                
                for crypto in cluster_cryptos:
                    if crypto in returns.columns:
                        # Align time series for correlation calculation (SAME AS BEFORE)
                        aligned_portfolio, aligned_crypto = portfolio_returns.align(returns[crypto], join='inner')
                        
                        if len(aligned_portfolio) > 30:  # Minimum data points for reliable correlation
                            corr = aligned_portfolio.corr(aligned_crypto)
                            crypto_correlations[crypto] = abs(corr) if not pd.isna(corr) else 1.0
                        else:
                            crypto_correlations[crypto] = 1.0  # High correlation if insufficient data
                    else:
                        crypto_correlations[crypto] = 1.0  # High correlation if data missing
                
                # Select crypto with lowest correlation in this cluster
                if crypto_correlations:
                    best_crypto = min(crypto_correlations.keys(), key=lambda crypto: crypto_correlations[crypto])
                    selected_cryptos.append(best_crypto)
                else:
                    # Fallback to random if no valid correlations
                    selected_cryptos.append(random.choice(cluster_cryptos))
    
    else:
        raise ValueError("selection_method must be 'random' or 'correlation'")
    
    # Create full portfolio with equal weights
    full_portfolio = existing_stocks + selected_cryptos
    total_assets = len(full_portfolio)
    equal_weight = 1.0 / total_assets
    
    return {ticker: equal_weight for ticker in full_portfolio}

# def select_complementary_cryptos(existing_stocks, crypto_candidates, cluster_assignments, df_prices, n_cryptos=3, verbose=False, rf_rate=0.02, selection_metric='sharpe'):
#     """
#     Select cryptocurrencies to complement an existing stock portfolio based on
#     cluster diversification, with a special case for when all cryptos are in the same cluster.
    
#     Parameters:
#     - existing_stocks: List of stock tickers in the current portfolio
#     - crypto_candidates: List of potential crypto assets to choose from
#     - cluster_assignments: Dictionary mapping each asset to its cluster ID
#     - df_prices: DataFrame with price data for metric calculations
#     - n_cryptos: Number of cryptocurrencies to select (default: 3)
#     - verbose: Whether to print cluster distribution info
#     - rf_rate: Risk-free rate for Sharpe ratio calculation
#     - selection_metric: 'sharpe' for Sharpe ratio or 'return' for raw returns
    
#     Returns:
#     - Tuple: (selected_cryptos, full_new_portfolio)
#     """
#     if verbose:
#         clusters_dict = {i: cluster_assignments[i] for i in existing_stocks}
#         print('Cluster Distribution in the original portfolio: \n')
#         print(pd.DataFrame(columns = ['ticker', 'cluster'], data=clusters_dict.items()).groupby('cluster').count())

#     # Calculate metrics based on selection_metric parameter
#     if selection_metric == 'sharpe':
#         returns_data = dict(sharpe_ratio_calculation(df_prices, rf_rate_annual=rf_rate))
#         metric_name = "Sharpe ratio"
#     elif selection_metric == 'return':
#         # Calculate annualized returns
#         returns_data = {}
#         daily_returns = df_prices.pct_change().dropna()
#         for asset in crypto_candidates + existing_stocks:
#             if asset in daily_returns.columns:
#                 annual_return = daily_returns[asset].mean() * 252  # Annualize daily returns
#                 returns_data[asset] = annual_return
#             else:
#                 returns_data[asset] = 0  # Default value if asset not found
#         metric_name = "annualized return"
#     else:
#         raise ValueError("selection_metric must be either 'sharpe' or 'return'")
    
#     if verbose:
#         print(f"Selection will be based on {metric_name}")
    
#     # Step 1: Identify clusters already represented in the portfolio
#     stock_clusters = set(cluster_assignments[stock] for stock in existing_stocks)
    
#     # Step 2: Check crypto cluster diversity
#     crypto_clusters = set(cluster_assignments[crypto] for crypto in crypto_candidates)
    
#     # Special case: All cryptos are in the same cluster - select by highest metric values
#     if len(crypto_clusters) == 1:
#         if verbose:
#             print(f"All cryptos in same cluster - selecting top {n_cryptos} by {metric_name}")
        
#         sorted_by_metric = sorted(
#             crypto_candidates,
#             key=lambda crypto: returns_data[crypto],
#             reverse=True
#         )
#         selected = sorted_by_metric[:n_cryptos]
#         return selected, existing_stocks + selected
    
#     # Step 3: Group crypto candidates by their cluster
#     crypto_by_cluster = {}
#     for crypto in crypto_candidates:
#         cluster = cluster_assignments[crypto]
#         if cluster not in crypto_by_cluster:
#             crypto_by_cluster[cluster] = []
#         crypto_by_cluster[cluster].append(crypto)
    
#     # Step 4: Select cryptos from unrepresented clusters first (diversification priority)
#     selected_cryptos = []
#     unrepresented_clusters = set(crypto_by_cluster.keys()) - stock_clusters
    
#     if verbose and unrepresented_clusters:
#         print(f"Prioritizing unrepresented clusters: {sorted(unrepresented_clusters)}")
    
#     # Sort unrepresented clusters by the best metric value in each cluster
#     cluster_best_metrics = {
#         cluster: max(returns_data[crypto] for crypto in cryptos)
#         for cluster, cryptos in crypto_by_cluster.items()
#         if cluster in unrepresented_clusters
#     }
    
#     sorted_unrepresented_clusters = sorted(
#         unrepresented_clusters, 
#         key=lambda cluster: cluster_best_metrics[cluster],
#         reverse=True
#     )
    
#     # For each unrepresented cluster, select the crypto with the best metric value
#     for cluster in sorted_unrepresented_clusters:
#         if len(selected_cryptos) >= n_cryptos:
#             break
            
#         # Choose the crypto with the highest metric value from this cluster
#         best_crypto = max(
#             crypto_by_cluster[cluster],
#             key=lambda crypto: returns_data[crypto]
#         )
#         selected_cryptos.append(best_crypto)
        
#         if verbose:
#             print(f"Selected {best_crypto} from cluster {cluster} ({metric_name}: {returns_data[best_crypto]:.4f})")
    
#     # Step 5: If we still need more cryptos, select remaining ones by highest metric values
#     if len(selected_cryptos) < n_cryptos:
#         remaining_cryptos = [
#             crypto for crypto in crypto_candidates 
#             if crypto not in selected_cryptos
#         ]
        
#         sorted_remaining = sorted(
#             remaining_cryptos,
#             key=lambda crypto: returns_data[crypto],
#             reverse=True
#         )
        
#         needed = n_cryptos - len(selected_cryptos)
#         additional_selections = sorted_remaining[:needed]
#         selected_cryptos.extend(additional_selections)
        
#         if verbose and additional_selections:
#             print(f"Added {len(additional_selections)} cryptos by best {metric_name}:")
#             for crypto in additional_selections:
#                 cluster = cluster_assignments[crypto]
#                 print(f"  {crypto} (cluster {cluster}, {metric_name}: {returns_data[crypto]:.4f})")
        
#     full_new_portfolio = existing_stocks + selected_cryptos
#     return selected_cryptos, full_new_portfolio




# def supplement_set_with_cryptos(portfolio_set:dict, cryptos_list, tickers_with_labels, df_prices, selection_metric='sharpe', n_cryptos=3, seed=30, random_alloc=False):
#     """
#     Loop based function based on "select_complementary_cryptos"
#     """
#     random.seed(seed)
#     new_portfolios_w_cryptos = dict()
    
#     for key, portfolio in portfolio_set.items():
#         if random_alloc:
#             existing_stocks = list(portfolio.keys())
#             indices = random.sample(range(len(cryptos_list)), n_cryptos)
#             cryptos = [cryptos_list[i] for i in indices]
#             crypto_supplemented_port = existing_stocks + cryptos
#         else:
#             existing_stocks = list(portfolio.keys())
#             cryptos, crypto_supplemented_port = select_complementary_cryptos(existing_stocks=existing_stocks, crypto_candidates=cryptos_list, cluster_assignments=tickers_with_labels, 
#                                                                               df_prices = df_prices, n_cryptos=n_cryptos, verbose=False, selection_metric=selection_metric)
        
#         new_portfolios_w_cryptos[key] = crypto_supplemented_port

#     return new_portfolios_w_cryptos




def select_cryptos_by_correlation(existing_stocks, crypto_candidates, df_prices, n_cryptos=3, selection_metric='sharpe', rf_rate=0.04):
    """
    Select cryptocurrencies with lowest correlation to existing portfolio.
    Returns dictionary with equal weights.
    """
    returns = df_prices.pct_change().dropna()
    
    # Calculate portfolio returns (equal weighted)
    available_stocks = [stock for stock in existing_stocks if stock in returns.columns]
    if not available_stocks:
        # Fallback to random selection if no stock data
        selected_cryptos = random.sample(crypto_candidates, n_cryptos)
        crypto_supplemented_port = existing_stocks + selected_cryptos
        total_assets = len(crypto_supplemented_port)
        equal_weight = 1.0 / total_assets
        return {ticker: equal_weight for ticker in crypto_supplemented_port}
    
    portfolio_returns = returns[available_stocks].mean(axis=1)  # Equal weighted portfolio
    
    # Calculate selection metric
    if selection_metric == 'sharpe':
        metric_data = dict(sharpe_ratio_calculation(df_prices, rf_rate_annual=rf_rate))
    else:
        metric_data = {asset: returns[asset].mean() * 252 for asset in crypto_candidates if asset in returns.columns}
    
    # Calculate correlation of each crypto with portfolio
    crypto_correlations = {}
    
    for crypto in crypto_candidates:
        if crypto in returns.columns:
            # Align time series for correlation calculation
            aligned_portfolio, aligned_crypto = portfolio_returns.align(returns[crypto], join='inner')
            
            if len(aligned_portfolio) > 30:  # Minimum data points for reliable correlation
                corr = aligned_portfolio.corr(aligned_crypto)
                crypto_correlations[crypto] = abs(corr) if not pd.isna(corr) else 1.0
            else:
                crypto_correlations[crypto] = 1.0  # High correlation if insufficient data
        else:
            crypto_correlations[crypto] = 1.0
    
    # Filter cryptos that have valid metrics
    valid_cryptos = [crypto for crypto in crypto_candidates if crypto in metric_data]
    
    # Sort by correlation (ascending), then by metric (descending) for tie-breaking
    sorted_cryptos = sorted(
        valid_cryptos,
        key=lambda crypto: (crypto_correlations[crypto], -metric_data.get(crypto, -999))
    )
    
    selected_cryptos = sorted_cryptos[:n_cryptos]
    crypto_supplemented_port = existing_stocks + selected_cryptos
    
    # Create equal-weighted dictionary
    total_assets = len(crypto_supplemented_port)
    equal_weight = 1.0 / total_assets
    weighted_portfolio = {ticker: equal_weight for ticker in crypto_supplemented_port}
    
    return weighted_portfolio


def supplement_set_with_cryptos(portfolio_set: dict, cryptos_list, tickers_with_labels, df_prices,
                                selection_metric='sharpe', 
                               selection_method='clustering', selection_method_clusters='random', 
                               n_cryptos=3, seed=30, rf_rate=0.04):
    """
    Loop based function to supplement portfolios with cryptocurrencies using various selection methods.
    
    Parameters:
    - portfolio_set: Dictionary of portfolios to supplement
    - cryptos_list: List of available cryptocurrencies
    - tickers_with_labels: Cluster assignments (only used for clustering method)
    - df_prices: Price data DataFrame
    - selection_method: 'random', 'clustering', or 'correlation'
    - selection_metric: 'sharpe' or 'return' (for performance-based selection)
    - n_cryptos: Number of cryptos to add
    - seed: Random seed for reproducibility
    - rf_rate: Risk-free rate for Sharpe calculation
    
    Returns:
    - Dictionary of portfolios supplemented with cryptocurrencies
    """
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
            
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}. Use 'random', 'clustering', or 'correlation'")
        
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
        cumulative_returns_per_simulation = calculate_cumulative_returns(daily_returns)
        mean_daily_return_for_portfolio = np.mean(daily_returns)
        #STD
        std_daily_return = np.std(daily_returns) #return variability per day


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