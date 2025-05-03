import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import normaltest

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns








def reoptimize_weights(df_prices, portfolio_set, how='max_sharpe', min_weight=0.01):
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

            ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, 1))
            weights = ef.max_sharpe()
            new_portfolio = ef.clean_weights()

        new_set[key] = new_portfolio
        
    return new_set
    

def run_simulation(portfolio_dict:dict, returns_for_portfolio:pd.DataFrame, n_sims=100, t=100, distribution_model='multivar_norm', plot=False, initialPortfolio=100):
    mean_returns = returns_for_portfolio.mean()
    cov_matrix = returns_for_portfolio.cov()

    weights = [v for _, v in portfolio_dict.items()]

    meanM = np.tile(mean_returns, (t, 1))  # Shape: (T, n_assets)

    portfolio_sims = np.zeros((t, n_sims))

    L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition

    for sim in range(n_sims):

        if distribution_model == 'bootstrap':
            sampled_returns = returns_for_portfolio.sample(n=t, replace=True).values
            portfolio_returns = sampled_returns @ weights
        else:

            if distribution_model == 'multivar_norm':
                Z = np.random.normal(size=(t, len(portfolio_dict)))  # Shape: (T, n_assets)
                daily_returns = meanM + Z @ L.T  # Shape: (T, n_assets)
            elif distribution_model == 'multivar_t':
                df = 5  # degrees of freedom
                Z = np.random.normal(size=(t, len(portfolio_dict)))
                chi2 = np.random.chisquare(df, size=(t, 1))
                Z_t = Z / np.sqrt(chi2 / df)  # now Z_t has t-distributed marginals

                daily_returns = meanM + Z_t @ L.T

            portfolio_returns = daily_returns @ weights  # Shape: (T,)
        
        
        
        portfolio_sims[:, sim] = np.cumprod(1 + portfolio_returns) * initialPortfolio


    if plot:
        plt.plot(portfolio_sims)
        plt.title("Monte Carlo Simulated Portfolio Paths")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        plt.show()

    return portfolio_sims




#Simulations for the whole subset

def simulate_evaluate_portfolio_subset(portfolios_subset:dict, return_df, n_sims=100, t=100, distribution_model='multivar_norm', rf_annual=0.04):
    simulations_results_dict = dict()
    subset_statistics_df = pd.DataFrame()

    for i, portfolio_dict in portfolios_subset.items():
        returns_portfolio = return_df[list(portfolio_dict.keys())]
        portfolio_sims = run_simulation(portfolio_dict, returns_portfolio, n_sims=n_sims, t=t, distribution_model=distribution_model, plot=False)

        simulations_results_dict[i] = portfolio_sims

        #CALCULATE STATISTICS PER PORTFOLIO:
        daily_returns = (portfolio_sims[1:, :] - portfolio_sims[:-1, :]) / portfolio_sims[:-1, :] #DO WE WANT TO ADD FIRST ROW OF 100 VALUES SO THAT WE HAVE A FULL THING?
        #average cumulative return 
        cumulative_returns_per_simulation = np.sum(daily_returns, axis=0)
        mean_cumulative_return_for_portfolio = np.mean(cumulative_returns_per_simulation)
        #average daily return
        mean_daily_return_for_portfolio = np.mean(daily_returns)
        #STD
        std_cumulative_return = np.std(cumulative_returns_per_simulation) # path uncertainty
        std_daily_return = np.std(daily_returns) #return variability per day
        #sharpe ratio
        rf_daily = rf_annual / 252
        rf_cumulative = (1 + rf_annual) ** (daily_returns.shape[0] / 252) - 1
        sharpe_daily = (mean_daily_return_for_portfolio - rf_daily) / std_daily_return
        sharpe_cumulative = (mean_cumulative_return_for_portfolio - rf_cumulative)  / std_cumulative_return
        sharpe_annual = sharpe_daily * np.sqrt(252)
        sharpe_cumulative_annual = sharpe_cumulative * np.sqrt(252)

        #Sortino ratio
        downside_returns = np.minimum(0, daily_returns - rf_daily)
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        if downside_std == 0:
            sortino_ratio = np.nan
        else:
            sortino_ratio = (mean_daily_return_for_portfolio - rf_daily) / downside_std
        
        sortino_annual = sortino_ratio * np.sqrt(252)

        #VaR:
        last_period_returns = portfolio_sims[-1:]
        initial_portfolio_value = 100
        portfolio_returns = (last_period_returns - initial_portfolio_value) / initial_portfolio_value
        VaR = np.percentile(portfolio_returns, 5)
        VaR_final = abs(VaR) * initial_portfolio_value
        #CVaR
        worst_losses = portfolio_returns[portfolio_returns <= VaR]
        CVaR_final = abs(worst_losses.mean()) * initial_portfolio_value


        stat_results = pd.DataFrame({'mean_cumulative_return': [mean_cumulative_return_for_portfolio],
                                     'mean_daily_return': [mean_daily_return_for_portfolio],
                                     'std_cumulative_return': [std_cumulative_return],
                                     'std_daily_return': [std_daily_return],
                                     'sharpe_daily': [sharpe_daily],
                                     'sharpe_cumulative': [sharpe_cumulative],
                                     'sharpe_annual': [sharpe_annual],
                                     'sharpe_cumulative_annual': [sharpe_cumulative_annual], 
                                     'VaR': [VaR_final],
                                     'CVaR': [CVaR_final],
                                     'sortino': [sortino_ratio],
                                     'sortino_annual': [sortino_annual]})

        subset_statistics_df = pd.concat([subset_statistics_df, stat_results])

    subset_statistics_df = subset_statistics_df.reset_index(drop=True)

    #RUN THE NORMALITY TEST
    results_normality_test = {}
    for col in subset_statistics_df.columns:
        stat, p_value = normaltest(subset_statistics_df[col])
        results_normality_test[col] = {'statistic': stat, 'p_value': p_value}

    normality_results_df = pd.DataFrame(results_normality_test).T
    normality_results_df['normal'] = normality_results_df['p_value'] > 0.05  # True if data is likely normal

    print('Normality Test results: \n')
    print(normality_results_df)

    return simulations_results_dict, subset_statistics_df, normality_results_df
