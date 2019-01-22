import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
import sympy as sm

def portfolio_annual_data(weights, mean_of_returns, cov_matrix):
    returns = np.dot(mean_of_returns, weights) * 12
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)
    return std, returns


def portfolio_volatility(weights, mean_of_returns, cov_matrix):
    return portfolio_annual_data(weights, mean_of_returns, cov_matrix)[0]

def min_variance(mean_of_returns, cov_matrix, i):
    num_assets = 10
    args = (mean_of_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    if(i == 1):
        bound = (0.0, 0.45)
    if(i == 2):
        bound = (0.0, 0.48)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args = args, method = 'SLSQP',
                          bounds = bounds, constraints = constraints)
    return result

def efficient_return(mean_of_returns, cov_matrix, target):
    num_assets = 10
    args = (mean_of_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annual_data(weights, mean_of_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_of_returns, cov_matrix, returns_range):
    points = []
    for r in returns_range:
        points.append(efficient_return(mean_of_returns, cov_matrix, r))
    return points

def display(mean_of_returns, cov_matrix):
    min_vol = min_variance(mean_of_returns, cov_matrix, 0)
    min_vol2 = min_variance(mean_of_returns, cov_matrix, 1)
    min_vol3 = min_variance(mean_of_returns, cov_matrix, 2)

    initial = [0.1, 0.05, 0.1, 0.05, 0.2, 0.01, 0.05, 0.24, 0.1, 0.1]
    initial_dp = pd.DataFrame(initial)
    initial_p_std, initial_p_r = portfolio_annual_data(initial_dp, mean_of_returns, cov_matrix)

    min_std_p, min_r_p = portfolio_annual_data(min_vol['x'], mean_of_returns, cov_matrix)
    min_vol_partition1 = pd.DataFrame(min_vol.x, index=data.columns, columns=['partition'])
    min_vol_partition1.partition = [round(i,2) for i in min_vol_partition1.partition]
    
    min_vol_partition2 = pd.DataFrame(min_vol2.x, index=data.columns, columns=['partition'])
    min_vol_partition2.partition = [round(i,2) for i in min_vol_partition2.partition]
    
    min_vol_partition3 = pd.DataFrame(min_vol3.x, index=data.columns, columns=['partition'])
    min_vol_partition3.partition = [round(i,2) for i in min_vol_partition3.partition]
    
    print "-"*80
    print "Annualized Return:", round(min_r_p,2)
    print "Annualized Standard Deviation:", round(min_std_p, 2)

    print "\n"
    min_vol_partition1.columns = ["Proportion of weights"]
    min_vol_partition2.columns = ["Proportion of weights"]
    min_vol_partition3.columns = ["Proportion of weights"]

    print min_vol_partition1
    print "\n"
    
    print "-"*80
    print "Annualized Return:", round(min_r_p,2)
    print "Annualized Standard Deviation:", round(min_std_p, 2)

    print "\n"

    print min_vol_partition2
    print "\n"
    
    print "-"*80
    print "Annualized Return:", round(min_r_p,2)
    print "Annualized Standard Deviation:", round(min_std_p, 2)

    print "\n"

    print min_vol_partition3
    plt.figure(figsize=(10, 7))
    plt.scatter(min_std_p, min_r_p, marker='*', color='g', s=500, label='Min Risk')
    plt.scatter(initial_p_std, initial_p_r, marker='*', color='r', s=500, label='Initial portfolio')

    target = np.linspace(min_r_p - 0.05, 0.35, 75)
    efficient_portfolio = efficient_frontier(mean_of_returns, cov_matrix, target)

    plt.plot([p['fun'] for p in efficient_portfolio], target, linestyle='-.', color='black', 
            label='Efficient Frontier')
    plt.title('Efficient Frontier')
    plt.xlabel('Annualized Standard Deviation')
    plt.ylabel('Annualized Return')
    plt.legend(labelspacing = 0.8)
    plt.savefig('ef.png')

plt.style.use('fivethirtyeight')
np.random.seed(53)

quandl.ApiConfig.api_key = 'rNFFK9r6dDbt7zcwEiKR'

symbol = ["NKE", "MSFT", "AMZN", "AAPL", "GOOGL", "JPM", "WMT", "GPS", "ADS", "CVGW"]

start = "2008-03-29"
end = "2018-07-20"
data = quandl.get(["WIKI/NKE", "WIKI/MSFT", "WIKI/AMZN", "WIKI/AAPL", "WIKI/GOOGL",
                    "WIKI/JPM", "WIKI/WMT", "WIKI/GPS", "WIKI/ADS", "WIKI/CVGW"],
                    start_date=start, 
                    end_date=end, 
                    collapse='monthly',
                    column_index = 4
)

data.columns = ["NKE", "MSFT", "AMZN", "AAPL", "GOOGL", "JPM", "WMT", "GPS", "ADS", "CVGW"]
returns_monthly = data.pct_change()
mean_of_returns = returns_monthly.mean()
cov_monthly = returns_monthly.cov()
display(mean_of_returns, cov_monthly)
