import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize

def drawdown(return_series: pd.Series):
    """
    Takes a times series of asset returns
    Computes and returns a DataFrame that contains: 
    the wealth index
    the previous peaks
    percent of drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        'Wealth': wealth_index,
        'Peaks': previous_peaks,
        'Drawdown': drawdowns
    })

def get_ffme_returns():
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv", header=0, index_col=0, parse_dates=True, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m')
    return rets

def get_hfi_returns():
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi

def skewness(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / sigma_r**3

def kurtosis(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / sigma_r**4

def is_normal(r, level=0.01):
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def semideviation(r):
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def var_historic(r, level=5):
    '''
    Calculate historic value at risk. 
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(lambda x: var_historic(x, level=level))
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def cvar_historic(r, level=5):
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(lambda x: cvar_historic(x, level=level))
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def var_gaussian(r, level = 5, modified = False): 
    def compute_var_gaussian(r, level=5, modified=False):
        z = norm.ppf(level/100)
        if modified:
            s = skewness(r)
            k = kurtosis(r)
            z = (z +
                 (z**2 - 1) * s / 6 +
                 (z**3 - 3*z) * (k - 3) / 24 -
                 (2*z**3 - 5*z) * (s**2) / 36)
        return -(r.mean() + z * r.std(ddof=0))
    if isinstance(r, pd.Series):
        return compute_var_gaussian(r, level, modified)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(lambda x: compute_var_gaussian(x, level, modified))
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def get_ind_returns():
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header = 0, index_col = 0, parse_dates = True)/100
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


import numpy as np
import pandas as pd

def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns.

    Parameters:
    r (pd.Series or np.ndarray): Returns for each period.
    periods_per_year (int): Number of periods in a year (e.g., 252 for daily returns, 12 for monthly returns).

    Returns:
    float: Annualized volatility.
    """
    return r.std() * (periods_per_year ** 0.5)

def annualize_rets(r, periods_per_year):
    """
    Annualizes the returns.

    Parameters:
    r (pd.Series or np.ndarray): Returns for each period.
    periods_per_year (int): Number of periods in a year (e.g., 252 for daily returns, 12 for monthly returns).

    Returns:
    float: Annualized return.
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Calculates the annualized Sharpe ratio of a set of returns.

    Parameters:
    r (pd.Series or np.ndarray): Returns for each period.
    riskfree_rate (float): Annual risk-free rate.
    periods_per_year (int): Number of periods in a year (e.g., 252 for daily returns, 12 for monthly returns).

    Returns:
    float: Annualized Sharpe ratio.
    """
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

def portfolio_return(weights, returns):
    """
    Calculates the return of a portfolio.

    Parameters:
    weights (np.ndarray): Weights of each asset in the portfolio.
    returns (np.ndarray or pd.Series): Returns of each asset.

    Returns:
    float: Portfolio return.
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Calculates the volatility (standard deviation) of a portfolio.

    Parameters:
    weights (np.ndarray): Weights of each asset in the portfolio.
    covmat (np.ndarray or pd.DataFrame): Covariance matrix of asset returns.

    Returns:
    float: Portfolio volatility.
    """
    return (weights.T @ covmat @ weights) ** 0.5


def plot_ef2(n_points, er, cov, style = ".-"):
    if er.shape[0] != 2: 
        raise ValueError("plot_erf2 can only plot 2-asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x = "Volatility", y = "Returns", style = style)


def minimize_vol(target_return, er, cov): 
    """
    target_ret -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess, 
                       args = (cov,), 
                       method = "SLSQP", 
                       options = {"disp": False}, 
                       constraints = (return_is_target, weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x


def optimal_weights(n_points, er, cov):
    
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def gmv(cov):
    '''
    returns the weights of the global minimum vol portfolio given the covariance matrix
    '''
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    def minimize_vol(weights, cov):
        return portfolio_vol(weights, cov)

    results = minimize(minimize_vol, init_guess, 
                       args=(cov,), 
                       method='SLSQP', 
                       bounds=bounds, 
                       constraints=[weights_sum_to_1])
    return results.x

def plot_ef(n_points, er, cov, show_cml = False, style = ".-", riskfree_rate = 0, show_ew = False, show_gmv = False):
   
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x = "Volatility", y = "Returns", style = style)

    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color = "midnightblue", marker = "o", markersize = 10)
        
    
    if show_ew: 
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color = "goldenrod", marker = "o", markersize = 10)
        
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color = "green", marker = "o", linestyle = "dashed", markersize = 12, linewidth = 2)

    return ax
    

def msr(riskfree_rate, er, cov): 
    """
    RiskFree rate + ER + COV -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    def neg_sharpe_ratio(weights, riskfree_rate, er, cov): 
        """
        Returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio, init_guess, 
                       args = (riskfree_rate, er, cov,), 
                       method = "SLSQP", 
                       options = {"disp": False}, 
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

