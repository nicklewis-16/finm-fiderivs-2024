import math
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import datetime
import pandas as pd
import matplotlib.pyplot as plt



def normal_cdf(x):
    """
    Calculate the cumulative distribution function (CDF) of a standard normal distribution.

    Parameters:
    x (float): The value at which to evaluate the CDF.

    Returns:
    float: The probability that a random variable from a standard normal distribution is less than or equal to x.
    """
    return (1 + math.erf(x/np.sqrt(2))) / 2


def normal_pdf(x):
    """
    Calculate the probability density function (PDF) of a standard normal distribution.

    Parameters:
    x (float): The value at which to evaluate the PDF.

    Returns:
    float: The PDF value at the given x.

    """
    return np.exp(-x**2/2) / np.sqrt(2*np.pi)


def bs_normargs(under=None,strike=None,T=None,rf=None,vol=None):
    """
    Calculate the normalized arguments for the Black-Scholes formula.

    Args:
        under (float): The underlying asset price.
        strike (float): The strike price of the option.
        T (float): Time to expiration in years.
        rf (float): Risk-free interest rate.
        vol (float): Volatility of the underlying asset.

    Returns:
        list: A list containing the normalized arguments [d1, d2].
    """
    d1 = (np.log(under/strike) + (rf + .5 * vol**2)*T ) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return [d1,d2]

def bs_delta(under=None,strike=None,T=None,rf=None,vol=None):
    """
    Calculate the delta of an option using the Black-Scholes formula.

    Parameters:
    under (float): The current price of the underlying asset.
    strike (float): The strike price of the option.
    T (float): Time to expiration in years.
    rf (float): Risk-free interest rate.
    vol (float): Volatility of the underlying asset.

    Returns:
    float: The delta of the option.
    """
    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]

    return normal_cdf(d1)

def bs_theta(under=None,strike=None,T=None,rf=None,vol=None):
    """
    Calculate the theta (time decay) of a European option using the Black-Scholes formula.

    Parameters:
    under (float): Current price of the underlying asset.
    strike (float): Strike price of the option.
    T (float): Time to expiration in years.
    rf (float): Risk-free interest rate.
    vol (float): Volatility of the underlying asset.

    Returns:
    float: The theta of the option.
    """
    d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol) 
    
    temp = (- under * normal_pdf(d1) * vol)/(2*np.sqrt(T)) - rf * strike * np.exp(-rf*T) * normal_cdf(d2)
    return temp

def bs_gamma(under=None,strike=None,T=None,rf=None,vol=None):
    """
    Calculate the gamma of an option using the Black-Scholes model.

    Parameters:
    under (float): The current price of the underlying asset.
    strike (float): The strike price of the option.
    T (float): Time to expiration in years.
    rf (float): Risk-free interest rate.
    vol (float): Volatility of the underlying asset.

    Returns:
    float: The gamma of the option.
    """
    
    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]
    return normal_pdf(d1) / (under * vol * np.sqrt(T))


def bs_vega(under=None,strike=None,T=None,rf=None,vol=None):
    """
    Calculate the Vega of a European option using the Black-Scholes formula.

    Parameters:
    under (float): The current price of the underlying asset.
    strike (float): The strike price of the option.
    T (float): Time to expiration in years.
    rf (float): Risk-free interest rate.
    vol (float): Volatility of the underlying asset.

    Returns:
    float: The Vega of the option.
    """
    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]
    return normal_pdf(d1) * (under * np.sqrt(T))

def bs_price(under=None,strike=None,T=None,rf=None,vol=None,option='call'):
    """
    Calculate the Black-Scholes price of an option.

    Parameters:
    under (float): The current price of the underlying asset.
    strike (float): The strike price of the option.
    T (float): The time to expiration of the option in years.
    rf (float): The risk-free interest rate.
    vol (float): The volatility of the underlying asset.
    option (str): The type of option, either 'call' or 'put'. Default is 'call'.

    Returns:
    float: The Black-Scholes price of the option.
    """
    d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol) 
    
    if option=='put':
        return np.exp(-rf*T)*strike * normal_cdf(-d2) - under * normal_cdf(-d1)
    else:
        return under * normal_cdf(d1) - np.exp(-rf*T)*strike * normal_cdf(d2)


def bs_rho(under=None,strike=None,T=None,rf=None,vol=None):
    """
    Calculate the rho (sensitivity to interest rate) for a Black-Scholes option.

    Parameters:
    under (float): The current price of the underlying asset.
    strike (float): The strike price of the option.
    T (float): Time to expiration of the option in years.
    rf (float): Risk-free interest rate.
    vol (float): Volatility of the underlying asset.

    Returns:
    float: The rho value of the option.
    """

    d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)
    return normal_cdf(d2) * strike * T * np.exp(-rf*T)



def bs_impvol(under=None,strike=None,T=None,rf=None,option='call',opt_price=None,volGuess=.25,showflag=False):
    """
    Calculate the implied volatility using the Black-Scholes model.

    Parameters:
    under (float): The current price of the underlying asset.
    strike (float): The strike price of the option.
    T (float): Time to expiration in years.
    rf (float): Risk-free interest rate.
    option (str): Type of option, either 'call' or 'put'.
    opt_price (float): Observed market price of the option.
    volGuess (float): Initial guess for the implied volatility.
    showflag (bool): Flag indicating whether to show additional information.

    Returns:
    float or tuple: The implied volatility if showflag is False, otherwise a tuple containing the implied volatility and additional information.
    """
    func = lambda ivol: (opt_price-bs_price(vol=ivol,under=under,strike=strike,T=T,rf=rf,option=option))**2
    xstar, analytics, flag, msg = fsolve(func, volGuess, full_output=True)
    
    if showflag:
        return xstar, msg
    else:
        return xstar
    
    
def to_maturity(expiration=None, current_date=None):
    """
    Calculates the time to maturity in years.

    Parameters:
    expiration (str): The expiration date of the option.
    current_date (str): The current date.

    Returns:
    float: The time to maturity in years.
    """
    return (pd.to_datetime(expiration) - pd.to_datetime(current_date)).total_seconds()/(24*60*60)/365


def filter_stale_quotes(opt_chain):
    """
    Filters out stale quotes from an options chain based on the last trade date.

    Parameters:
    opt_chain (DataFrame): The options chain DataFrame.

    Returns:
    list: A list of indices corresponding to the rows that contain non-stale quotes.
    """
    LDATE = opt_chain.sort_values('lastTradeDate')['lastTradeDate'].iloc[-1]
    mask = list()

    for idx in opt_chain.index:
        dt = opt_chain.loc[idx, 'lastTradeDate']
        if (dt - LDATE).total_seconds() / 3600 > -24:
            mask.append(idx)

    return mask

def clean_options(calls_raw,puts_raw):
    """
    Cleans the options data by filtering out stale quotes and selecting options with high volume.
    
    Parameters:
    calls_raw (DataFrame): Raw data for call options.
    puts_raw (DataFrame): Raw data for put options.
    
    Returns:
    calls (DataFrame): Cleaned data for call options.
    puts (DataFrame): Cleaned data for put options.
    """
    idx = filter_stale_quotes(calls_raw)
    calls = calls_raw.loc[idx,:]
    idx = filter_stale_quotes(puts_raw)
    puts = puts_raw.loc[idx,:]

    calls = calls[calls['volume'] > calls['volume'].quantile(.75)].set_index('contractSymbol')
    puts = puts[puts['volume'] > puts['volume'].quantile(.75)].set_index('contractSymbol')
    
    calls['lastTradeDate'] = calls['lastTradeDate'].dt.tz_localize(None)
    puts['lastTradeDate'] = puts['lastTradeDate'].dt.tz_localize(None)
    
    return calls, puts



def treeUnder(start,T,Nt,sigma=None,u=None,d=None):
    """
    Generates a binomial tree for an underlying asset price.

    Parameters:
    start (float): The starting price of the underlying asset.
    T (float): The time to expiration in years.
    Nt (int): The number of time steps in the binomial tree.
    sigma (float, optional): The volatility of the underlying asset. If not provided, it is calculated based on u and d.
    u (float, optional): The up factor of the binomial tree. If not provided, it is calculated based on sigma.
    d (float, optional): The down factor of the binomial tree. If not provided, it is calculated based on sigma.

    Returns:
    tuple: A tuple containing the binomial tree as a DataFrame and additional information as a Series.
    """
    dt = T/Nt
    Ns = Nt+1
    
    if u is None:
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        
    grid = np.empty((Ns,Nt+1))
    grid[:] = np.nan
    
    tree = pd.DataFrame(grid)
    
    for t in tree.columns:
        for s in range(0,t+1):
            tree.loc[s,t] = start * (d**s * u**(t-s))

    treeinfo = pd.Series({'u':u,'d':d,'Nt':Nt,'dt':dt}).T
            
    return tree, treeinfo




def treeAsset(funPayoff, treeUnder, treeInfo, Z=None, pstar=None, style='european'):
    """
    Calculates the option tree for a given asset.

    Parameters:
    - funPayoff (function): The payoff function for the option.
    - treeUnder (DataFrame): The tree of underlying asset prices.
    - treeInfo (object): Information about the tree structure.
    - Z (array-like, optional): Array of discount factors. Default is None.
    - pstar (array-like, optional): Array of probabilities. Default is None.
    - style (str, optional): Option style, either 'european' or 'american'. Default is 'european'.

    Returns:
    - treeV (DataFrame): The option value tree.
    - treeExer (DataFrame, optional): The exercise tree for American options. Only returned if style is 'american'.
    """
    treeV = pd.DataFrame(np.nan, index=list(range(int(treeInfo.Nt+1))), columns=list(range(int(treeInfo.Nt+1))))

    if style == 'american':
        treeExer = treeV.copy()

    for t in reversed(treeV.columns):
        if t == treeV.columns[-1]:
            for s in treeV.index:
                treeV.loc[s, t] = funPayoff(treeUnder.loc[s, t])
                if style == 'american':
                    if treeV.loc[s, t] > 0:
                        treeExer.loc[s, t] = True
                    else:
                        treeExer.loc[s, t] = False

        else:
            probvec = [pstar[t-1], 1-pstar[t-1]]

            for s in treeV.index[:-1]:
                treeV.loc[s, t] = Z[t-1] * treeV.loc[[s, s+1], t+1] @ probvec

                if style == 'american':
                    exerV = funPayoff(treeUnder.loc[s, t])
                    if exerV > treeV.loc[s, t]:
                        treeExer.loc[s, t] = True
                        treeV.loc[s, t] = exerV
                    else:
                        treeExer.loc[s, t] = False

    if style == 'american':
        return treeV, treeExer
    else:
        return treeV
    
    
def bs_delta_to_strike(under,delta,sigma,T,isCall=True,r=0):
    """
    Calculates the strike price given the underlying asset price, delta, volatility, time to expiration,
    option type (call or put), and risk-free interest rate.
    
    Parameters:
    under (float): The underlying asset price.
    delta (float): The delta value.
    sigma (float): The volatility of the underlying asset.
    T (float): The time to expiration in years.
    isCall (bool, optional): Specifies whether the option is a call option (True) or a put option (False). Default is True.
    r (float, optional): The risk-free interest rate. Default is 0.
    
    Returns:
    float: The strike price.
    """
    
    if isCall:
        phi = 1
    else:
        phi = -1
        if delta > 0:
            delta *= -1
        
    strike = under * np.exp(-phi * norm.ppf(phi*delta) * sigma * np.sqrt(T) + .5*sigma**2*T)
    
    return strike
    

    