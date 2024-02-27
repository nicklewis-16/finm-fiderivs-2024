import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve





def ratecurve_to_discountcurve(ratecurve, n_compound=None):
    """
    Converts a rate curve to a discount curve.

    Parameters:
    ratecurve (pd.DataFrame or pd.Series): The rate curve to be converted.
    n_compound (int, optional): The number of compounding periods per year. If not provided, continuous compounding is assumed.

    Returns:
    discountcurve (pd.Series): The resulting discount curve.
    """

    if isinstance(ratecurve, pd.DataFrame):
        ratecurve = ratecurve.iloc[:, 0]
        
    if n_compound is None:
        discountcurve = np.exp(-ratecurve * ratecurve.index)
    else:
        discountcurve = 1 / (1 + (ratecurve / n_compound))**(n_compound * ratecurve.index)

    return discountcurve





def ratecurve_to_forwardcurve(ratecurve, n_compound=None, dt=None):
    """
    Converts a rate curve to a forward curve.

    Args:
        ratecurve (pd.DataFrame or pd.Series): The rate curve data.
        n_compound (int, optional): The number of compounding periods per year. Defaults to None.
        dt (float, optional): The time interval between rate curve points. Defaults to None.

    Returns:
        pd.Series: The forward curve data.
    """
    if isinstance(ratecurve, pd.DataFrame):
        ratecurve = ratecurve.iloc[:, 0]

    if dt is None:
        dt = ratecurve.index[1] - ratecurve.index[0]

    discountcurve = ratecurve_to_discountcurve(ratecurve, n_compound=n_compound)

    F = discountcurve / discountcurve.shift()

    if n_compound is None:
        display('TODO')
    else:
        forwardcurve = n_compound * (1 / (F**(n_compound * dt)) - 1)

    return forwardcurve




def discount_to_intrate(discount, maturity, n_compound=None):
    """
    Calculates the interest rate corresponding to a given discount factor.
    
    Parameters:
        discount (float): The discount factor.
        maturity (float): The time to maturity in years.
        n_compound (int, optional): The number of compounding periods per year. Defaults to None.
    
    Returns:
        float: The corresponding interest rate.
    """
    if n_compound is None:
        intrate = - np.log(discount) / maturity
    
    else:
        intrate = n_compound * (1/discount**(1/(n_compound * maturity)) - 1)    
        
    return intrate



def interp_curves(data, dt=None, date=None, interp_method='linear', order=None, extrapolate=True):
    """
    Interpolates rate curves based on the given data.

    Args:
        data (DataFrame): The input data containing rate curves.
        dt (float, optional): The time step between data points. If not provided, it is calculated from the data.
        date (str, optional): The specific date for which to interpolate the rate curves. If not provided, all dates are considered.
        interp_method (str, optional): The interpolation method to use. Default is 'linear'.
        order (int, optional): The order of the interpolation method. Default is None.
        extrapolate (bool, optional): Whether to extrapolate the rate curves beyond the given data. Default is True.

    Returns:
        DataFrame: The interpolated rate curves.

    """
    if dt is None:
        dt = data.columns[1] - data.columns[0]
    
    freq = 1/dt
    
    if date is None:
        temp = data
    else:
        temp = data.loc[date,:]

    newgrid = pd.DataFrame(dtype=float, index=np.arange(dt,temp.index[-1]+dt,dt),columns=['quotes'])
    # sofr curve last index often 10.02 command above extends to 10+. If barely overruns, toss last value
    overrun = (temp.index[-1] % dt)/dt
    if overrun>0 and overrun < .1:
        newgrid = newgrid.iloc[:-1,:]
        
    #newgrid.index = (freq*newgrid.index.values).round(0)/freq

    curves = temp.to_frame().rename(columns={temp.name:'quotes'})
    curves = pd.concat([curves,newgrid],axis=0)
    curves['interp'] = curves['quotes']

    if extrapolate:
        curves['interp'].interpolate(method=interp_method, order=order, limit_direction='both', fill_value = 'extrapolate',inplace=True)
    else:
        curves['interp'].interpolate(method=interp_method, order=order,inplace=True)
    
    curves = curves.loc[newgrid.index,:]
    curves = curves[~curves.index.duplicated()].sort_index()
    
    return curves




def plot_interp_curves(curves, plot_contin=True):
    """
    Plot interpolated curves.

    Parameters:
    curves (DataFrame): DataFrame containing the curves data.
    plot_contin (bool, optional): Flag to plot continuous curves. Defaults to True.
    """
    fig, ax = plt.subplots()
    curves['quotes'].plot.line(ax=ax, linestyle='None', marker='*')
    curves.iloc[:, 1:].plot.line(ax=ax, linestyle='--', marker='')

    plt.legend()
    plt.show()
    
    
def price_bond(ytm, T, cpn, cpnfreq=2, face=100, accr_frac=None):
    """
    Calculate the price of a bond.

    Parameters:
    - ytm (float): Yield to maturity of the bond.
    - T (float): Time to maturity of the bond in years.
    - cpn (float): Coupon payment of the bond.
    - cpnfreq (int, optional): Number of coupon payments per year. Default is 2.
    - face (float, optional): Face value of the bond. Default is 100.
    - accr_frac (float, optional): Accrual fraction of the bond. Default is None.

    Returns:
    - price (float): Price of the bond.
    """
    ytm_n = ytm/cpnfreq
    cpn_n = cpn/cpnfreq
    
    if accr_frac is None:
        accr_frac = (T-round(T))*cpnfreq
    
    N = T * cpnfreq
    price = face * ((cpn_n / ytm_n) * (1-(1+ytm_n)**(-N)) + (1+ytm_n)**(-N)) * (1+ytm_n)**(accr_frac)
    return price




def duration_closed_formula(tau, ytm, cpnrate=None, freq=2):
    """
    Calculates the duration of a bond using the closed-formula approach.

    Parameters:
    tau (float): Time to maturity of the bond in years.
    ytm (float): Yield to maturity of the bond.
    cpnrate (float, optional): Coupon rate of the bond. If not provided, it is assumed to be equal to the yield to maturity.
    freq (int, optional): Number of coupon payments per year. Default is 2.

    Returns:
    float: The duration of the bond.
    """
    if cpnrate is None:
        cpnrate = ytm
        
    y = ytm/freq
    c = cpnrate/freq
    T = tau * freq
        
    if cpnrate==ytm:
        duration = (1+y)/y  * (1 - 1/(1+y)**T)
        
    else:
        duration = (1+y)/y - (1+y+T*(c-y)) / (c*((1+y)**T-1)+y)

    duration /= freq
    
    return duration






def ytm(price, T, cpn, cpnfreq=2, face=100, accr_frac=None):
    """
    Calculate the Yield to Maturity (YTM) of a bond.

    Parameters:
    - price (float): The current price of the bond.
    - T (float): The time to maturity of the bond in years.
    - cpn (float): The coupon rate of the bond.
    - cpnfreq (int, optional): The number of coupon payments per year. Default is 2.
    - face (float, optional): The face value of the bond. Default is 100.
    - accr_frac (float, optional): The fraction of the accrual period. Default is None.

    Returns:
    - ytm (float): The Yield to Maturity of the bond.
    """
    pv_wrapper = lambda y: price - price_bond(y, T, cpn, cpnfreq=cpnfreq, face=face, accr_frac=accr_frac)
    ytm = fsolve(pv_wrapper,.01)
    return ytm





def calc_swaprate(discounts,T,freqswap):
    """
    Calculates the swap rate given a series of discount factors, a maturity date, and the frequency of the swap.

    Parameters:
    discounts (pandas.Series): Series of discount factors.
    T (datetime): Maturity date of the swap.
    freqswap (int): Frequency of the swap.

    Returns:
    float: The calculated swap rate.
    """
    freqdisc = round(1/discounts.index.to_series().diff().mean())
    step = round(freqdisc / freqswap)
    
    periods_swap = discounts.index.get_loc(T)
    # get exclusive of left and inclusive of right by shifting both by 1
    periods_swap += 1

    swaprate = freqswap * (1 - discounts.loc[T])/discounts.iloc[step-1:periods_swap:step].sum()
    return swaprate





def calc_fwdswaprate(discounts, Tfwd, Tswap, freqswap):
    """
    Calculates the forward swap rate given a series of discount factors, forward start date, swap maturity date, and swap frequency.

    Parameters:
    - discounts (pd.Series): Series of discount factors
    - Tfwd (datetime): Forward start date
    - Tswap (datetime): Swap maturity date
    - freqswap (int): Swap frequency

    Returns:
    - fwdswaprate (float): Forward swap rate
    """
    freqdisc = round(1/discounts.index.to_series().diff().mean())
    step = round(freqdisc / freqswap)
    
    periods_fwd = discounts.index.get_loc(Tfwd)
    periods_swap = discounts.index.get_loc(Tswap)
    # get exclusive of left and inclusive of right by shifting both by 1
    periods_fwd += step
    periods_swap += 1
    
    fwdswaprate = freqswap * (discounts.loc[Tfwd] - discounts.loc[Tswap]) / discounts.iloc[periods_fwd:periods_swap:step].sum()
    return fwdswaprate





def extract_fedpath(curves, feddates, spotfedrate):
    """
    Extracts the expected fed rate path based on given curves, fed meeting dates, and spot fed rate.

    Args:
        curves (DataFrame): DataFrame containing rate curves data.
        feddates (DataFrame): DataFrame containing fed meeting dates.
        spotfedrate (float): Spot fed rate.

    Returns:
        DataFrame: DataFrame with the expected fed rate path.

    """
    r0 = spotfedrate
    
    tag = [dt.strftime('%Y-%m') for dt in curves['last_tradeable_dt']]
    curves['date'] = tag
    curves.reset_index(inplace=True)
    curves.set_index('date', inplace=True)

    tag = [dt.strftime('%Y-%m') for dt in feddates['meeting dates']]
    feddates['date'] = tag
    feddates.set_index('date', inplace=True)

    curves = curves.join(feddates)
    curves['meeting day'] = [dt.day for dt in curves['meeting dates']]
    curves['contract days'] = [dt.day for dt in curves['last_tradeable_dt']]

    curves['futures rate'] = (100 - curves['px_last']) / 100
    curves.drop(columns=['px_last'], inplace=True)
    curves['expected fed rate'] = np.nan

    for step, month in enumerate(curves.index[:-1]):
        if step == 0:
            Eprev = r0
        else:
            Eprev = curves['expected fed rate'].iloc[step - 1]

        if np.isnan(curves['meeting day'].iloc[step]):
            curves['expected fed rate'].iloc[step] = Eprev
        else:
            if np.isnan(curves['meeting day'].iloc[step + 1]):
                curves['expected fed rate'].iloc[step] = curves['futures rate'].iloc[step + 1]
            else:
                n = curves['contract days'].iloc[step]
                m = curves['meeting day'].iloc[step]
                curves['expected fed rate'].iloc[step] = (n * curves['futures rate'].iloc[step] - m * Eprev) / (n - m)

    return curves
