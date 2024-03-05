import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm


def cap_vol_to_price(flatvol, strike, fwds, discounts, dt=.25, notional=100):
    """
    Calculates the price of a cap option based on the given parameters.

    Parameters:
    - flatvol (float): The flat volatility used in the Black's formula.
    - strike (float): The strike price of the cap option.
    - fwds (pd.Series): A pandas Series containing the forward rates.
    - discounts (pd.Series): A pandas Series containing the discount factors.
    - dt (float, optional): The time step used in the calculation. Defaults to 0.25.
    - notional (float, optional): The notional amount of the cap option. Defaults to 100.

    Returns:
    - capvalue (float): The price of the cap option.
    """
    T = discounts.index[-1]
    flatvalues = pd.Series(dtype=float, index=discounts.index, name='flat values')
    
    tprev = discounts.index[0]
    for t in discounts.index[1:]:
        flatvalues.loc[t] = notional * dt * blacks_formula(tprev, flatvol, strike, fwds.loc[t], discounts.loc[t])
        tprev = t
        
    capvalue = flatvalues.sum()        
    return capvalue




def cap_vol_to_price_rev(flatvol, strike, fwds, discounts, dt=.25, notional=100):
    """
    Calculates the price of a cap using flat volatility.

    Args:
        flatvol (float): The flat volatility value.
        strike (float): The strike price of the cap.
        fwds (pd.Series): A pandas Series containing forward rates.
        discounts (pd.Series): A pandas Series containing discount factors.
        dt (float, optional): The time step size. Defaults to 0.25.
        notional (float, optional): The notional amount. Defaults to 100.

    Returns:
        float: The price of the cap.

    """
    T = discounts.index[-1]
    flatvalues = pd.Series(dtype=float, index=discounts.index, name='flat values')
    
    tprev = discounts.index[0]
    for t in discounts.index[1:]:
        flatvalues.loc[t] = notional * dt * blacks_formula(tprev, flatvol, strike, fwds, discounts.loc[t])
        tprev = t
        
    capvalue = flatvalues.sum()        
    return capvalue


def blacks_formula(T, vol, strike, fwd, discount=1, isCall=True):
    """
    Calculates the value of an option using Black's formula.

    Parameters:
    - T (float): Time to expiration in years.
    - vol (float): Volatility of the underlying asset.
    - strike (float): Strike price of the option.
    - fwd (float): Forward price of the underlying asset.
    - discount (float, optional): Discount factor. Default is 1.
    - isCall (bool, optional): True if the option is a call option, False if it's a put option. Default is True.

    Returns:
    - val (float): Value of the option.

    """
    sigT = vol * np.sqrt(T)
    d1 = (1/sigT) * np.log(fwd/strike) + .5*sigT
    d2 = d1 - sigT
    
    if isCall:
        val = discount * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        val = discount * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    
    return val





def price_caplet(T_rateset, vol, strike, fwd, discount, freq=4, notional=100):
    """
    Calculate the price of a caplet.

    Args:
        T_rateset (float): The time to the rateset in years.
        vol (float): The volatility of the underlying rate.
        strike (float): The strike rate.
        fwd (float): The forward rate.
        discount (float): The discount factor.
        freq (int, optional): The compounding frequency per year. Defaults to 4.
        notional (float, optional): The notional amount. Defaults to 100.

    Returns:
        float: The price of the caplet.
    """
    dt = 1 / freq
    price = notional * dt * blacks_formula(T_rateset, vol, strike, fwd, discount)
    return price


# wrapper for better version of the function
def flat_to_forward_vol(curves, freq=None, notional=100):
    """
    Converts flat volatilities to forward volatilities.

    Parameters:
    - curves (dict): A dictionary containing the following curves:
        - 'flat vols' (list): List of flat volatilities.
        - 'swap rates' (list): List of swap rates.
        - 'forwards' (list): List of forward rates.
        - 'discounts' (list): List of discount factors.
    - freq (int): Frequency of the forward rates. Default is None.
    - notional (float): Notional value. Default is 100.

    Returns:
    - capcurves (list): List of forward volatilities.

    """
    capcurves = flat_to_forward_vol_rev(
        curves['flat vols'],
        curves['swap rates'],
        curves['forwards'],
        curves['discounts'],
        freq=4,
        returnCaplets=False)

    return capcurves
    
    
    

def flat_to_forward_vol_rev(flatvols, strikes, fwds, discounts, freq=None, notional=100, returnCaplets=False):
    """
    Convert flat volatilities to forward volatilities and calculate cap prices.

    Args:
        flatvols (pd.Series): Series of flat volatilities.
        strikes (pd.Series): Series of strike values.
        fwds (pd.Series): Series of forward rates.
        discounts (pd.Series): Series of discount factors.
        freq (int, optional): Frequency of the time grid. Defaults to None.
        notional (float, optional): Notional value. Defaults to 100.
        returnCaplets (bool, optional): Flag indicating whether to return caplets. Defaults to False.

    Returns:
        pd.DataFrame or tuple: If returnCaplets is False, returns a DataFrame with columns 'flat vols', 'fwd vols', and 'cap prices'.
        If returnCaplets is True, returns a tuple containing the DataFrame and a DataFrame of caplet prices.
    """
    # TODO: allow for timegrid to differ from cap timing
    if freq != 4:
        display('Warning: freq parameter controls timegrid and cap timing.')
        
    dt = 1 / freq
    
    out = pd.DataFrame(dtype=float, index=flatvols.index, columns=['fwd vols', 'cap prices'])
    caplets = pd.DataFrame(dtype=float, index=flatvols.index, columns=strikes.values)

    first_cap = flatvols.index.get_loc(2 * dt)

    for step, t in enumerate(flatvols.index):
        if step < first_cap:
            out.loc[t, 'cap prices'] = np.nan
            out.loc[t, 'fwd vols'] = np.nan
            tprev = t
        else:
            out.loc[t, 'cap prices'] = cap_vol_to_price(flatvols.loc[t], strikes.loc[t], fwds.loc[:t], discounts.loc[:t], dt=dt, notional=notional)
            if step == first_cap:
                out.loc[t, 'fwd vols'] = flatvols.loc[t]
                caplets.loc[t, strikes.loc[t]] = out.loc[t, 'cap prices']
                tprev = t
            else:
                strikeT = strikes.loc[t]

                for j in flatvols.index[first_cap:step]:
                    caplets.loc[j, strikeT] = price_caplet(j - dt, out.loc[j, 'fwd vols'], strikeT, fwds.loc[j], discounts.loc[j], freq=freq, notional=notional)

                caplets.loc[t, strikeT] = out.loc[t, 'cap prices'] - caplets.loc[:tprev, strikeT].sum()

                wrapper = lambda vol: caplets.loc[t, strikeT] - price_caplet(tprev, vol, strikeT, fwds.loc[t], discounts.loc[t], freq=freq, notional=notional)

                out.loc[t, 'fwd vols'] = fsolve(wrapper, out.loc[tprev, 'fwd vols'])[0]            
                tprev = t            

    out.insert(0, 'flat vols', flatvols)
    
    if returnCaplets:
        return out, caplets
    else:
        return out
    

def shiftrates_fwdvols(dr, curves):
    """
    Shifts the swap rates in the given curves by the specified amount and calculates the modified forward volatilities.

    Parameters:
    dr (float): The amount by which to shift the swap rates.
    curves (DataFrame): The input curves containing swap rates, discounts, and forwards.

    Returns:
    DataFrame: A DataFrame containing the modified forward volatilities and the updated discounts.
    """

    curves_mod = curves.copy()
    curves_mod['swap rates'] = curves['swap rates'] + dr
    
    curves_mod['discounts'] = ratecurve_to_discountcurve(curves_mod['swap rates'], n_compound=compound)
    curves_mod['forwards'] = ratecurve_to_forwardcurve(curves_mod['swap rates'], n_compound=compound)

    capcurves = flat_to_forward_vol(curves_mod)

    sigmas = capcurves['fwd vols']
    sigmas.iloc[0] = sigmas.iloc[1]
    
    return pd.concat([sigmas, curves_mod['discounts']], axis=1)
