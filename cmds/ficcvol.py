import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm


def cap_vol_to_price(flatvol, strike, fwds, discounts, dt=.25, notional=100):
    """
    Calculates the price of a cap option based on flat volatility.

    Parameters:
    - flatvol (float): Flat volatility value.
    - strike (float): Strike price of the cap option.
    - fwds (pd.Series): Series of forward rates.
    - discounts (pd.Series): Series of discount factors.
    - dt (float, optional): Time increment. Default is 0.25.
    - notional (float, optional): Notional amount. Default is 100.

    Returns:
    - capvalue (float): Price of the cap option.
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

    Parameters:
    flatvol (float): The flat volatility value.
    strike (float): The strike price of the cap.
    fwds (pd.Series): The forward rates.
    discounts (pd.Series): The discount factors.
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




def blacks_formula(T,vol,strike,fwd,discount=1,isCall=True):
    """
    Calculates the value of an option using Black's formula.

    Parameters:
    - T (float): Time to expiration in years.
    - vol (float): Volatility of the underlying asset.
    - strike (float): Strike price of the option.
    - fwd (float): Forward price of the underlying asset.
    - discount (float, optional): Discount factor. Default is 1.
    - isCall (bool, optional): True if the option is a call option, False if it is a put option. Default is True.

    Returns:
    - val (float): Value of the option.
    """
        
    sigT = vol * np.sqrt(T)
    d1 = (1/sigT) * np.log(fwd/strike) + .5*sigT
    d2 = d1-sigT
    
    if isCall:
        val = discount * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        val = discount * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    return val





def price_caplet(T_rateset,vol,strike,fwd,discount,freq=4,notional=100):
    """
    Calculates the price of a caplet using Black's formula.

    Parameters:
    - T_rateset (float): Time to the rateset in years.
    - vol (float): Volatility of the underlying rate.
    - strike (float): Strike rate of the caplet.
    - fwd (float): Forward rate.
    - discount (float): Discount factor.
    - freq (int, optional): Frequency of compounding per year. Defaults to 4.
    - notional (float, optional): Notional amount. Defaults to 100.

    Returns:
    - price (float): Price of the caplet.
    """
    dt = 1/freq
    price = notional * dt * blacks_formula(T_rateset, vol, strike, fwd, discount)
    return price



# wrapper for better version of the function
def flat_to_forward_vol(curves, freq=None, notional=100):
    capcurves = flat_to_forward_vol_rev(
        curves['flat vols'],
        curves['swap rates'],
        curves['forwards'],
        curves['discounts'],
        freq=4,
        returnCaplets=False)

    return capcurves
    
    
    

def flat_to_forward_vol_rev(flatvols,strikes,fwds, discounts, freq=None, notional=100, returnCaplets=False):
    """
    Converts flat volatilities to forward volatilities using cap pricing.

    Args:
        flatvols (pd.Series): Series of flat volatilities.
        strikes (pd.Series): Series of strikes.
        fwds (pd.Series): Series of forward rates.
        discounts (pd.Series): Series of discount factors.
        freq (int, optional): Frequency of the time grid. Defaults to None.
        notional (float, optional): Notional amount. Defaults to 100.
        returnCaplets (bool, optional): Flag indicating whether to return caplets. Defaults to False.
    """
    #TODO: allow for timegrid to differ from cap timing
    if freq!=4:
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
    
    

# old code for pedagogical instruction in 2023--pending deletion
def flat_to_forward_vol_old(curves, freq=None, notional=100):
    
    dt = curves.index[1] - curves.index[0]
    if freq is None:
        freq = int(1/dt)
   
    capcurves = curves[['flat vols']].copy()

    for tstep, t in enumerate(capcurves.index):
    
        if tstep == 0:
            capcurves.loc[t,'caplet prices'] = np.nan
            capcurves.loc[t,'fwd vols'] = np.nan
        else:
            capcurves.loc[t,'cap prices'] = cap_vol_to_price(capcurves.loc[t,'flat vols'], curves.loc[t,'swap rates'], curves.loc[:t,'forwards'], curves.loc[:t,'discounts'], dt=dt)
            capcurves['caplet prices'].loc[t] = capcurves.loc[t,'cap prices'] - capcurves.loc[:tprev,'caplet prices'].sum()
            wrapper = lambda vol: capcurves['caplet prices'].loc[t] - notional * (1/freq) * blacks_formula(tprev, vol, curves.loc[t,'swap rates'], curves.loc[t,'forwards'], curves.loc[t,'discounts'])
            capcurves.loc[t,'fwd vols'] = fsolve(wrapper,capcurves.loc[t,'flat vols'])[0]

        tprev = t
        
    return capcurves



def shiftrates_fwdvols(dr,curves):
    """
    Calculates the shifted forward volatilities and discounts based on the given shift rate and curves.

    Parameters:
    dr (float): The shift rate to be applied to the swap rates.
    curves (DataFrame): The input curves containing swap rates.

    Returns:
    DataFrame: The shifted forward volatilities and discounts.
    """
    curves_mod = curves.copy()
    curves_mod['swap rates'] = curves['swap rates'] + dr
    
    curves_mod['discounts'] = ratecurve_to_discountcurve(curves_mod['swap rates'], n_compound=compound)
    curves_mod['forwards'] = ratecurve_to_forwardcurve(curves_mod['swap rates'], n_compound=compound)

    capcurves = flat_to_forward_vol(curves_mod)

    sigmas = capcurves['fwd vols']
    sigmas.iloc[0] = sigmas.iloc[1]
    
    return pd.concat([sigmas, curves_mod['discounts']],axis=1)