# these functions were used in the 2024 final exam for FINM 37400

# the functions in treasury_cmds.py mostly presume dataframes with cashflow schedules

# these functions are more useful for clean, textbook-like problems


###########

import numpy as np
import pandas as pd


def bond_pricer_formula(ttm,ytm,cpn=None,freq=2,face=100):
    
    if cpn is None:
        cpn = ytm
    
    y = ytm/freq
    
    disc_factor = 1/((1+y)**(freq*ttm))

    cf_cpn = face * (cpn/freq)
    pv_cpns = cf_cpn * (1-disc_factor)/y

    pv_tv = face * disc_factor
    
    pv = pv_tv + pv_cpns
    
    return pv
    
    
    

def bond_pricer_dcf(ttm,ytm,cpn=None,freq=2,face=100):
    
    if cpn is None:
        cpn = ytm
        
    pv = 0
    
    c = (cpn/freq)*face
    disc = 1/(1+ytm/freq)
    
    for t in range(ttm*freq):
        pv += c * disc**(t+1)
    
    pv += face * disc**(ttm*freq)

    return pv
    
    
    

def duration_closed_formula(tau, ytm, cpnrate=None, freq=2):

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
    
    

def bootstrap_discounts_clean(df, compounding=2,key=None):
    """
    Bootstraps spot discount curve--assuming clean, EVENLY spaced term structure without missing values.
    
    Input is a pandas DataFrame of YTMs, with adjustable compounding frequency.
    
    Parameters:
    df (pd.DataFrame): DataFrame with index as maturities and a 'rates' column for YTMs.
    compounding (int): Number of times interest is compounded per year.
    key (string): name of column of df which has ytms. If none, assumes only one column.

    Returns:
    pd.DataFrame: DataFrame with spot rates.
    """
    spot_rates = pd.Series(index=df.index, dtype=float)

    if key is None:
        key = df.columns[0]
    
    for maturity, row in df.iterrows():
        ytm = row[[key]]
        if maturity == df.index[0]:
            # For the first bond, the spot rate is the same as the YTM
            spot_rates[maturity] = ytm
            continue
        
        # For subsequent bonds, calculate the spot rate
        num_cash_flows = int(maturity * compounding)
        cash_flows = [100 * ytm / compounding] * (num_cash_flows - 1) + [100 * (1 + ytm / compounding)]
        discount_factors = [(1 + spot_rates[m]/compounding)**(-m*compounding) for m in df.index if m < maturity]
        price = sum(cf * df for cf, df in zip(cash_flows[:-1], discount_factors))
        last_cash_flow = cash_flows[-1]
        
        # Solving for the spot rate
        spot_rate = ((last_cash_flow / (100 - price))**(1/(maturity*compounding)) - 1) * compounding
        spot_rates[maturity] = spot_rate

    return pd.DataFrame({'spot rates': spot_rates})