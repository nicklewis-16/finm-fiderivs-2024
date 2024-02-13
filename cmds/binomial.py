import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from treasury_cmds import compound_rate


def format_bintree(df, style='{:.2f}'):
    """
    Formats a binary tree represented as a DataFrame.

    Args:
        df (pandas.DataFrame): The binary tree represented as a DataFrame.
        style (str, optional): The formatting style to apply. Defaults to '{:.2f}'.

    Returns:
        pandas.io.formats.style.Styler: The formatted binary tree.
    """
    return df.style.format(style, na_rep='').format_index('{:.2f}', axis=1)


def construct_rate_tree(dt,T):
    """
    Constructs a rate tree based on the given time step and time horizon.

    Parameters:
    dt (float): The time step.
    T (float): The time horizon.

    Returns:
    pd.DataFrame: The rate tree.
    """
    timegrid = pd.Series((np.arange(0,round(T/dt)+1)*dt).round(6),name='time',index=pd.Index(range(round(T/dt)+1),name='state'))
    tree = pd.DataFrame(dtype=float,columns=timegrid,index=timegrid.index)
    return tree



def construct_quotes(maturities, prices):
    """
    Constructs a DataFrame of quotes with maturities, prices, and continuous yield to maturity (YTM).

    Parameters:
    maturities (list): List of maturities for each quote.
    prices (list): List of prices for each quote.

    Returns:
    quotes (DataFrame): DataFrame containing maturities, prices, and continuous YTM.
    """
    quotes = pd.DataFrame({'maturity': maturities, 'price': prices})    
    quotes['continuous ytm'] = -np.log(quotes['price']/100) / quotes['maturity']
    quotes.set_index('maturity', inplace=True)
    
    return quotes

def payoff_bond(r, dt, facevalue=100):
    """
    Calculates the payoff of a bond.

    Parameters:
    - r (float): The interest rate.
    - dt (float): The time period.
    - facevalue (float, optional): The face value of the bond. Default is 100.

    Returns:
    - price (float): The calculated bond price.
    """
    price = np.exp(-r * dt) * facevalue
    return price

def payoff_swap(r, swaprate, freqswap, ispayer=True, N=100):
    """
    Calculates the payoff of a swap contract.

    Parameters:
    - r (float): The interest rate.
    - swaprate (float): The fixed rate of the swap.
    - freqswap (float): The frequency of the swap payments.
    - ispayer (bool, optional): Specifies whether the party is the payer or receiver of the swap. Default is True.
    - N (int, optional): The notional amount of the swap. Default is 100.

    Returns:
    - payoff (float): The payoff of the swap contract.
    """
    if ispayer:
        payoff = N * (r - swaprate) / freqswap
    else:
        payoff = N * (swaprate - r) / freqswap

    return payoff


def replicating_port(quotes, undertree, derivtree, dt=None, Ncash=100):
    """
    Calculates the replicating portfolio for a derivative given the quotes, undertree, and derivtree.

    Parameters:
    quotes (array-like): The quotes for the derivative.
    undertree (DataFrame): The underlying asset price tree.
    derivtree (DataFrame): The derivative price tree.
    dt (float, optional): The time step size. If not provided, it is calculated from the undertree.
    Ncash (float, optional): The amount of cash to invest. Default is 100.

    Returns:
    DataFrame: The replicating portfolio positions and their values.
    """

    if dt is None:
        dt = undertree.columns[1] - undertree.columns[0]
    
    delta = (derivtree.loc[0, dt] - derivtree.loc[1, dt]) / (undertree.loc[0, dt] - undertree.loc[1, dt]) 
    cash = (derivtree.loc[0, dt] - delta * undertree.loc[0, dt]) / Ncash
    
    out = pd.DataFrame({'positions': [cash, delta], 'value': quotes}, index=['cash', 'under'])
    out.loc['derivative', 'value'] = out['positions'] @ out['value']
    return out


def bintree_pricing_old(payoff=None, ratetree=None, undertree=None, cftree=None, pstars=None, timing=None, style='european'):
    """
    Calculates the binomial tree pricing for a financial derivative.

    Parameters:
    - payoff (function): The payoff function for the derivative. Default is None.
    - ratetree (DataFrame): The interest rate tree. Default is None.
    - undertree (DataFrame): The underlying asset price tree. Default is None.
    - cftree (DataFrame): The cash flow tree. Default is None.
    - pstars (Series): The probability tree. Default is None.
    - timing (str): The timing of the derivative. Default is None.
    - style (str): The style of the derivative. Default is 'european'.

    Returns:
    - valuetree (DataFrame): The binomial tree with calculated values.
    """

    if payoff is None:
        payoff = lambda r: payoff_bond(r, dt)

    if undertree is None:
        undertree = ratetree

    if cftree is None:
        cftree = pd.DataFrame(0, index=undertree.index, columns=undertree.columns)

    if pstars is None:
        pstars = pd.Series(.5, index=undertree.columns)

    if undertree.columns.to_series().diff().std() > 1e-5:
        display('time grid is unevenly spaced')
    dt = undertree.columns[1] - undertree.columns[0]

    valuetree = pd.DataFrame(dtype=float, index=undertree.index, columns=undertree.columns)

    for steps_back, t in enumerate(valuetree.columns[-1::-1]):
        if steps_back == 0:
            valuetree[t] = payoff(undertree[t])
            if timing == 'deferred':
                valuetree[t] *= np.exp(-ratetree[t] * dt)
        else:
            for state in valuetree[t].index[:-1]:
                valuetree.loc[state, t] = np.exp(-ratetree.loc[state, t] * dt) * (
                            pstars[t] * valuetree.iloc[state, -steps_back] + (1 - pstars[t]) *
                            valuetree.iloc[state + 1, -steps_back] + cftree.loc[state, t])

            if style == 'american':
                valuetree.loc[:, t] = np.maximum(valuetree.loc[:, t],
                                                 payoff(undertree.loc[:, t]) + np.exp(-ratetree.loc[:, t] * dt) *
                                                 cftree.loc[:, t])

    return valuetree



def bintree_pricing(payoff=None, ratetree=None, undertree=None, cftree=None, dt=None, pstars=None, timing=None, cfdelay=False, style='european', Tamerican=0):
    """
    Calculates the pricing of an option using a binomial tree model.

    Parameters:
    - payoff (function): The payoff function of the option. Default is None.
    - ratetree (DataFrame): The interest rate tree. Default is None.
    - undertree (DataFrame): The underlying asset price tree. Default is None.
    - cftree (DataFrame): The cash flow tree. Default is None.
    - dt (float): The time step size. Default is None.
    - pstars (Series): The probability of up movement in the binomial tree. Default is None.
    - timing (str): The timing of the cash flows. Default is None.
    - cfdelay (bool): Whether to delay the cash flows. Default is False.
    - style (str): The option style. Default is 'european'.
    - Tamerican (float): The time at which the option becomes American style. Default is 0.

    Returns:
    - valuetree (DataFrame): The tree of option values at each node.
    """
    
    if payoff is None:
        payoff = lambda r: 0
    
    if undertree is None:
        undertree = ratetree
        
    if cftree is None:
        cftree = pd.DataFrame(0, index=undertree.index, columns=undertree.columns)
        
    if pstars is None:
        pstars = pd.Series(.5, index=undertree.columns)

    if dt is None:
        dt = undertree.columns.to_series().diff().mean()
        dt = undertree.columns[1]-undertree.columns[0]
    
    if timing == 'deferred':
        cfdelay = True
    
    if dt<.25 and cfdelay:
        display('Warning: cfdelay setting only delays by dt.')
        
    valuetree = pd.DataFrame(dtype=float, index=undertree.index, columns=undertree.columns)

    for steps_back, t in enumerate(valuetree.columns[-1::-1]):
        if steps_back==0:                           
            valuetree[t] = payoff(undertree[t])
            if cfdelay:
                valuetree[t] *= np.exp(-ratetree[t]*dt)
        else:
            for state in valuetree[t].index[:-1]:
                val_avg = pstars[t] * valuetree.iloc[state,-steps_back] + (1-pstars[t]) * valuetree.iloc[state+1,-steps_back]
                
                if cfdelay:
                    cf = cftree.loc[state,t]
                else:                    
                    cf = cftree.iloc[state,-steps_back]
                
                valuetree.loc[state,t] = np.exp(-ratetree.loc[state,t]*dt) * (val_avg + cf)

            if style=='american':
                if t>= Tamerican:
                    valuetree.loc[:,t] = np.maximum(valuetree.loc[:,t],payoff(undertree.loc[:,t]))
        
    return valuetree


def bond_price_error(quote, pstars, ratetree, style='european'):
    """
    Calculates the error between the model price and the given quote for a bond.

    Parameters:
    quote (float): The quoted price of the bond.
    pstars (list): List of probabilities for the binomial tree.
    ratetree (DataFrame): DataFrame representing the interest rate tree.
    style (str, optional): The style of the option. Defaults to 'european'.

    Returns:
    float: The error between the model price and the quote.
    """
    FACEVALUE = 100
    dt = ratetree.columns[1] - ratetree.columns[0]    
    payoff = lambda r: payoff_bond(r,dt)
    modelprice = bintree_pricing(payoff, ratetree, pstars=pstars, style=style).loc[0,0]
    error = modelprice - quote

    return error


def estimate_pstar(quotes,ratetree,style='european'):
    """
    Estimates the pstar values for each time step in the ratetree.

    Parameters:
    - quotes (DataFrame): DataFrame containing quotes data.
    - ratetree (DataFrame): DataFrame representing the rate tree.
    - style (str): Style of the option (default is 'european').

    Returns:
    - pstars (Series): Series containing the estimated pstar values.
    """
    pstars = pd.Series(dtype=float, index= ratetree.columns[:-1], name='pstar')
    p0 = .5
    
    for steps_forward, t in enumerate(ratetree.columns[1:]):        
        ratetreeT = ratetree.copy().loc[:,:t].dropna(axis=0,how='all')
        t_prev = ratetreeT.columns[steps_forward]
        
        pstars_solved = pstars.loc[:t_prev].iloc[:-1]
        wrapper_fun = lambda p: bond_price_error(quotes['price'].iloc[steps_forward+1], pd.concat([pstars_solved, pd.Series(p,index=[t_prev])]), ratetreeT, style=style)

        pstars[t_prev] = fsolve(wrapper_fun,p0)[0]

    return pstars



def exercise_decisions(payoff, undertree, derivtree):
    """
    Determines the exercise decisions for a derivative based on the payoff function, the underlying tree, and the derivative tree.

    Parameters:
    payoff (function): The payoff function that calculates the derivative's payoff based on the underlying tree.
    undertree (numpy.ndarray): The underlying tree.
    derivtree (numpy.ndarray): The derivative tree.

    Returns:
    numpy.ndarray: A boolean array indicating the exercise decisions for the derivative.
    """
    exer = (derivtree == payoff(undertree)) & (derivtree > 0)
    return exer


def rates_to_BDTstates(ratetree):
    """
    Converts a rate tree to a BDT state tree.

    Parameters:
    ratetree (numpy.ndarray): The rate tree to be converted.

    Returns:
    numpy.ndarray: The BDT state tree.
    """
    ztree = np.log(100*ratetree)
    return ztree

def BDTstates_to_rates(ztree):
    """
    Converts the binomial tree of zero rates to the binomial tree of continuously compounded rates.

    Parameters:
    ztree (numpy.ndarray): The binomial tree of zero rates.

    Returns:
    numpy.ndarray: The binomial tree of continuously compounded rates.
    """
    ratetree = np.exp(ztree)/100
    return ratetree


def incrementBDTtree(ratetree, theta, sigma, dt=None):
    """
    Increment the BDT tree by updating the last column of the rate tree.

    Parameters:
    ratetree (DataFrame): The rate tree.
    theta (float): The theta parameter.
    sigma (float): The sigma parameter.
    dt (float, optional): The time step. If not provided, it is calculated from the rate tree.

    Returns:
    DataFrame: The updated rate tree.
    """
    if dt is None:
        dt = ratetree.columns[1] - ratetree.columns[0]

    tstep = len(ratetree.columns)-1
    
    ztree = rates_to_BDTstates(ratetree)
    ztree.iloc[:,-1] = ztree.iloc[:,-2] + theta * dt + sigma * np.sqrt(dt)
    ztree.iloc[-1,-1] = ztree.iloc[-2,-2] + theta * dt - sigma * np.sqrt(dt)
    
    newtree = BDTstates_to_rates(ztree)
    return newtree

def incremental_BDT_pricing(tree, theta, sigma_new, dt=None):
        """
        Calculates the model price using the incremental Binomial
        Distribution Tree (BDT) pricing method.

        Parameters:
        - tree (DataFrame): The original BDT tree.
        - theta (float): The mean reversion parameter.
        - sigma_new (float): The new volatility parameter.
        - dt (float, optional): The time step size. If not provided,
            it is calculated from the tree.

        Returns:
        - model_price (float): The calculated model price.
        """
        if dt==None:
                dt = tree.columns[1] - tree.columns[0]
        
        payoff = lambda r: payoff_bond(r,dt)
        newtree = incrementBDTtree(tree, theta, sigma_new)
        model_price = bintree_pricing(payoff, newtree)
        return model_price


def estimate_theta(sigmas, quotes_zeros, dt=None, T=None):
    """
    Estimates the theta values for a given set of sigmas and zero quotes.

    Parameters:
    - sigmas (float or pd.Series): The volatility values for each time step.
    - quotes_zeros (pd.Series): The zero quotes for each time step.
    - dt (float, optional): The time step size. If not provided, it is calculated from the index of quotes_zeros.
    - T (float, optional): The maturity time. If not provided, it is calculated from the index of quotes_zeros.

    Returns:
    - theta (pd.Series): The estimated theta values for each time step.
    - ratetree (pd.DataFrame): The rate tree constructed during the estimation process.
    """
    if dt is None:
        dt = quotes_zeros.index[1] - quotes_zeros.index[0]

    if T is None:
        T = quotes_zeros.index[-2]

    if quotes_zeros.mean() < 1:
        scale = 1
    else:
        scale = 100
        
    ratetree = construct_rate_tree(dt, T)
    theta = pd.Series(dtype=float, index=ratetree.columns, name='theta')
    dt = ratetree.columns[1] - ratetree.columns[0]
    
    if type(sigmas) is float:
        sigmas = pd.Series(sigmas, index=theta.index)

    for tsteps, t in enumerate(quotes_zeros.index):
        if tsteps == 0:
            ratetree.loc[0, 0] = -np.log(quotes_zeros.iloc[tsteps] / scale) / dt
        else:
            subtree = ratetree.iloc[:tsteps+1, :tsteps+1]
            wrapper = lambda theta: incremental_BDT_pricing(subtree, theta, sigmas.iloc[tsteps]).loc[0, 0] - quotes_zeros.iloc[tsteps] * 100 / scale
            
            theta.iloc[tsteps] = fsolve(wrapper, .5)[0]
            ratetree.iloc[:tsteps+1, tsteps] = incrementBDTtree(subtree, theta.iloc[tsteps], sigmas.iloc[tsteps]).iloc[:, tsteps]
            
            #print(f'Completed: {tsteps/len(quotes_zeros.index):.1%}')
            
    return theta, ratetree


def construct_bond_cftree(T, compound, cpn, cpn_freq=2, face=100):
    """
    Constructs a cashflow tree for a bond.

    Parameters:
    - T (float): Time to maturity in years.
    - compound (float): Compound interest rate per year.
    - cpn (float): Coupon rate per year.
    - cpn_freq (int, optional): Coupon frequency per year. Defaults to 2.
    - face (float, optional): Face value of the bond. Defaults to 100.

    Returns:
    - cftree (DataFrame): Cashflow tree for the bond.
    """
    step = int(compound/cpn_freq)

    cftree = construct_rate_tree(1/compound, T)
    cftree.iloc[:,:] = 0
    cftree.iloc[:, -1:0:-step] = (cpn/cpn_freq) * face
    
    # final cashflow is accounted for in payoff function
    # drop final period cashflow from cashflow tree
    cftree = cftree.iloc[:-1,:-1]
    
    return cftree



# def construct_accinttree_old(cftree, compound, cpn, cpn_freq=2, face=100, cleancall=True):
#     accinttree = cftree.copy()
#     step = int(compound/cpn_freq)
#     if cleancall is True:
#         accinttree.iloc[:,-1::-step] = face * (cpn/compound)
        
#     return accinttree


def construct_accint(timenodes, freq, cpn, cpn_freq=2, face=100):
    """
    Constructs an accrued interest schedule based on the given parameters.

    Parameters:
    - timenodes (array-like): Array of time nodes.
    - freq (int): Frequency of compounding per year.
    - cpn (float): Coupon rate.
    - cpn_freq (int, optional): Frequency of coupon payments per year. Default is 2.
    - face (float, optional): Face value of the instrument. Default is 100.

    Returns:
    - accint (pd.Series): Series containing the accrued interest schedule.
    """
    mod = freq/cpn_freq
    cpn_pmnt = face * cpn/cpn_freq

    temp = np.arange(len(timenodes)) % mod
    # shift to ensure end is considered coupon (not necessarily start)
    temp = (temp - temp[-1] - 1) % mod
    temp = cpn_pmnt * temp.astype(float)/mod

    accint = pd.Series(temp,index=timenodes)

    return accint



def idx_payoff_periods(series_periods, freq_payoffs, freq_periods=None):
    """
    Checks if the given series_periods is a multiple of freq_periods divided by freq_payoffs.

    Args:
        series_periods (int): The number of series periods.
        freq_payoffs (int): The frequency of payoffs.
        freq_periods (int, optional): The frequency of periods. Defaults to None.

    Returns:
        bool: True if series_periods is a multiple of freq_periods divided by freq_payoffs, False otherwise.
    """
    return ((series_periods * freq_periods) % (freq_periods / freq_payoffs)) == 0

def construct_swap_cftree(ratetree, swaprate, freqswap=1, T=None, freq=None, ispayer=True, N=100):
    """
    Constructs a cashflow tree for a swap contract based on a given rate tree and swap rate.

    Parameters:
    - ratetree (pd.DataFrame): A DataFrame representing the rate tree.
    - swaprate (float): The fixed rate of the swap contract.
    - freqswap (int, optional): The frequency of swap payments in a year. Default is 1.
    - T (float, optional): The maturity time of the swap contract. Default is None.
    - freq (int, optional): The frequency of rate tree columns. Default is None.
    - ispayer (bool, optional): True if the party is the payer of the swap, False if receiver. Default is True.
    - N (int, optional): The number of steps in the binomial tree. Default is 100.

    Returns:
    - cftree (pd.DataFrame): A DataFrame representing the cashflow tree.
    - refratetree (pd.DataFrame): A DataFrame representing the compounded rate tree.
    """
    cftree = pd.DataFrame(0, index=ratetree.index, columns=ratetree.columns)
    cftree[ratetree.isna()] = np.nan

    if freq is None:
        freq = round(1/cftree.columns.to_series().diff().mean())
    
    if T is None:
        T = cftree.columns[-1] + 1/freq
        
    mask_swap_dates = idx_payoff_periods(cftree.columns, freqswap, freq)
    mask_cols = cftree.columns[mask_swap_dates]
    
    payoff = lambda r: payoff_swap(r,swaprate,freqswap,ispayer=ispayer,N=100)
    
    refratetree = compound_rate(ratetree,None,freqswap)
    
    cftree[mask_cols] = payoff(refratetree[mask_cols])

    # final cashflow is accounted for in payoff function
    # will not impact bintree_pricing, but should drop them for clarity
    #cftree.iloc[:,-1] = 0
    
    return cftree, refratetree



def price_callable(quotes, fwdvols, cftree, accint, wrapper_bond, payoff_call, cleanstrike=True):
    """
    Calculates the price of a callable bond using a binomial tree model.

    Parameters:
    - quotes (array-like): Array of market quotes for the underlying bond.
    - fwdvols (array-like): Array of forward volatilities for the underlying bond.
    - cftree (DataFrame): Cash flow tree representing the bond's cash flows.
    - accint (float): Accrued interest for the bond.
    - wrapper_bond (Payoff): Payoff object representing the bond.
    - payoff_call (Payoff): Payoff object representing the call option.
    - cleanstrike (bool, optional): Flag indicating whether to use clean strike price. Defaults to True.

    Returns:
    - model_price_dirty (float): The dirty price of the callable bond.
    """
    theta, ratetree = estimate_theta(fwdvols, quotes)
    bondtree = bintree_pricing(payoff=wrapper_bond, ratetree=ratetree, cftree=cftree)
    if cleanstrike:
        cleantree = np.maximum(bondtree.subtract(accint, axis=1), 0)
        undertree = cleantree
    else:
        undertree = bondtree

    calltree = bintree_pricing(payoff=payoff_call, ratetree=ratetree, undertree=undertree, style='american')
    callablebondtree = bondtree - calltree
    model_price_dirty = callablebondtree.loc[0, 0]

    return model_price_dirty




def BDTtree(thetas, sigmas, r0=None, px_bond0=None, dt=None, T=None):
    """
    Constructs a Binomial Distribution Tree (BDT) for interest rates.

    Parameters:
    - thetas (pd.Series): Series of theta values representing the mean reversion rates.
    - sigmas (pd.Series): Series of sigma values representing the volatility rates.
    - r0 (float, optional): Initial interest rate. If not provided, it is calculated based on px_bond0 and dt.
    - px_bond0 (float, optional): Initial bond price. Required if r0 is not provided.
    - dt (float, optional): Time step size. If not provided, it is calculated based on thetas index.
    - T (float, optional): Time horizon. If not provided, it is calculated based on thetas index.

    Returns:
    - bdttree (pd.DataFrame): DataFrame representing the BDT with interest rates.

    """
    if dt is None:
        dt = thetas.index[1] - thetas.index[0]

    if T is None:
        T = thetas.index[-1]

    if r0 is None:
        r0 = -np.log(px_bond0)/dt

    ztree = construct_rate_tree(dt,T)
    ztree.iloc[0,0] = rates_to_BDTstates(r0)

    # sigmas is indexed starting at dt, so tsteps is lagged
    for tsteps, t in enumerate(sigmas.index):
        ztree.iloc[:,tsteps+1] = ztree.iloc[:,tsteps] + thetas.iloc[tsteps] * dt + sigmas.iloc[tsteps] * np.sqrt(dt)
        ztree.iloc[tsteps+1,tsteps+1] = ztree.iloc[tsteps,tsteps] + thetas.iloc[tsteps] * dt - sigmas.iloc[tsteps] * np.sqrt(dt)
            
    bdttree = BDTstates_to_rates(ztree)

    return bdttree


def align_days_interval_to_tree_periods(days, freq):
    """
    Aligns the given number of days to the nearest tree periods based on the specified frequency.

    Parameters:
    days (float): The number of days to align.
    freq (float): The frequency of the tree periods.

    Returns:
    float: The aligned number of tree periods in years.
    """
    yrs = days / 365.25
    treeyrs = round(round(yrs * freq) / freq, 6)

    return treeyrs