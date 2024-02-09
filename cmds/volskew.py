import pandas as pd
import numpy as np

from scipy.optimize import fsolve
from scipy.optimize import minimize

from pandas.tseries.offsets import Day, BDay
from datetime import date

import numpy.polynomial.polynomial as poly


### SABR

def alpha_polynomial(beta, nu, rho, T, volATM, f):
    
    coefs = np.ones(4)
    coefs[0] = (1-beta)**2 * T / (24*f**(2-2*beta))
    coefs[1] = rho * beta * nu * T / (4*f**(1-beta))
    coefs[2] = 1 + (2-3*rho**2) * nu**2 * T / 24
    coefs[3] = -volATM * f**(1-beta)
    
    return np.poly1d(coefs)

def solve_alpha(beta, nu, rho, T, volATM, f):
    coefs = np.ones(4)
    coefs[3] = (1-beta)**2 * T / (24*f**(2-2*beta))
    coefs[2] = rho * beta * nu * T / (4*f**(1-beta))
    coefs[1] = 1 + (2-3*rho**2) * nu**2 * T / 24
    coefs[0] = -volATM * f**(1-beta)

    roots = poly.polyroots(coefs)
    alpha = np.real(roots[np.abs(np.imag(roots))<1e-6][0])
    return alpha

def zfun(beta, nu, alpha, f, K):
    return (nu/alpha) * (f*K)**((1-beta)/2) * np.log(f/K)

def chi(z, rho):
    return np.log((np.sqrt(1-2*rho*z + z**2) + z - rho) / (1-rho))




def sabr_slim(beta,nu,rho,f,K,T,volATM):

    alpha = solve_alpha(beta, nu, rho, T, volATM, f)
    
    squareNUM = (((1-beta)**2)/24) * (alpha**2)/((f*K)**(1-beta)) + (1/4) * (rho*beta*nu*alpha)/((f*K)**((1-beta)/2))+((2-3*rho**2)/24)*nu**2
    NUM = alpha * (1 + squareNUM * T)
    squareDEN = 1 + (((1-beta)**2)/24) * ((np.log(f/K))**2) + (((1-beta)**4)/1920) * ((np.log(f/K))**4)
    DEN = (f*K)**((1-beta)/2) * squareDEN
    z = zfun(beta,nu,alpha,f,K)
    sigmaB = (NUM/DEN) * (z/chi(z,rho))
        
    if (type(K) is np.float64) | (type(K) is float):
        if (f==K):
            sigmaB = sabrATM(beta,nu,rho,alpha,f,K,T)
    else:
        mask = f==K
        sigmaB[mask] = sabrATM(beta,nu,rho,alpha,f,K[mask],T)
        
    return sigmaB




def sabr(beta,nu,rho,alpha,f,K,T):
   
    squareNUM = (((1-beta)**2)/24) * (alpha**2)/((f*K)**(1-beta)) + (1/4) * (rho*beta*nu*alpha)/((f*K)**((1-beta)/2))+((2-3*rho**2)/24)*nu**2
    NUM = alpha * (1 + squareNUM * T)
    squareDEN = 1 + (((1-beta)**2)/24) * (np.log(f/K)**2) + (((1-beta)**4)/1920) * (np.log(f/K)**4)
    DEN = (f*K)**((1-beta)/2) * squareDEN
    z = zfun(beta,nu,alpha,f,K)        
    sigmaB = (NUM/DEN) * (z/chi(z,rho))
    
    if (type(K) is np.float64) | (type(K) is float):
        if f==K:
            sigmaB = sabrATM(beta,nu,rho,alpha,f,K,T)
    else:
        mask = f==K
        sigmaB[mask] = sabrATM(beta,nu,rho,alpha,f,K[mask],T)
        
    return sigmaB



def sabrATM(beta,nu,rho,alpha,f,K,T):
    brack = (((1-beta)**2)/24) * ((alpha**2)/(f**(2-2*beta))) + (rho * beta * nu * alpha)/(4*f**(1-beta)) + ((2-3*rho**2)/24) * nu**2
    
    sigma = alpha * (1+brack*T) / f**(1-beta)

    return sigma




    








### Auxiliary Functions for Handling Nasdaq Skew Data

def load_vol_surface(LOADFILE,SHEET,ISCALL=False):

    info = pd.read_excel(LOADFILE,sheet_name='descriptions').set_index('specs')
    labels = info.columns

    if type(SHEET) == int or type(SHEET) == float:
        lab = labels[SHEET]
    else:
        lab = SHEET
        
    raw = pd.read_excel(LOADFILE,sheet_name=lab).set_index('Date')

    ts = raw.loc[:,['Future Price','Expiration Future','Expiration Option']]
    surf = raw.drop(columns=ts.columns)

    indPuts = surf.columns.str.contains('P')
    indCalls = surf.columns.str.contains('C')

    calls = surf[surf.columns[indCalls]]
    puts = surf[surf.columns[indPuts]]

    if ISCALL:
        opts = calls
    else:
        opts = puts
        
    return ts, opts



def get_notable_dates(opts, ts, maxdiff=False):

    if maxdiff==True:
        dtgrid = pd.DataFrame([opts.diff().abs().idxmax()[0], ts[['Future Price']].diff().abs().idxmax()[0]],columns=['notable date'],index=['max curve shift','max underlying shift'])
    else:
        dtgrid = pd.DataFrame([opts.diff().abs().idxmax()[0], ts[['Future Price']].pct_change().abs().idxmax()[0]],columns=['notable date'],index=['max curve shift','max underlying shift'])
    for row in dtgrid.index:
        dtgrid.loc[row,'day before'] = opts.loc[:dtgrid.loc[row, 'notable date'],:].index[-2]
    dtgrid = dtgrid.iloc[:,::-1].T
    
    return dtgrid
    
    
    
    
    
    
def get_strikes_from_vol_moneyness(ISCALL,opts,ts):

    phi = ISCALL * 2 - 1

    deltas = pd.DataFrame(np.array([float(col[1:3])/100 for col in opts.columns]) * phi, index=opts.columns,columns = ['delta'])

    strikes = pd.DataFrame(index=opts.index, columns=opts.columns, dtype=float)
    for t in opts.index:
        T = ts.loc[t,'Expiration Option']
        for col in deltas.index:
            strikes.loc[t,col] = bs_delta_to_strike(under = ts.loc[t,'Future Price'], delta=deltas.loc[col,'delta'], sigma=opts.loc[t,col], T=T, isCall=ISCALL)
            
            
    return strikes






def graph_vol_surface_as_strikes(dtgrid,opts,strikes,ts,label):

    fig, ax = plt.subplots(2,1,figsize=(10,10))

    for j, dt in enumerate(dtgrid.columns):

        colorgrid = ['b','r','g']

        for i, tstep in enumerate(dtgrid[dt]):
            tstep = tstep.strftime('%Y-%m-%d')
            plotdata = pd.concat([opts.loc[tstep,:],strikes.loc[tstep,:]],axis=1)
            plotdata.columns = [tstep,'strike']
            plotdata.set_index('strike',inplace=True)
            plotdata.plot(ax=ax[j],color=colorgrid[i]);

            ax[j].axvline(x=ts.loc[tstep,'Future Price'],color=colorgrid[i],linestyle='--')

            if j==0:        
                ax[j].set_title(f'Curve shock: {label}')
            elif j==1:
                ax[j].set_title(f'Underlying shock: {label}')

            if label.split(' ')[-2] == 'ED':
                ax[j].set_xlim(xmin=0,xmax=.08)
            
            plt.tight_layout()

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
def sabr_volpaths(LOADFILE, idSHEET, ISCALL, BETA, TARG_T, doSLIM=False):
    # helper function for displaying SABR volpaths
    # used in C5
    # based on data pulled from NASDAQ DataLink
    # relies on particular excel exports from build_data file associated with Module C
    
    ticksRates = ['ED']

    info = pd.read_excel(LOADFILE,sheet_name='descriptions').set_index('specs')

    labels = info.columns
    sheet = labels[idSHEET]

    tick = info.loc['futures ticker',sheet]

    ts, ivol_mkt = load_vol_surface(LOADFILE,sheet,ISCALL)

    if tick in ticksRates:
        ts['Future Price'] = 100-ts['Future Price']
        ISCALL = 1-ISCALL

    strikes = get_strikes_from_vol_moneyness(ISCALL,ivol_mkt,ts)

    bdays = BDay()

    t = pd.to_datetime(info.loc['option expiration',sheet]) - TARG_T * datetime.timedelta(365)
    t += 1 * bdays
    t = t.strftime('%Y-%m-%d')

    if 'P50dVol' in ivol_mkt.columns:
        colATM = 'P50dVol'
    else:
        colATM = 'C50dVol'

    volATM = ivol_mkt.loc[t,colATM]
    F = ts.loc[t,'Future Price']
    strike_grid = strikes.loc[t]
    T = ts.loc[t,'Expiration Option']

    ivol_obs = ivol_mkt.loc[t]

    
    
    
    ### OPTIMIZATION
    
    def obj_fun(xargs):
        nu = xargs[0]
        rho = xargs[1]
        alpha = xargs[2]
        ivol_mod = np.zeros(len(strike_grid))

        for i,K in enumerate(strike_grid):
             ivol_mod[i] = sabr(BETA,nu,rho,alpha,F,K,T)

        error = ((ivol_mod - ivol_obs)**2).sum()

        return error


    def obj_fun_slim(xargs):
        nu = xargs[0]
        rho = xargs[1]
        ivol_mod = np.zeros(len(strike_grid))

        for i,K in enumerate(strike_grid):
             ivol_mod[i] = sabr_slim(BETA,nu,rho,F,K,T,volATM)

        error = ((ivol_mod - ivol_obs)**2).sum()

        return error
    
    
    
    
    
    if not doSLIM:
        x0 = np.array([.6,0,.1])
        fun = obj_fun
    else:
        fun = obj_fun_slim
        x0 = np.array([.6,0,.1])


    optim = minimize(fun,x0)
    xstar = optim.x
    nustar = xstar[0]
    rhostar = xstar[1]


    if doSLIM:
        alphastar = solve_alpha(BETA,nustar,rhostar,T,volATM,F)
        ivolSABR = sabr_slim(BETA,nustar,rhostar,F,strike_grid,T,volATM)
    else:
        alphastar = xstar[2]
        ivolSABR = sabr(BETA,nustar,rhostar,alphastar,F,strike_grid,T)

        
        
        
        
        
        
        
    ### SOLVE SABR on GRID
    Fgrid = np.arange(F*(1-volATM),F*(1+volATM),F*volATM/2)

    volPath = pd.DataFrame(columns=ivolSABR.index,index=Fgrid)

    if doSLIM:
        for f in Fgrid:
            volPath.loc[f,:] = sabr_slim(BETA,nustar,rhostar,f,strike_grid,T,volATM)
    else:
        for f in Fgrid:
            volPath.loc[f,:] = sabr(BETA,nustar,rhostar,alphastar,f,strike_grid,T)

    strikesPath = pd.DataFrame(np.repeat(strike_grid.values.reshape(-1,1),len(Fgrid),axis=1).T, index=Fgrid, columns=ivolSABR.index)

    backbone = pd.DataFrame(index=Fgrid,dtype=float,columns=['vol path'])
    backbone.index.name = 'strike'
    for f in Fgrid:
        backbone.loc[f] = sabrATM(BETA,nustar,rhostar,alphastar,f,f,T)

    backbone['vol path approx'] = alphastar/(Fgrid**(1-BETA))

    
    
    
    
    
    ### SAVE OUTPUTS
    fig, ax = plt.subplots();

    for row in volPath.index:
        plotdata = pd.concat([strikesPath.loc[row],volPath.loc[row]],axis=1)
        plotdata.columns = ['strike','vol']
        plotdata.set_index('strike',inplace=True)
        plotdata.plot(ax=ax);

    backbone.plot(ax=ax,color=['black','gray'],linewidth=2.5,linestyle='--');

    plt.legend([f'{c:.2f}' for c in volPath.index] + ['vol path','vol path approx']);
    plt.ylabel('implied volatility');
    plt.title(f'Volatility Skews and Volatility Path: {tick}, (beta={BETA})');
    
    
    summary = pd.DataFrame([tick,t,f'{T:.2f} years'],index=['ticker','date','expiration'],columns=['reference data'])
    
    error = optim.fun
    paramtab = pd.DataFrame([BETA,alphastar,nustar,rhostar,error],index=['beta ($\\beta$)','alpha ($\\alpha$)','nu ($\\nu$)','rho ($\\rho$)','fit error'],columns=['SABR Parameters'])
    
    output = dict()
    output['figure'] = fig;
    output['summary'] = summary
    output['params'] = paramtab
    
    return output        