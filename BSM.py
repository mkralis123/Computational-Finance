"""
@author: Michael Kralis
"""

import numpy as np
from scipy.stats import norm


def BSM(callput, S, K, sig, t, r, q = 0):
    
    """
    S is the current stock price
    K is the strike price
    sig is the volatility
    t is the time till experiation
    r is the risk free rate
    q is the continuous dividend yield
    """
    def vega(S, K, sig, t, r, q=0):
        d1 = (np.log(S/K)+(r-q+(sig**2)/2)*t)/(sig*np.sqrt(t))
        ans = S*np.sqrt(t)*norm.pdf(d1)*np.exp(t*(-q))
        return ans
    
    d1 = (np.log(S/K)+(r-q+(sig**2)/2)*t)/(sig*np.sqrt(t))
    d2 = d1 - sig*np.sqrt(t)


    if callput == 1:
        
        call = (S*np.exp(t*(-q))*norm.cdf(d1))-(K*np.exp(t*(-r))*norm.cdf(d2))
        deltacall = np.exp(t*(-q))*norm.cdf(d1)
        
        return (call, deltacall, vega)
    
    elif callput == -1:
        
        put = (K*np.exp(t*(-r))*norm.cdf(-d2)) - (S*np.exp(t*(-q))*norm.cdf(-d1))
        deltaput = np.exp(t*(-q))*(norm.cdf(d1)-1)
        
        return(put, deltaput, vega)
    
    else:
        return None
    
call = BSM(1,50,50,0.4,1,0.025,0.015)