"""
@author: Michael Kralis
"""

def newton(f, df, x = 2, lb = 0, ub = 100, maxiter = 100, err = 0.000001): 
    
    """ 
    f is the function that gets passed through 
    df is the derivative function that gets passed through 
    x is the initial guess 
    lb is the lower bound that the algorithm cannot go below
    ub is the upper bound that the algorithm cannot exceed
    maxiter is the maximum number of iterations the algorithm will run for 
    err is the the amount of error tolerance
    """ 
    
    n = 0
    e = abs(f(x)-0)
    xvals = []
    fdiff = []

    
    if lb < x < ub:
        while n <= maxiter and e>err: 
            x1 = x - (f(x)/df(x)) 
            x = x1
            xvals.append(x)
            fdiff.append(abs(f(x)-0))
            
            n += 1
            e = abs(f(x)-0)
        
        return x, xvals, fdiff, len(xvals)
    else:
        return None

def bisect(f, lb = -0.0000001, ub = 100, maxiter = 1000, err = 0.000001):
    
    """
    f is the function that gets passed through
    lb is the lower bound estimate
    ub is the upper bound estimate
    maxiter is the maximum number of iterations the algorithm will run for
    err is the the amount of error tolerance
    """
    
    m = (ub+lb)/2
    n = 0
    e = abs(f(m)-0)
    
    while n <= maxiter and e>err:
    
        m = (ub+lb)/2
        
        if f(lb) == 0:
            return lb
        elif f(ub) == 0:
            return ub
        elif f(m) == 0:
            return m
        elif f(m) > 0:
            ub = m
            n+=1
            e = abs(f(m)-0)
        elif f(m) < 0:
            lb = m
            n+=1
            e = abs(f(m)-0)
        
    return m, n