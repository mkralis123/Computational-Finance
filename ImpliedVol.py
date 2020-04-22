"""
@author: Franster
"""
from BSM import BSM
from RootFinder import newton
from RootFinder import bisect

callput = -1
S = 100
K = 100
price = 30 
t = 1
r = 0.025
q = 0.05
method = 'bisect'
reportCalls = True

def impliedvol(callput, S, K, r, t, price, q, method, reportCalls = False):
    
    if callput == 1 or callput == -1:
        f = lambda x: BSM(callput, S, K, x, t, r, q)[0] - price
        
        if method == 'newton':
            
            df = lambda x: BSM(callput, S, K, x, t, r, q)[2](S, K, x, t, r, q)
            
        
            if reportCalls == False:
                values = newton(f,df)
                return values[0]
            elif reportCalls == True:
                values = newton(f,df)
                return values[0], values[3]
            else:
                return None
        
        elif method == 'bisect':
            
            if reportCalls == False:
                return bisect(f)[0]
            elif reportCalls == True:
                return bisect(f)
            else:
                return None
    else:
        return None

print(impliedvol(callput, S, K, r, t, price, q, method, reportCalls))