import numpy as np

def fdEuropean(callput, S0, K, r, T, sigma, q, M, N, Smax):
    
    """
    callput takes on a value 1 for call or -1 for put
    S0 is the current stock price
    K is the strike price
    r is the risk free interest rate
    T is the time till expiration
    sigma is the annualized volatility
    q is the annualized dividend yield
    M is the number of spacial (stock) steps
    N is the number of time steps
    Smax is the upper bound stock price
    """
    
    #Set up values to use for the finite difference method
    M = int(M)
    N = int(N)
    dt = T/N
    ds = Smax/M
    t_index = np.arange(N)
    s_index = np.arange(M)
    grid = np.zeros((M+1,N+1))
    bounds = np.linspace(0, Smax, M+1)
    
    #Establish the grid for a call
    if callput == 1:
        grid[:,-1] = np.maximum(bounds-K,0)
        grid[-1,:-1] = np.maximum(Smax-K,0)*np.exp(-r*dt*(N-t_index))
        
    #Establish the grid for a put    
    if callput == -1:
        grid[:,-1] = np.maximum(K-bounds,0)
        grid[0,:-1] = K * np.exp(-r*dt*(N-t_index))
    
    #Set up coefficients
    a = 0.5*dt*((sigma**2)*(s_index**2) - (r-q)*s_index)    
    b = 1 - dt*((sigma**2)*(s_index**2) + (r))
    c = 0.5*dt*((sigma**2)*(s_index**2) + (r-q)*s_index)
    
    #Iterate backwards through time
    for j in reversed(t_index):
        
        #Iterate along different stock prices at each given time
        for i in range(M)[1:]:
            grid[i,j] = a[i]*grid[i-1,j+1] + b[i]*grid[i,j+1] + c[i]*grid[i+1,j+1]
    
    #interpolate the value for S0
    value = np.interp(S0,bounds,grid[:,0])
    
    
    return value
    
S0 = 50
K = 50
Smax = 100
r = 0.025
T = 1
sigma = 0.4
q = 0
callput = 1
M = 85
N = 1000

optionPrice = fdEuropean(callput,S0,K,r, T, sigma, q, M, N, Smax)

#The code below was used to generate plots for the report
"""
import matplotlib.pyplot as plt

Ms = [10,20,30,40,50,60,70,80]
Prices = np.zeros(len(Ms))

for i in range(len(Ms)):
    price = fdEuropean(callput, S0, K, r, T, sigma, q, Ms[i], N, Smax)
    Prices[i] = price

plt.figure(figsize = (10,5))
plt.title("Number of Spacial steps by Option Price S0=50, K=50, vol = 40%")
plt.plot(Ms, Prices)
plt.xlabel("Spacial Steps")
plt.ylabel("Option Price") 
plt.legend() 
plt.show() 
"""