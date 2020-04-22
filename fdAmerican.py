import numpy as np
import sys

def fdAmerican(callput, S0, K, r, T, sigma, q, M, N, Smax):
    
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
    t_index = np.arange(N+1)
    s_index = np.arange(M+1)
    bounds = np.linspace(0, Smax, M+1)
    tol = 0.001
    
    #Set up coefficients and matrices
    alpha = 0.25*dt*((sigma**2)*(s_index**2) - (r-q)*s_index)
    beta = -dt*0.5*((sigma**2)*(s_index**2) + (r))
    gamma = 0.25*dt*((sigma**2)*(s_index**2) + (r-q)*s_index)
    M1 = -np.diag(alpha[2:M], -1) + np.diag(1-beta[1:M]) - np.diag(gamma[1:M-1], 1)
    M2 = np.diag(alpha[2:M], -1) + np.diag(1+beta[1:M]) + np.diag(gamma[1:M-1], 1)
    
    #Check for a call
    if callput == 1:
        
        payoffs = np.maximum(bounds[1:M]-K, 0)
    
    #Check for a put    
    if callput == -1:
        
        payoffs = np.maximum(K-bounds[1:M], 0)
      
    #create new array to store payoffs
    past_vals = payoffs
    
    #Create boundaries
    lower_bounds = K * np.exp(-r *dt * (N-t_index))
    upper_bounds = 0* t_index
    
    aux = np.zeros(M-1)
    new_vals = np.zeros(M-1)
    
    #Iterate through the "grid" backwards from Time N-1
    for j in reversed(range(N)):
        
        aux[0] = alpha[1]*(lower_bounds[j] + lower_bounds[j+1])
        aux[M-2] = gamma[M-1]*(upper_bounds[j] + upper_bounds[j+1])
        
        #Set up for Gauss-Seidel scheme
        rhs = np.dot(M2, past_vals) + aux
        old_vals = np.copy(past_vals)
        err = sys.float_info.max
        
        #Run a Gauss-Seidel scheme to iterate to generate new values
        while tol < err:
            
            new_vals[0] = max(payoffs[0], old_vals[0] + 1.0/(1-beta[1]) * \
                      (rhs[0] - (1-beta[1])*old_vals[0] + (gamma[1]*old_vals[1])))

            for k in range(M-2)[1:]:
                
                new_vals[k] = max(payoffs[k], old_vals[k] + 1.0/(1-beta[k+1]) * \
                          (rhs[k] + alpha[k+1]*new_vals[k-1] - (1-beta[k+1])*old_vals[k] + \
                           gamma[k+1]*old_vals[k+1]))

            new_vals[-1] = max(payoffs[-1], old_vals[-1] + 1.0/(1-beta[-2]) * \
                      (rhs[-1] + alpha[-2]*new_vals[-2] - (1-beta[-2])*old_vals[-1]))

            
            err = np.linalg.norm(new_vals - old_vals)
            
            old_vals = np.copy(new_vals)

        past_vals = np.copy(new_vals)
        
    values = np.concatenate(([upper_bounds[0]], new_vals,[0]))
    
    #Interpolate to get option value for the given S0
    value = np.interp(S0,bounds,values)
    
    return value

S0 = 50
K = 50
Smax = 250
r = 0.025
T = 1
sigma = 0.4
q = 0
callput = 1
M = 200
N = 1000

optionPrice = fdAmerican(callput, S0, K, r, T, sigma, q, M, N, Smax)


#The code below was used to generate plots for the report
"""
import matplotlib.pyplot as plt

Ms = [25, 50 , 75, 100, 125, 150, 175, 200]
Prices = np.zeros(len(Ms))

for i in range(len(Ms)):
    price = fdAmerican(callput, S0, K, r, T, sigma, q, Ms[i], N, Smax)
    Prices[i] = price

plt.figure(figsize = (10,5))
plt.title("Number of Spacial Steps by Option Price S0=50, K=50, vol = 40%")
plt.plot(Ms, Prices)
plt.xlabel("Spacial Steps")
plt.ylabel("Option Price") 
plt.legend() 
plt.show() 

Ns = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
Prices = np.zeros(len(Ns))

for i in range(len(Ns)):
    price = fdAmerican(callput, S0, K, r, T, sigma, q, M, Ns[i], Smax)
    Prices[i] = price

plt.figure(figsize = (10,5))
plt.title("Number of Time Steps by Option Price S0=50, K=50, vol = 40%")
plt.plot(Ns, Prices)
plt.xlabel("Time Steps")
plt.ylabel("Option Price") 
plt.legend() 
plt.show() 
"""
