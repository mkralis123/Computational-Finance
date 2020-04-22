import numpy as np
import matplotlib.pyplot as plt

def MCOptionPrices( S0, K, T, sigma, t, checkpoints , integrator, rateCurve = [0.0179, 0.0177, 0.0182, 0.0181, 0.0173], samples = None):
    
    """
    ‘t’ is an array of fixing times ti, i = 1 . . . N to simulate to.
    ’K’ is the strike price.
    ‘T’ is the expiration date of the European option.
    ‘rateCurve’ is an InterestRateCurve stored as a numpy array.
    ‘checkpoints’ is an ordered list of integer sample counts in
    the range [1, M] at which to return the running mean, standard
    deviation, and estimated error.
    ‘samples’ is an array of uniform random samples to use. The
    length of samples should be M × N where N is the number of
    fixing times and M is the number of paths.
    integrator controls how the samples are generated according
    to the following value list
        ’standard’, where the paths are generated by using the
        solution of the Black-Scholes SDE step-by-step
        ’euler’, to use Euler-method integration of the BlackScholes SDE
        ’milstein’, to use Milstein-method integration of the BlackScholes SDE
    """
    if T >= 1:
        r = rateCurve[-1]
    elif T > 0.5:
        r = rateCurve[-1]
    elif T > 0.25:
        r = rateCurve[-2]
    elif T > (1/6):
        r = rateCurve[-3]
    elif T > (1/12):
        r = rateCurve[-4]
    else: 
        r = rateCurve[-5]
    
    M = len(checkpoints)
    N = len(t)
    
    
    dic = {"TV": 0, "Means": np.empty((M,N)),"StdDevs": np.empty((M,N)),"StdErrs": np.empty((M,N))}
    
    for i in range(N):
        
        num_ts = t[i]
        
        for j in range(M):
            
            samples = np.random.randn(checkpoints[j], num_ts)
                
            if integrator == 'standard':
                
                standard_integration = S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*np.sum(samples,axis=1)/np.sqrt( num_ts ))
                
                vals=np.exp(-r*T) * np.maximum(0,standard_integration-K)
                vals_std = np.std(vals)
                
                dic["Means"][j,i] = np.mean(vals)
                dic["StdDevs"][j,i] = vals_std
                dic["StdErrs"][j,i] = vals_std/np.sqrt(checkpoints[j])
                
            
            elif integrator == 'euler':
                
                euler_integration = S0 * np.prod(1.+(r)*T/num_ts + sigma*np.sqrt(T/num_ts)*samples, axis=1)
                
                vals=np.exp(-r*T) * np.maximum(0,euler_integration-K)
                vals_std = np.std(vals)
                
                dic["Means"][j,i] = np.mean(vals)
                dic["StdDevs"][j,i] = vals_std
                dic["StdErrs"][j,i] = vals_std/np.sqrt(checkpoints[j])
                
            elif integrator == 'milstein':
                
                delStochCoeffdelAsset = sigma
                
                milstein_integration = S0*np.prod(1.+(r)*T/num_ts + sigma*np.sqrt(T/num_ts)*samples + 0.5*sigma*delStochCoeffdelAsset*(samples**2*T/num_ts-T/num_ts), axis=1)
                
                vals=np.exp(-r*T) * np.maximum(0,milstein_integration-K)
                vals_std = np.std(vals)
                
                dic["Means"][j,i] = np.mean(vals)
                dic["StdDevs"][j,i] = vals_std
                dic["StdErrs"][j,i] = vals_std/np.sqrt(checkpoints[j])
                
            else:
                return None
    
    dic["TV"] = np.mean(dic["Means"])

    
    
    
    """
    This function must return a dictionary with the following entries:
        { ’TV’: , # The final value ( i.e. mean of option price at time t_0
         using MxN uniform random samples)
        ’Means’: , # The running mean at each checkpoint
        ’StdDevs’: , # The running standard deviation at each checkpoint
        ’StdErrs’: , # The running standard error at each checkpoint
        }
    """
    return dic

S0 = 50
K = 50
T = 1
sig = 0.25
t = [10,25,100,250]
checkpoints = [50, 100, 250, 500, 1000, 2500]

test_option_standard = MCOptionPrices(S0,K,T,sig,t, checkpoints,'standard')
test_option_euler = MCOptionPrices(S0,K,T,sig,t, checkpoints,'euler')
test_option_milstein = MCOptionPrices(S0,K,T,sig,t, checkpoints,'milstein')

#The code below was used to generate graphs for the report
"""
plt.figure( figsize = (10,5) )
plt.title("Plotting Standard Error of Number of Paths (Keeping N fixed)")
plt.plot(checkpoints, test_option_standard['StdErrs'][:,-1], label = 'Standard Standard Error')
plt.plot(checkpoints, test_option_euler['StdErrs'][:,-1], label = 'Standard Euler Error')
plt.plot(checkpoints, test_option_milstein['StdErrs'][:,-1], label = 'Standard Milstein Error')
plt.xlabel("Number of Paths")
plt.ylabel("Error")
plt.legend()
plt.show()
"""