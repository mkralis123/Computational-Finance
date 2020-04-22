import numpy as np
import matplotlib.pyplot as plt

def BSMonteCarlo(S0, K, T, sigma, checkpoints, rateCurve = [0.0179, 0.0177,	 0.0182, 0.0181, 0.0173], samples = None):
    
    """
    S0, K, T, sigma are the underlying price, the option strike, the
    maturity and the volatility respectively.
    ‘checkpoints’ is an ordered list of integer sample counts at
    which to return the running mean, standard deviation, and estimated error.
    ‘rateCurve’ is an InterestRateCurve stored as a numpy array.
    ‘samples’ is a numpy array of uniform random samples to use.
    If this input is None then your code should generate a fresh 1-
    dimensional sample array with M elements where M is the final
    entry in checkpoints.
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
    
    dic = {"TV": 0, "Means": [],"StdDevs": [],"StdErrs": []}   
    
    if samples == None:
        
        
        for m in checkpoints:
            
            samples = np.random.randn(m,1)
            
            price_samples = S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*samples)
            
            vals=np.exp(-r*T) * np.maximum(0,price_samples-K)
            
            val= np.mean(vals)
            std_val = np.std(vals)
            error_est=std_val/np.sqrt(m)
            
            dic['Means'].append(val)
            dic['StdDevs'].append(std_val)
            dic['StdErrs'].append(error_est)
            
    else:
        
        
        for m in checkpoints:
            
            
            price_samples = S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*samples)
            
            vals=np.exp(-r*T) * np.maximum(0,price_samples-K)
            
            val= np.mean(vals)
            std_val = np.std(vals)
            error_est=std_val/np.sqrt(m)
            
            dic['Means'].append(val)
            dic['StdDevs'].append(std_val)
            dic['StdErrs'].append(error_est)
        
        
    """
    This function must return a dictionary with the following entries:
        { ’TV’: , # The final value ( i.e. mean at checkpoints[-1] )
         ’Means’: , # The running mean at each checkpoint
         ’StdDevs’: , # The running standard deviation at each checkpoint
         ’StdErrs’: , # The running standard error at each checkpoint
         }

    """
    dic["TV"] = dic["Means"][-1]
    
    return dic

S0 = 50
K = 50
T = 1
sig = 0.25
checkpoints = [50, 100, 250, 500, 1000, 2500]

test_BS = BSMonteCarlo(S0, K, T, sig, checkpoints)


#The code below was used to generate graphs in the report
"""
BSMvalue = 5.372
diff = np.array(test_BS["Means"])
diff = diff - BSMvalue


plt.figure( figsize = (10,5) )
plt.title("Monte Carlo Simluator for Options Price")
plt.plot(checkpoints,diff, label = 'Means minus BSM')
plt.ylabel("Error from Black Scholes")
plt.xlabel("Number of Paths")
plt.legend()
plt.show()


plt.figure( figsize = (10,5))
plt.title("Plotting Standard Error against Number of Paths")
plt.plot(checkpoints, test_BS['StdErrs'],label = "Standard Errors")
plt.xlabel("Number of Paths")
plt.ylabel("Error")
plt.legend()
plt.show()
"""