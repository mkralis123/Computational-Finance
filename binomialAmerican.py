import numpy as np

def binomialAmerican(S0, Ks, r, T, sigma, q, callputs, M):
    
    """
    S0 is the current stock price
    Ks is an array of different strike prices to evalute the option price at
    r is the risk free interest rate
    T is the time till expiration
    sigma is the annualized volatility
    q is the annualized dividend yield rate
    callputs is 1 for calls and -1 for puts
    M is the number of time steps in the tree
    """
    #Set up all values needed for binomial tree
    M = int(M)
    dt = T/M
    df = np.exp(-(r)*dt)
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    pu = (np.exp((r-q)*dt)-d)/(u-d)
    pd = 1-pu
    numK = len(Ks)
    
    #Initialize arrays for future use
    STs = np.ones(M+1)
    payoffs = np.ones(M+1)
    payoffs_from_Ks = np.ones(numK)
    
    #Check if calls are wanted
    if callputs == 1:
        
        #Run payoffs for each strike price
        for j in range(numK):
            
            #Calculate end nodes (stock prices)
            for i in range(M+1):
                
                STs[i] = (S0 * (d**(M-i)) * (u**(i)))
            
            #Calculate option payoffs at each end node
            payoffs = np.maximum(STs-Ks[j],0)
            
            #Traverse tree backwards
            for k in range(M):
                
                #Move the stock tree and options tree back 1 period
                STs = STs[:-1]*u
                payoffs = ((payoffs[:-1]*pd)+(payoffs[1:]*pu))*df
                
                #Check Intrinsic Value against expected payoff
                """Take this out and it becomes a european option"""
                payoffs = np.maximum(payoffs,STs-Ks[j])
            
            #Add terminal Value from loop for the given strike price
            payoffs_from_Ks[j] = payoffs[0]
            
            #Re-initialize option values and stock values
            payoffs = np.ones(M+1)
            STs = np.ones(M+1)
            
    #Check if puts are wanted        
    elif callputs == -1:
        
        #Run payoffs for each strike price
        for j in range(numK):
            
            #Calculate end nodes (stock prices)
            for i in range(M+1):
                
                STs[i] = (S0 * (d**(M-i)) * (u**(i)))
            
            #Calculate option payoffs at each end node
            payoffs = np.maximum(Ks[j]-STs,0)
            
            #Traverse tree backwards
            for k in range(M):
                
                #Move the stock tree and options tree back 1 period
                STs = STs[:-1]*u
                payoffs = ((payoffs[:-1]*pd)+(payoffs[1:]*pu))*df
                
                #Check Intrinsic Value against expected payoff
                """Take this out and it becomes a european option"""
                payoffs = np.maximum(payoffs,Ks[j]-STs)
                
            #Add terminal Value from loop for the given strike price    
            payoffs_from_Ks[j] = payoffs[0]
            
            #Re-initialize option values and stock values
            payoffs = np.ones(M+1)
            STs = np.ones(M+1)
    
    #If callputs isn't 1 or -1 return nothing        
    else:
        return None
    
    #return the option prices in order that strike prices were given    
    return payoffs_from_Ks

S0 = 40
Ks = [39]
r = 0.05
T = 3
sigma = 0.1
q = 0.01
callputs = 1
M = 3

optionPrices = binomialAmerican(S0, Ks, r, T, sigma, q, callputs, M)

#The code below generated the plots for the report
"""
import matplotlib.pyplot as plt

Ms = [50, 100, 150, 200, 250, 300, 350, 400, 450, 400]
Prices = np.zeros(len(Ms))

for i in range(len(Ms)):
    prices = binomialAmerican(S0, [50], r, T, sigma, q, callputs, Ms[i])
    Prices[i] = prices

plt.figure(figsize = (10,5))
plt.title("Number of Time steps by Option Price S0=50, K=50, vol = 40%")
plt.plot(Ms, Prices)
plt.xlabel("Time Steps")
plt.ylabel("Option Price") 
plt.legend() 
plt.show()  
""" 