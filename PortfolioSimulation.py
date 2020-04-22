import numpy as np
import matplotlib.pyplot as plt

def PortfolioEvolution(r, sig, covmat, weights, M, N, T):
    
    """
    r is an array of rates of return (annualized)
    sig is an array of standard deviation of return (annualized)
    covmat is the covariance matrix of returns
    weights is an array of how much (in dollar amount) is in each asset
    M is the number of paths
    N is the number of time steps
    T is the time to simulate to
    """
    
    
    # Check to make sure the dimensions are all consistent with eachother
    if np.shape(r) == np.shape(sig) and np.shape(r) == np.shape(weights) and np.shape(covmat)[0] == np.shape(r)[0] and np.shape(covmat)[0] == np.shape(covmat)[1]:
        
        # Make sure the M and N are integers
        N = int(N)
        M = int(M)
        
        # Calculate size of each time step
        dt = T/N
        
        # Decompose the covariance matrix
        chol = np.linalg.cholesky(covmat)
        
        # Create an integer P that represents number of stocks in simulation
        P = len(r)
        
        # Create a stock matrix that will update through each time step
        STMat = np.ones((P, M))
        STMatEuler = np.ones((P, M))
        
        # Create a portval matrix that will append the new values based
        # on the stock prices that were simulated in the stock matrix
        PortVal = np.ones((N+1, M))
        PortValEuler = np.ones((N+1, M))
        
        # Initialize the first values across all paths to be the sum of weights
        PortVal[0,:] = np.sum(weights)
        PortValEuler[0,:] = np.sum(weights)
        
        # Initialize the initial stock prices based the weights
        for k in range(P):
            
            STMat[k,:] = weights[k]*STMat[k,:]
            STMatEuler[k,:] = weights[k]*STMatEuler[k,:]
        
        #Iterate forwards in time
        for i in range(N): 
    
            # Generate random numbers and multiply them with the cholesky decomp
            x = chol.dot(np.random.randn(P,M))
            
            # Generate the standard Monte Carlo simulation interating through each stock
            for k in range(P):
                
                STMat[k,:] = STMat[k,:]*np.exp((r[k]-0.5*sig[k]**2)*dt + np.sqrt(dt)*x[k])
                STMatEuler[k,:] = STMatEuler[k,:]*(1 + r[k]*dt + np.sqrt(dt)*x[k])
                
            # Update Portfolio Value by taking the sum
            PortVal[i+1,:] = np.sum(STMat, axis = 0)
            PortValEuler[i+1,:] = np.sum(STMatEuler, axis = 0)
        
        return PortVal, PortValEuler
    
    else:
        
        return None
    
    
# Input parameters for the portfolio evolution
r = [0.1,0.1]
sig = [0.1,0.1]
covmat = np.array([[sig[0]**2,-0.001],[-0.001,sig[1]**2]])
weights = [50,50]
M = 100
N = 100
T = 1

# Plot the evolutions of each scheme

PortValSTD, PortValEuler = PortfolioEvolution(r, sig, covmat, weights, M, N, T)

# Plot the standard scheme evolution
plt.figure(figsize = (12,5))
plt.plot(PortValSTD)
plt.title("Portfolio Simulation Standard Monte Carlo")
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value")
plt.show()

# Plot the Euler scheme evolution
plt.figure(figsize = (12,5))
plt.plot(PortValEuler)
plt.title("Portfolio Simulation Euler Scheme")
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value")
plt.show()

# Print out the 99th percentile loss for each scheme
print("\n")
print("Your 99th percentile for loss in the Standard Scheme is \n")
print(np.sum(weights) - np.sort(PortValSTD[N,:])[int(M*0.01)])

print("\n")
print("Your 99th percentile for loss in the Euler Scheme is \n")
print(np.sum(weights) - np.sort(PortValEuler[N,:])[int(M*0.01)])


# The code below is used for plotting losses against time steps and paths
# If you wish to run some convergence plots then uncomment the code below


steps = 50

Ns = np.linspace(200, 5000, steps).astype(int)
Ms = np.linspace(200, 5000, steps).astype(int)

PortLossSTDMs = np.zeros((steps, 1))
PortLossEulerMs = np.zeros((steps, 1))

PortLossSTDNs = np.zeros((steps, 1))
PortLossEulerNs = np.zeros((steps, 1))

for i in range(steps):
    
    PortValSTD, PortValEuler = PortfolioEvolution(r, sig, covmat, weights, Ms[i], N, T)
    PortLossSTDMs[i, 0] =  np.sum(weights) - np.sort(PortValSTD[N,:])[int(Ms[i]*0.01)]
    PortLossEulerMs[i, 0] = np.sum(weights) - np.sort(PortValEuler[N,:])[int(Ms[i]*0.01)]
    
    PortValSTD, PortValEuler = PortfolioEvolution(r, sig, covmat, weights, M, Ns[i], T)
    PortLossSTDNs[i, 0] =  np.sum(weights) - np.sort(PortValSTD[Ns[i],:])[int(M*0.01)]
    PortLossEulerNs[i, 0] = np.sum(weights) - np.sort(PortValEuler[Ns[i],:])[int(M*0.01)]
    
plt.figure(figsize = (12,5))
plt.plot(Ms, PortLossSTDMs)
plt.plot(Ms, PortLossEulerMs)
plt.title("99th Portfolio Losses by Number of Paths")
plt.xlabel("Number of Paths")
plt.ylabel("Loss on the portfolio")
plt.legend(("Standard", "Euler"))
plt.show()

plt.figure( figsize = (12,5) )
plt.plot(Ns, PortLossSTDNs)
plt.plot(Ns, PortLossEulerNs)
plt.title("99th Portfolio Losses by Number of Time-Steps")
plt.xlabel("Number of Time-Steps")
plt.ylabel("Loss on the portfolio")
plt.legend(("Standard", "Euler"))
plt.show()