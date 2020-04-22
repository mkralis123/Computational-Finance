import numpy as np
import pandas_datareader as web
from datetime import datetime
import matplotlib.pyplot as plt

start = datetime(2015, 11, 1)
end = datetime(2019, 11, 12)
source = 'yahoo'

Stock1 = 'AAPL'
Stock2 = 'GOOG'
Stock3 = 'MSFT'
Stock4 = 'NKE'
Stock5 = 'ATVI'

#Load in the price data for 5 different stocks
Stock1data = web.DataReader( Stock1, source, start, end )
Stock2data = web.DataReader( Stock2, source, start, end )
Stock3data = web.DataReader( Stock3, source, start, end )
Stock4data = web.DataReader( Stock4, source, start, end )
Stock5data = web.DataReader( Stock5, source, start, end )

#Load in the Adj Close Prices
Stock1P = Stock1data['Adj Close']
Stock2P = Stock2data['Adj Close']
Stock3P = Stock3data['Adj Close']
Stock4P = Stock4data['Adj Close']
Stock5P = Stock5data['Adj Close']

#Determine the number of short and long windows to process
short_windows = 20
long_windows = 20

#Create short and long windows
short = np.linspace(5, 24, short_windows).astype(int)
long = np.linspace(25, 44, long_windows).astype(int)

#Set an intial captial marker
initial_capital = 1

#Create all sharpe ratios
Stock1_sharpe = np.zeros( (len(short),len(long)) )
Stock2_sharpe = np.zeros( (len(short),len(long)) )
Stock3_sharpe = np.zeros( (len(short),len(long)) )
Stock4_sharpe = np.zeros( (len(short),len(long)) )
Stock5_sharpe = np.zeros( (len(short),len(long)) )

#Go over all different short windows
for i in range(short_windows):
    
    #Go over all different long windows
    for j in range(long_windows):
        
        #Calculate sharpe ratio of a given strategy for AAPL
        temp_strat = np.where(Stock1P.rolling(short[i]).mean() > Stock1P.rolling(long[j]).mean(), 
                      np.log(Stock1P/Stock1P.shift(1)), 0)
        temp_strat = initial_capital*np.exp(np.cumsum(temp_strat))
        temp_strat = np.log(temp_strat[1:]/temp_strat[:-1])
        
        Stock1_sharpe[i,j] = np.mean(temp_strat)/np.std(temp_strat)
        
        #Calculate sharpe ratio of a given strategy for GOOG
        temp_strat = np.where(Stock2P.rolling(short[i]).mean() > Stock2P.rolling(long[j]).mean(), 
                      np.log(Stock2P/Stock2P.shift(1)), 0)
        temp_strat = initial_capital*np.exp(np.cumsum(temp_strat))
        temp_strat = np.log(temp_strat[1:]/temp_strat[:-1])
        
        Stock2_sharpe[i,j] = np.mean(temp_strat)/np.std(temp_strat)
        
        #Calculate sharpe ratio of a given strategy for MSFT
        temp_strat = np.where(Stock3P.rolling(short[i]).mean() > Stock3P.rolling(long[j]).mean(), 
                      np.log(Stock3P/Stock3P.shift(1)), 0)
        temp_strat = initial_capital*np.exp(np.cumsum(temp_strat))
        temp_strat = np.log(temp_strat[1:]/temp_strat[:-1])
        
        Stock3_sharpe[i,j] = np.mean(temp_strat)/np.std(temp_strat)
        
        #Calculate sharpe ratio of a given strategy for ZNGA
        temp_strat = np.where(Stock4P.rolling(short[i]).mean() > Stock4P.rolling(long[j]).mean(), 
                      np.log(Stock4P/Stock4P.shift(1)), 0)
        temp_strat = initial_capital*np.exp(np.cumsum(temp_strat))
        temp_strat = np.log(temp_strat[1:]/temp_strat[:-1])
        
        Stock4_sharpe[i,j] = np.mean(temp_strat)/np.std(temp_strat)
        
        #Calculate sharpe ratio of a given strategy for TWTR
        temp_strat = np.where(Stock5P.rolling(short[i]).mean() > Stock5P.rolling(long[j]).mean(), 
                      np.log(Stock5P/Stock5P.shift(1)), 0)
        temp_strat = initial_capital*np.exp(np.cumsum(temp_strat))
        temp_strat = np.log(temp_strat[1:]/temp_strat[:-1])
        
        Stock5_sharpe[i,j] = np.mean(temp_strat)/np.std(temp_strat)
        
#Locate the max sharpe ratio strategy for AAPL
maxm = np.where( Stock1_sharpe == np.amax(Stock1_sharpe) )
index_tup = list(zip(maxm[0], maxm[1]))
index1 = np.zeros(2).astype(int)
index1[0] = index_tup[0][0]
index1[1] = index_tup[0][1]

print("\nOPTIMAL STRATEGY FOR", Stock1)
print("\nShort Window Size:\n", short[index1[0]])
print("Long Window Size:\n", long[index1[1]])

Stock1_strat = np.where(Stock1P.rolling(short[index1[0]]).mean() > Stock1P.rolling(long[index1[1]]).mean(), 
                      np.log(Stock1P/Stock1P.shift(1)), 0)
Stock1_strat = initial_capital*np.exp(np.cumsum(Stock1_strat))
Stock1data['Strategy'] = Stock1_strat*Stock1data['Adj Close'][0]
Stock1P_strat = Stock1data['Strategy']

#Locate the max sharpe ratio strategy for GOOG
maxm = np.where( Stock2_sharpe == np.amax(Stock2_sharpe) )
index_tuple = list(zip(maxm[0], maxm[1]))
index2 = np.zeros(2).astype(int)
index2[0] = index_tuple[0][0]
index2[1] = index_tuple[0][1]

print("\nOPTIMAL STRATEGY FOR", Stock2)
print("\nShort Window Size:\n", short[index2[0]])
print("Long Window Size:\n", long[index2[1]])

Stock2_strat = np.where(Stock2P.rolling(short[index2[0]]).mean() > Stock2P.rolling(long[index2[1]]).mean(), 
                      np.log(Stock2P/Stock2P.shift(1)), 0)
Stock2_strat = initial_capital*np.exp(np.cumsum(Stock2_strat))
Stock2data['Strategy'] = Stock2_strat*Stock2data['Adj Close'][0]
Stock2P_strat = Stock2data['Strategy']

#Locate the max sharpe ratio strategy for MSFT
maxm = np.where( Stock3_sharpe == np.amax(Stock3_sharpe) )
index_tuple = list(zip(maxm[0], maxm[1]))
index3 = np.zeros(2).astype(int)
index3[0] = index_tuple[0][0]
index3[1] = index_tuple[0][1]

print("\nOPTIMAL STRATEGY FOR", Stock3)
print("\nShort Window Size:\n", short[index3[0]])
print("Long Window Size:\n", long[index3[1]])

Stock3_strat = np.where(Stock3P.rolling(short[index3[0]]).mean() > Stock3P.rolling(long[index3[1]]).mean(), 
                      np.log(Stock3P/Stock3P.shift(1)), 0)
Stock3_strat = initial_capital*np.exp(np.cumsum(Stock3_strat))
Stock3data['Strategy'] = Stock3_strat*Stock3data['Adj Close'][0]
Stock3P_strat = Stock3data['Strategy']

#Locate the max sharpe ratio strategy for ZNGA
maxm = np.where( Stock4_sharpe == np.amax(Stock4_sharpe) )
index_tuple = list(zip(maxm[0], maxm[1]))
index4 = np.zeros(2).astype(int)
index4[0] = index_tuple[0][0]
index4[1] = index_tuple[0][1]

print("\nOPTIMAL STRATEGY FOR", Stock4)
print("\nShort Window Size:\n", short[index4[0]])
print("Long Window Size:\n", long[index4[1]])

Stock4_strat = np.where(Stock4P.rolling(short[index4[0]]).mean() > Stock4P.rolling(long[index4[1]]).mean(), 
                      np.log(Stock4P/Stock4P.shift(1)), 0)
Stock4_strat = initial_capital*np.exp(np.cumsum(Stock4_strat))
Stock4data['Strategy'] = Stock4_strat*Stock4data['Adj Close'][0]
Stock4P_strat = Stock4data['Strategy']

#Locate the max sharpe ratio strategy for TWTR
maxm = np.where( Stock5_sharpe == np.amax(Stock5_sharpe) )
index_tuple = list(zip(maxm[0], maxm[1]))
index5 = np.zeros(2).astype(int)
index5[0] = index_tuple[0][0]
index5[1] = index_tuple[0][1]

print("\nOPTIMAL STRATEGY FOR", Stock5)
print("\nShort Window Size:\n", short[index5[0]])
print("Long Window Size:\n", long[index5[1]])

Stock5_strat = np.where(Stock5P.rolling(short[index5[0]]).mean() > Stock5P.rolling(long[index5[1]]).mean(), 
                      np.log(Stock5P/Stock5P.shift(1)), 0)
Stock5_strat = initial_capital*np.exp(np.cumsum(Stock5_strat))
Stock5data['Strategy'] = Stock5_strat*Stock5data['Adj Close'][0]
Stock5P_strat = Stock5data['Strategy']

#Plot the strategies
plt.figure( figsize = (10,5) )
plt.plot(Stock1_strat)
plt.plot(Stock2_strat)
plt.plot(Stock3_strat)
plt.plot(Stock4_strat)
plt.plot(Stock5_strat)
plt.title("Portfolio Value from Strategies")
plt.xlabel("Days from begining of strategy")
plt.ylabel("Portfolio Value")
plt.legend((Stock1,Stock2,Stock3,Stock4,Stock5))
plt.show()

#Stock1 plot
plt.figure( figsize = (10,5) )
plt.plot( Stock1P )
plt.plot( Stock1P_strat )
plt.plot( Stock1P.rolling(short[index1[0]]).mean() )
plt.plot( Stock1P.rolling(long[index1[1]]).mean() )
plt.title("Optimal " + Stock1 + " Moving Average Crossover Strategy")
plt.legend(("Stock Price", "Strategy", "Short Moving Average", "Long Moving Average"))
plt.show()

#Stock2 plot
plt.figure( figsize = (10,5) )
plt.plot( Stock2P )
plt.plot( Stock2P_strat )
plt.plot( Stock2P.rolling(short[index2[0]]).mean() )
plt.plot( Stock2P.rolling(long[index2[1]]).mean() )
plt.title("Optimal " + Stock2 + " Moving Average Crossover Strategy")
plt.legend(("Stock Price", "Strategy", "Short Moving Average", "Long Moving Average"))
plt.show()

#Stock3 plot
plt.figure( figsize = (10,5) )
plt.plot( Stock3P )
plt.plot( Stock3P_strat )
plt.plot( Stock3P.rolling(short[index3[0]]).mean() )
plt.plot( Stock3P.rolling(long[index3[1]]).mean() )
plt.title("Optimal " + Stock3 + " Moving Average Crossover Strategy")
plt.legend(("Stock Price", "Strategy", "Short Moving Average", "Long Moving Average"))
plt.show()

#Stock4 plot
plt.figure( figsize = (10,5) )
plt.plot( Stock4P )
plt.plot( Stock4P_strat )
plt.plot( Stock4P.rolling(short[index4[0]]).mean() )
plt.plot( Stock4P.rolling(long[index4[1]]).mean() )
plt.title("Optimal " + Stock4 + " Moving Average Crossover Strategy")
plt.legend(("Stock Price", "Strategy", "Short Moving Average", "Long Moving Average"))
plt.show()

#Stock5 plot
plt.figure( figsize = (10,5) )
plt.plot( Stock5P )
plt.plot( Stock5P_strat )
plt.plot( Stock5P.rolling(short[index5[0]]).mean() )
plt.plot( Stock5P.rolling(long[index5[1]]).mean() )
plt.title("Optimal " + Stock5 + " Moving Average Crossover Strategy")
plt.legend(("Stock Price", "Strategy", "Short Moving Average", "Long Moving Average"))
plt.show()
