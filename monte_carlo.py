import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override() # <== that's all it takes :-)

def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change(fill_method=None)
    print(returns)
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['BNS.TO', 'GOOGL', 'XOM', 'NIO', 'KO', 'PEP','ARCC','IBM','AGNC','LCID']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

print(covMatrix)

# These weights represent the relative importance or allocation of each stock in the portfolio. 
# For instance, if there are three stocks in the portfolio and the weights are [0.3, 0.4, 0.3], 
# it means that 30% of the portfolio is invested in the first stock, 40% in the second, and 30% in the third.
# Generate random numbers
weights = np.random.random(len(meanReturns))
# find the proportion of each number over the total 
# to generate random proportions
weights = weights/np.sum(weights)



# Monte Carlo simulation
sims = 1000
time = 100

meanM = np.full(shape=(time,len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(time, sims), fill_value=0)

# print(meanM)

initial = 100000

for sim in range(sims):
    Z = np.random.normal(size=(time, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L,Z)
    portfolio_sims[:,sim] = np.cumprod(np.inner(weights,dailyReturns.T)+1) * initial


# print(portfolio_sims)
plt.plot(portfolio_sims)
plt.ylabel("Portfolio Value ($)")
plt.xlabel("Days")
plt.title("MC Simulation")
plt.show()