'''This script numerically calculates the probability distribution
of outcomes if a time-limited opportunity is rejected.
'''
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import scipy
from tools import*

#1 Set parameters
n = 10**5

#2 Read options, values and probabilities
data = pd.read_csv('options.csv')
m = data.shape[0]
    
#3 Create n Monte Carlo scenarios
outcomes = np.random.random((n,m))
p = np.tile(data['p'],(n,1))
outcomes = np.clip(scipy.sign(p - outcomes),0,1) #matrix of binary random outcomes
outcomes = np.multiply(outcomes, list(data['value']))
values = np.amax(outcomes,axis=1)

#4 Print results
print augmented_stats(pd.DataFrame({'values':values}))
print '5th Percentile:', np.percentile(values,5)
print '95th Percentile:', np.percentile(values,95)
plt.hist(values,bins = max(10,int(n/1000.)))
plt.title('Value Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
fig = plt.gcf()
fig.savefig('figs/decision.png')
plt.close(fig)
