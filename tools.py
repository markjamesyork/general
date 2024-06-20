
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 22:26:24 2016

@author: markj
"""
import numpy as np
import os
import pandas as pd
import math
import nass #https://pypi.python.org/pypi/nass
#import python-twitter as pt #https://github.com/bear/python-twitter

def ncr(n,r):
    if int(n) != n or int(r) != r:
        return 'n and r must be integers'
    c = math.factorial(n)
    c /= math.factorial(r)*math.factorial(n-r)
    return c

def npr(n,r):
    if int(n) != n or int(r) != r:
        return 'n and r must be integers'
    p = math.factorial(n)
    p /= math.factorial(n-r)
    return p

def augmented_stats(data):
    '''This function returns the descriptive statistics for a pandas data frame.
    The statistics themselves are returned as a pandas data frame.
    '''
    from scipy import stats
    data_stats = data.describe()
    dict_temp = {}
    for key in data_stats.keys():
        new_stats = stats.describe(data[key])
        dict_temp[key] = [new_stats[4],new_stats[5]]
    df_temp = pd.DataFrame(dict_temp,index=['skewness','kurtosis'])
    data_stats = pd.concat([data_stats,df_temp])

    return data_stats

def read_progress(fname):
    data = pd.read_csv(fname)
    return data
    
def nass():
    
    return data
    
def nasa_wx():
    
    return data

def power_list(lst,remaining):
    lst.append(lst[0]*lst[-1])
    remaining -= 1
    if remaining > 0:
        lst, remaining = power_list(lst, remaining)
    return lst,remaining
    
