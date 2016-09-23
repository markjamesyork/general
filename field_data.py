# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 06:57:38 2016

@author: markj
"""
import datetime as dt
import math
#import matplotlib as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import multiprocessing as mp
import numpy as np
import pandas as pd
import pdb
import plotly.plotly as py
from pandas import Series
from scipy import stats
#import statsmodels.api as sm

def loc_match(file1,file2):
    '''This function fulfills part 1 of the FBN problem. It iterates through each
    point in file1 and finds the nearest point in file2. Once making this association,
    it creates a dataframe with file2 data included for each associated file1 point.
    '''
    #Variable Declarations
    t1 = dt.datetime.now()
    max_dist = 10. #maximum distance in meters of lat or lon to be considered for matching to file1 point.
    m_per_deg_lat = 111320.
    new_fields = ['variety','seeding_rate','seed_spacing']
    
    #1 Read files and calculate offsets
    f1 = pd.read_csv(file1)
    f2 = pd.read_csv(file2)
    lat = np.mean(f1['lat'])
    m_per_deg_lon = math.cos(math.pi*lat/180) * m_per_deg_lat
    lat_offset = max_dist/m_per_deg_lat
    lon_offset = max_dist/m_per_deg_lon
    
    #2 Loop through each point in file1 and find matching points in file2
    for item in new_fields:
        f1[item] = Series(np.zeros(len(f1['lat'])), index=f1.index)
    for i in f1.index:
        f2_trim = f2.loc[lambda df: abs(df.lat - f1['lat'][i])<lat_offset,:]
        f2_trim = f2_trim.loc[lambda df: abs(df.long - f1['long'][i])<lon_offset,:]
        dist = (((f2_trim['lat']-f1['lat'][i])*m_per_deg_lat)**2 + ((f2_trim['long']\
                    -f1['long'][i])*m_per_deg_lon)**2)**.5
        match = dist.index[np.argwhere(dist == min(dist))[0][0]]
        print 'index', i
        print 'min(dist)',min(dist)
        print 'len(dist)', len(dist)
        print 'match',match
        for item in new_fields:
            f1[item][i] = f2_trim[item][match]

    #3 Write aggregated data to a .csv file
    f1.to_csv('matched_locs.csv')
    
    print 'loc_match runtime:', dt.datetime.now() - t1
    return f1

def eda(file3):
    '''This function reads a .csv file containing point yield data and associated
    planting data. The function then uses these data to establish relationships
    between planting variables and yield and creates various visualizations of these
    data
    Helpful charting website: http://pandas.pydata.org/pandas-docs/stable/visualization.html
    '''
    #1 Variable Declarations    
    t1 = dt.datetime.now()
    data = pd.read_csv(file3)
    colors = ['DarkBlue','DarkGreen','Red','Orange']
    max_dist = 10. #maximum distance in meters of lat or lon to be considered for matching to file1 point.
    m_per_deg_lat = 111320.
    lat = np.mean(data['lat'])
    m_per_deg_lon = math.cos(math.pi*lat/180) * m_per_deg_lat
    lat_offset = max_dist/m_per_deg_lat
    lon_offset = max_dist/m_per_deg_lon
    
    #2 Calculate and print basic statistics for each variable
    print 'Overall Summary Statistics:', augmented_stats(data)
    df = {}
    for key in data['variety'].value_counts().keys():
        df[key] = data.loc[lambda df: df.variety == key,:]
        print key, 'Summary Statistics: ',  augmented_stats(df[key])
        
    #3 Create a histogram for each variable
    for key in data.keys():
        if key == 'Unnamed: 0': continue #skips index
        if type(data[key][0]) != str: #Numeric Data
            plt.hist(data[key],bins = 100)
        else: #Non-numeric data
            D = data[key].value_counts()
            plt.bar(range(len(D)), D.values, align='center')
            plt.xticks(range(len(D)),D.keys())  
        plt.title(key + ' Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        fig = plt.gcf()
        fig.savefig(key + '.png')
        plt.close(fig)
        
    #4 Create a scatter plot of each dependent variable relative to yield
    for key in data.keys():
        if key in ['Unnamed: 0','lat','long','yield','variety']: continue
        ax = df[df.keys()[0]].plot.scatter(x=key, y='yield', color=colors[0], label=df.keys()[0])
        for i in range(1,len(df.keys())):
            df[df.keys()[i]].plot.scatter(x=key, y='yield', color=colors[i], label=df.keys()[i], ax=ax)
        plt.title(key + ' vs. Yield')
        fig = plt.gcf()
        fig.savefig(key + '_scatter.png')
        plt.close(fig)
        
    #5 Run Regression
    '''
    variety_map = {df.keys()[0]:0,df.keys()[1]:1}
    X = np.ones((len(data['variety']),4))
    X[:,1] = data['seed_spacing']
    X[:,2] = data['seeding_rate']
    X[:,3] = [variety_map[k] for k in data['variety']]
    model = sm.OLS(data['yield'], X)
    result = model.fit()
    print '*************** Linear Regression Yield Model ****************'
    print 'Features', ['intercept','seed_spacing','seeding_rate','variety']
    print 'Varieties', variety_map
    print 'T Values', result.tvalues
    print 'Coefficients', result.params
    stderr = (np.sum(np.square(result.predict(X) - data['yield']))/len(data['yield']))**0.5
    print 'Model Standard Error', stderr
    print 'Yield Standard Deviation', np.std(data['yield'])
    '''
    
    #6 Create maps for each variable
    for key in data.keys():
        if key in ['Unnamed: 0','lat','long']: continue
        if key == 'variety':    
            ax = df[df.keys()[0]].plot.scatter(x='long', y='lat', color=colors[0], label=df.keys()[0])
            for i in range(1,len(df.keys())):
                df[df.keys()[i]].plot.scatter(x='long', y='lat', color=colors[i], label=df.keys()[i], ax=ax)
        else:
            #ax = data.plot.scatter(x='long', y='lat', c=key, s=50, label=df.keys()[0])
            data.plot.hexbin(x='long', y='lat', C=key, reduce_C_function=np.mean, gridsize=25)
        plt.title(key + ' Map')
        plt.xlim(min(data['long'])-.0001,max(data['long']+.0001))
        plt.ylim(min(data['lat'])-.0001,max(data['lat']+.0001))
        fig = plt.gcf()
        fig.savefig(key + '_map.png')
        plt.close(fig)
        
    #7 Side-by-side yield comparison
    '''Since the two varieties were planted in different parts of the field, the yield
    comparison between them is not a true test of their yield. The code below finds
    point where the two varieties were planted side-by-side and compares their yield
    potential on this basis.
    '''
    dfs = df
    if len(df) <3: #skip this comparison if more than 2 varieties are present
        if len(df[dfs.keys()[0]]['yield']) > len(dfs[dfs.keys()[0]]['yield']):
            key = dfs.keys()[1]
            key2 = dfs.keys()[0]
        else:
            key = dfs.keys()[0]
            key2 = dfs.keys()[1]
        yields = {dfs.keys()[0]:[], dfs.keys()[1]:[]}
        for i in dfs[key].index:
            trim = dfs[key2].loc[lambda df: abs(df.lat - dfs[key]['lat'][i])<lat_offset,:]
            trim = trim.loc[lambda df: abs(df.long - dfs[key]['long'][i])<lon_offset,:]
            if len(trim['yield']) > 0:
                dist = (((trim['lat']-df[key]['lat'][i])*m_per_deg_lat)**2 + ((trim['long']\
                            -dfs[key]['long'][i])*m_per_deg_lon)**2)**.5
                match = dist.index[np.argwhere(dist == min(dist))[0][0]]
                yields[key].append(df[key]['yield'][i])
                yields[key2].append(trim['yield'][match])
        print '*************** Side-by-Side Yield Comparison Results ***************'
        dif = [k-v for (k,v) in zip(yields[key], yields[key2])]
        for key in dfs.keys():
            print key, 'mean:', np.mean(yields[key]), 'Standard Deviation', np.std(yields[key])
        print 'Mean Difference:', np.mean(dif)
        print 'Max Difference:', np.max(dif)
        print 'Min Difference:', np.min(dif)
        print 'Difference Standard Error:', np.std(dif)
        conf = 1.96*np.std(dif)/(len(yields[dfs.keys()[0]])**.5)
        print 'Difference 95% Confidence Interval:', np.mean(dif)-conf, np.mean(dif)+conf
        print 'Sample Size:', len(yields[dfs.keys()[0]])
    
    print 'eda runtime:', dt.datetime.now() - t1
    return    
    
def augmented_stats(data):
    data_stats = data.describe()
    dict_temp = {}
    for key in data_stats.keys():
        new_stats = stats.describe(data[key])
        dict_temp[key] = [new_stats[4],new_stats[5]]
    df_temp = pd.DataFrame(dict_temp,index=['skewness','kurtosis'])
    data_stats = pd.concat([data_stats,df_temp])

    return data_stats
    
        
file1 = 'harvest_sample_data.csv'
file2 = 'planting_sample_data.csv'
file3 = 'matched_locs.csv'
#loc_match(file1,file2)
eda(file3)
