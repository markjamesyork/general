# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:01:44 2016

@author: markj
#Adjust parameters on lines 65-72
Life table source: https://www.ssa.gov/oact/STATS/table4c6.html
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def power_list(lst,remaining):
    lst.append(lst[0]*lst[-1])
    remaining -= 1
    if remaining > 0:
        lst, remaining = power_list(lst, remaining)
    return lst,remaining

def bar_chart(plans, exp, min_90, max_90):
    N = len(plans)
    ind = np.arange(N)  # the x locations for the groups
    width = 1/4.       # the width of the bars
    global ax
    fig, ax = plt.subplots()
    plot1 = ax.bar(ind, exp, width, color='r', yerr=[(b-a)/2 for a,b in zip(min_90,max_90)])
    plot2 = ax.bar(ind + width, min_90, width, color='y')
    plot3 = ax.bar(ind + 2*width, max_90, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_xlabel('Plan')
    ax.set_ylabel('Total 2016 Dollars Paid Out')
    ax.set_title('Retirement Plan Comparison')
    ax.set_xticks(ind + width*1.6)
    ax.set_xticklabels(tuple(plans))

    #ax.legend((plot1[0],plot2[0],plot3[0]), ('Expected','5th%', '95th%'))

    autolabel(plot1)
    autolabel(plot2)
    autolabel(plot3)

    plt.show()    
    return

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#*****MAIN RETIREMENT ANALYSIS*****#
   
#1 Read and chart data
data = pd.read_csv('life_table.csv',index_col=0)
print(data)
fig = plt.figure()
plt.plot(data.index,data.m_life_expectancy,c='b')
plt.plot(data.index,data.f_life_expectancy,c='r')
plt.close()

#2 Hard-coded monthly payment and current_state data
plans = ['100%', '75%', '50%', '15yr', 'refund', 'no_refund']
pmt = [2292.95, 2338.8, 2386.59, 2439.44, 2482.18, 2487.62]
#plt.bar(range(6),pmt,width=1,align='center',color='g')
e_age = 58 #employee age, rounded to nearest whole year
s_age = 57 #spouse age, rounded to nearest whole year
e_gen = 'f'
s_gen = 'm'
annual_loss_real_pct = 1

#3 Setup percentile dictionaries
cohort = np.asarray(data[e_gen+'_alive_100000'][e_age-1:])
e_p_death = []
for n in range(len(cohort)-1):
    e_p_death.append((cohort[n]-cohort[n+1])/float(cohort[0]))

cohort = np.asarray(data[s_gen+'_alive_100000'][s_age-1:])
s_p_death = []
for n in range(len(cohort)-1):
    s_p_death.append((cohort[n]-cohort[n+1])/float(cohort[0]))

e_p_death = np.asarray(e_p_death)
e_p_death = np.reshape(e_p_death,(1,len(e_p_death)))
s_p_death = np.asarray(s_p_death)
s_p_death = np.reshape(s_p_death,(1,len(s_p_death)))

p_death = np.dot(e_p_death.T,s_p_death)
p_e_before_s = 0

#4 Create matrices of outcomes under each scenario
result = {}
defl, tmp = power_list([1/(1+annual_loss_real_pct/100.)],max(p_death.shape))

for plan in plans: result[plan] = np.zeros((e_p_death.shape[1],s_p_death.shape[1]))
for i in range(e_p_death.shape[1]):
    for j in range(s_p_death.shape[1]):
        if i < j: p_e_before_s += p_death[i,j]
        result['100%'][i,j] = np.sum([n*m for n,m in zip(max(i,j)*[pmt[0]*12],defl[:max(i,j)])]) #assumes death at the end of the year starting at t=0
        tmp = i*[pmt[1]*12] + np.clip(j-i,0,100)*[pmt[1]*12*.75]
        result['75%'][i,j] = np.sum([n*m for n,m in zip(tmp,defl[:len(tmp)])])
        tmp = i*[pmt[2]*12] + np.clip(j-i,0,100)*[pmt[2]*12*.5]
        result['50%'][i,j] = np.sum([n*m for n,m in zip(tmp,defl[:len(tmp)])])
        result['15yr'][i,j] = np.sum([n*m for n,m in zip(max(i,15)*[pmt[3]*12],defl[:max(i,15)])])
        result['refund'][i,j] = np.sum([n*m for n,m in zip(max(i,4)*[pmt[4]*12],defl[:max(i,4)])])
        result['no_refund'][i,j] = np.sum([n*m for n,m in zip(i*[pmt[5]*12],defl[:i])])

#5 Combine outcome and probability matrices to create expected values and confidence intervals
expected_value = []
min_90 = []
median = []
max_90 = []
freq_death = np.round(p_death*10**6,0)
for plan in plans:
    print(plan)
    expected_value.append(np.round(np.sum(np.multiply(p_death,result[plan])),-3))
    tmp = []
    for i in range(e_p_death.shape[1]):
        for j in range(s_p_death.shape[1]):
            tmp += [result[plan][i,j]]*freq_death[i,j]
    min_90.append(np.round(np.percentile(tmp,5),-3))
    median.append(np.round(np.percentile(tmp,50),-3))
    max_90.append(np.round(np.percentile(tmp,95),-3))
    
#6 Print and chart results
print('Plans:',plans)
print('Expected Values:',expected_value)
print('90% Min:',min_90)
print('90% Max:',max_90)
print('Median:',median)
print('Probability of Employee Dying Before Spouse:',np.round(p_e_before_s,2))
#plt.bar(range(len(plans)),expected_value,color='r',width=1,align='center')
bar_chart(plans,expected_value,min_90,max_90)