# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:51:57 2020

@author: baccarini_a
"""


import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#%% Main
def AO18_Alert_comp(CIMS,Alert):
    '''
    CIMS data can be find: https://doi.org/10.17043/ao2018-aerosol-cims
    Alert data can be donwnloaded from the EBAS database
    '''
    
    # =============================================================================
    # Prepare data    
    # =============================================================================
    # Alert
    # Get mean julian day per measurements
    Alert['DayofYear'] = ((Alert.Start.dt.dayofyear+Alert.End.dt.dayofyear)/2).round(0).astype(int)
    
    # report everything to 2018 for comparison
    Alert['Date'] = pd.to_datetime(np.repeat(2018,len(Alert.index)),format='%Y')+pd.to_timedelta(Alert['DayofYear'],unit='d')
    Alert['WeekofYear'] = Alert['Date'].dt.weekofyear
    
    # Group data
    AlertGrouped = pd.DataFrame(index = np.sort(Alert['WeekofYear'].unique()))
    AlertGrouped['mean'] = Alert[['Iodine [ug m-3]','WeekofYear']].groupby('WeekofYear').mean()
    AlertGrouped['std'] = Alert[['Iodine [ug m-3]','WeekofYear']].groupby('WeekofYear').std()

    # CIMS
    IodicAcid = CIMS['IO3-'][CIMS['Pollution mask']==1]*126.9*1e12/6.022E+23
    IodicAcidrs=pd.DataFrame({'Iod mean':IodicAcid.resample('1W').mean()})
    IodicAcidrs['Iod std']=IodicAcid.resample('1W').std()

    # =============================================================================
    # Plot
    # =============================================================================

    fig,axs=plt.subplots(figsize=(18,10),constrained_layout=True)
    axs.plot(AlertGrouped['mean'],'o',label='Aerosol iodine, Alert 1981-2006',markersize=12,markeredgecolor='k')
    
    minAlert=AlertGrouped['mean']-AlertGrouped['std']
    minAlert[minAlert<0]=0 #negative values don't make sense in this plot
    maxAlert=AlertGrouped['mean']+AlertGrouped['std']
    
    axs.fill_between(AlertGrouped.index, minAlert,
                     maxAlert,alpha=0.4)
    
    axs.plot(IodicAcidrs.index.week,IodicAcidrs['Iod mean'],'o',label='Gasphase iodic acid, AO18',markersize=12,markeredgecolor='k')
    minIO3=IodicAcidrs['Iod mean']-IodicAcidrs['Iod std']
    minIO3[minIO3<0]=0
    axs.fill_between(IodicAcidrs.index.week,minIO3,
                     IodicAcidrs['Iod mean']+IodicAcidrs['Iod std'],alpha=0.4)
    
    
    
    idx_axis=pd.date_range(start='2018-07-01',end='2018-11-08',freq='W')
    
    
    axs.set_xticks(idx_axis.week)
    axs.xaxis.set_major_formatter(ticker.FixedFormatter(idx_axis.strftime('%m-%d')))
    
    axs.set_ylabel(r'Concentration [$\mu$g m$^{-3}$]')
    axs.set_xlabel('Date [Month - Day]')
    fig.autofmt_xdate()
    #axs.set_title('Aerosol Iodine intercomparison \n weekly average +/- standard deviation')
    axs.legend()
    axs.set_xlim((25.21,44.83))
    
    plt.tight_layout()


