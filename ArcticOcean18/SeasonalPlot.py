# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:53:33 2020

@author: baccarini_a
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:25:28 2019

@author: baccarini_a
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors


#%%matplotlib parameters
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['font.size'] = 27
#%% Main

def seasonal_boxplot(UFP, CIMS, Weather, Ozone):
    # =============================================================================
    # Prepare UFP for boxplot
    # =============================================================================
    UFPboxplot=pd.DataFrame({'dayyear':UFP.index.floor('D'),
                             'UFP_Conc':UFP})
    # =============================================================================
    # CIMS
    # =============================================================================
    master_time_10min= pd.date_range(start='2018-08-01 14:00:00', end='2018-09-19 21:00:00',freq='10min')
    master_time_hr= pd.date_range(start='2018-08-01 14:00:00', end='2018-09-19 21:00:00',freq='1h')
    
    IO3clean=CIMS['IO3-'][CIMS['Pollution mask']==1]
    IO3clean=IO3clean.reindex(master_time_10min)

    IO3box=pd.DataFrame({'dayyear':IO3clean.index.floor('D'),
                         'IO3':IO3clean})

    # =============================================================================
    # Temperature
    # =============================================================================
    Temperature=Weather['temp'].resample('10min').median().reindex(master_time)    
    Tdaily=Temperature.resample('d').median()
    
    # Warm and cold Temperature
    Tfreeze=Temperature[Temperature<=-2].reindex(master_time_10min)
    Twarm=Temperature[Temperature>=-2].reindex(master_time_10min)
    
    # Create fake minute axis for plotting on top pf boxplot
    numMin=len(master_time_10min)
    trd_xarray=np.linspace(0.5,50.5,np.int(numMin))
    
    # create color palette
    norm=colors.Normalize(vmin=np.floor(Tdaily.min()),vmax=11)
    Tcol=plt.cm.coolwarm(norm(np.array(Tdaily)))

    # =============================================================================
    # Ozone
    # =============================================================================
    Ozone_avg = Ozone.iloc[:,0][Ozone.iloc[:,1]==1].resample('1h').median().reindex(master_time_hr)
    # Create fake hour axis for plotting on top pf boxplot
    numHr=len(master_time_hr)
    hrl_xarray=np.linspace(0.5,50.5,numHr)

    # =============================================================================
    # Plot
    # =============================================================================
    
    # Boxplot properties
    boxprops = dict(linestyle='-', linewidth=2)
    flierprops = dict(marker='.', markerfacecolor='k', markersize=5,
                      linestyle='none')
    medianprops = dict(linestyle='-', linewidth=2)
    whiskerprops = dict(linestyle='-', linewidth=2)
    capprops = dict(linestyle='-', linewidth=2, color='#1f77b4')


    fig, axs = plt.subplots(3,1,figsize=(25,12),gridspec_kw={'height_ratios': [1,10,10]},
                            constrained_layout=True)
    
    bp_dict_I = IO3box.boxplot('IO3',by='dayyear',ax=axs[1], widths=0.8,
                                 boxprops=boxprops,flierprops=flierprops,medianprops=medianprops,
                                 whiskerprops=whiskerprops, capprops = capprops,
                                 return_type='both',patch_artist = True)
    
    bp_dict_UFP = UFPboxplot.boxplot('UFP_Conc',by='dayyear',ax=axs[2], widths=0.8,
                                 boxprops=boxprops,flierprops=flierprops,medianprops=medianprops,
                                 whiskerprops=whiskerprops, capprops = capprops,
                                 return_type='both',patch_artist = True)

    # =============================================================================
    # Set boxplot median and face colors
    # =============================================================================
    for row_key, (ax,row) in bp_dict_I.iteritems():
        ax.set_xlabel('')
        
        for box in row['medians']:
            box.set_color('k')
    
        for i,box in enumerate(row['boxes']):
            box.set_facecolor(Tcol[i])
    
    
    for row_key, (ax,row) in bp_dict_UFP.iteritems():
        ax.set_xlabel('')
        
        for box in row['medians']:
            box.set_color('k')
    
        for i,box in enumerate(row['boxes']):
            box.set_facecolor(Tcol[i])

    axs[1].set_yscale('log')
    axs[1].set_ylim((3e3,2e7))
    
    axs[2].set_yscale('log')
    axs[2].set_ylim((0.5,2e4))
    
    axs[1].grid(False)
    axs[2].grid(False)
    #Change format of date tick labels
    axs[1].set_xlim((0.5,50.5))
    axs[2].set_xlim((0.5,50.5))
    
    xticks=axs[1].get_xticks()
    axs[2].set_xticks(xticks[1::3])
    axs[1].set_xticks(xticks[1::3])
    axs[2].set_xticklabels(UFP.index.floor('D').unique()[1::3].astype(str))
    fig.autofmt_xdate()
    
    fig.suptitle('')
    axs[1].set_title('')
    axs[2].set_title('')    
        
    axs[2].set_ylabel(r'UFP (2.5-15nm) [cm$^{-3}$]')
    axs[1].set_ylabel(r'HIO$_3$ [molec. cm$^{-3}$]')

    # =============================================================================
    # Other timeseries
    # =============================================================================
    # Ozone
    xlim = axs[1].get_xlim()
    ax4=axs[1].twinx() 
    ax4.plot(hrl_xarray, Ozone_avg,lw=4,zorder=3,color='#02ab2e',alpha=0.7)
    ax4.set_xlim(xlim)
    ax4.set_ylabel(r'Ozone [ppb]')
    
    # Temperature
    xlim = axs[2].get_xlim()
    ax3=axs[2].twinx() 
    ax3.plot(trd_xarray,Twarm,lw=4,zorder=3,color='#feb308',alpha=0.6)
    ax3.plot(trd_xarray,Tfreeze,lw=4,zorder=3,color='#056eee',alpha=0.7)
    
    ax3.set_ylim((-19,4))
    ax3.set_xlim(xlim)
    ax3.set_ylabel(r'Air temperature [$^{\circ}$C]')
    
    # =============================================================================
    # Colorbar
    # =============================================================================
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap = plt.get_cmap('coolwarm')), cax=axs[0],orientation='horizontal')
    xticks = cbar.ax.get_xticks()
    cbar.ax.set_xticklabels(xticks)
    cbar.ax.invert_xaxis()
    cbar.ax.set_title(r'Air Temperature [$^{\circ}$C]', fontsize=26)
    
    # =============================================================================
    # Adjust subplot spacing
    # =============================================================================
    plt.tight_layout()
    plt.subplots_adjust( hspace=0.14)

    plt.savefig('SeasonalTrend_Temps_fin_O3.pdf')
    plt.savefig('SeasonalTrend_Temps_fin_O3.png',dpi=300)