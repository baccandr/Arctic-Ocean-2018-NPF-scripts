# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 04:39:14 2019

@author: baccarini_a
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from pathlib import Path


#%%matplotlib parameters
matplotlib.rcParams['xtick.labelsize'] = 23
matplotlib.rcParams['ytick.labelsize'] = 23
matplotlib.rcParams['font.size'] = 27


#%% Main

def Figure3a_b_plot(NAIS, CIMS, GRdata, Visibility):
    '''
    Input:
    NAIS data can be obtained from: doi:https://doi.org/10.17043/ao2018-aerosol-nais
    CIMS data can be obtained from: doi:https://doi.org/10.17043/ao2018-aerosol-cims
    Growth rate data: https://doi.org/10.17043/baccarini-2020-new-particle-formation
    The relevant file is Growthrate_mode_fit_updated30Aug2020.csv
    Visibility can be obtained from: https://doi.org/10.17043/ao2018-misu-weather-2
    '''
    
    # Prepare NAIS data
    cols = NAIS.columns[1:-1]
    DpNAIS=np.array([s.split('Diameter[nm] : ',1)[1] for s in cols], dtype=float)
    NAIS_PSD=NAIS.iloc[:,1:-1].copy() 
    NAIS_PSD.columns=DpNAIS

    # Select data
    start_time = '2018-09-04 22:00'
    end_time = '2018-09-06 23:00'
    NAISsel = NAIS_PSD[start_time:end_time]
    CIMSsel = CIMS[start_time:end_time].iloc[:,:-1].resample('10min').asfreq()
    VisibilitySel = Visibility[start_time:end_time]
    GR_sel = GRdata['2018-09-05'].dropna(axis=1,how='all')
    
    # =============================================================================
    # identify Fog periods
    # =============================================================================
    Fog = VisibilitySel[VisibilitySel<2000]
    delta_t = Fog.reset_index().diff(-1).DateTime/ np.timedelta64(1, 'm')

    Fog_idxend = pd.Series(np.append(Fog.iloc[delta_t[delta_t<-20].index].index.values[1:],
                                   np.datetime64(Fog.index[-1])))+pd.Timedelta('5min')
    
    Fog_idxstart = pd.Series(Fog.iloc[delta_t[delta_t<-20].index+1].index)

    # =============================================================================
    # #PSD plot
    # =============================================================================
    
    fig, axs = plt.subplots(figsize=(20, 10), nrows=3,ncols=1,gridspec_kw={'height_ratios': [20,1.5,15]},constrained_layout=True)
    # Matrix rotation for plot
    matrix = np.array(NAISsel)
    matrix[np.where(matrix<=0)]=0.01 #replace the zeros for graphical purposes (cannot handle log scale)
    size = np.shape(matrix)
    Tmatrix = np.zeros((size[1],size[0]))
    
    for k in range(size[1]):
        Tmatrix[k,:] = matrix[:,k]
    
    pcm = axs[0].pcolormesh(NAISsel.index,DpNAIS,Tmatrix,norm=LogNorm(vmin=1,vmax=10000),cmap='viridis')
    
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Diameter [nm]')
    xlim = axs[0].get_xlim()
    
    axs[0].set_yticks([3,4,5,6,8,20,30],minor=True)
    axs[0].yaxis.set_minor_formatter(ticker.FixedFormatter(['3','4','5','6','8','20','30']))
    axs[0].tick_params(axis='y', which='minor', labelsize=18)
    axs[0].set_yticks([10])
    axs[0].yaxis.set_major_formatter(ticker.FixedFormatter(['10']))

    # Plot GR fitted modes
    l1, = axs[0].plot(GR_sel.iloc[:,0][:'2018-09-05 05:00'],'o',color=sns.xkcd_rgb["pale red"],markersize=8)
    
    l2, = axs[0].plot(GR_sel.iloc[:,0]['2018-09-05 05:00':],'o',color='#6a79f7',markersize=8)
    
    axs[0].legend((l1,l2),(r'Growth Rate = 0.35 nm/h',r'Growth Rate = 0.51 nm/h'),
                  loc='upper left')

    cbar = fig.colorbar(pcm,cax=axs[1], extend='both', orientation='horizontal')
    cbar.set_label('dN/dlogDp [cm$^{-3}$]')

    # Plot CIMS data on second plot
    
    axs[2].plot(CIMSsel.iloc[:,0],color='k',lw=4)
    axs[2].plot(CIMSsel.iloc[:,1],color='k',lw=4)
    l1, = axs[2].plot(CIMSsel.iloc[:,0],color="#e74c3c",lw=3)
    l2, = axs[2].plot(CIMSsel.iloc[:,1],color="#9b59b6",lw=3)
    axs[2].set_yscale('log')
    axs[2].set_ylim((1e5,1e7))

    axs[2].set_ylabel('Concentration\n'+r'[molecules cm$^{-3}$]')
    axs[2].set_xlim(xlim)
    
    # Visibility plot
    ax3 = axs[2].twinx()
    ax3.plot(VisibilitySel.resample('5min').mean()/1000,color='k',lw=3)
    l3, = ax3.plot(VisibilitySel.resample('5min').mean()/1000,color='#929591',lw=2.5)
    ax3.set_ylim((-0.1,5))
    ax3.legend((l2,l1,l3),(r'HIO$_3$',r'H$_2$SO$_4$','Visibility'),loc='upper right')
    ax3.set_ylabel('Visibility [km]')


    # =============================================================================
    # Highlight fog
    # =============================================================================
    ylim1 = axs[2].get_ylim()
    for j in range(len(Fog_idxstart)):
        axs[2].fill_between((Fog_idxstart[j],Fog_idxend[j]),
                            (ylim1[0],ylim1[0]),
                            (ylim1[1],ylim1[1]),
                            color='#d8dcd6',alpha=0.7)   
        
    axs[2].set_ylim(ylim1)
    
    # =============================================================================
    # Highlight pollution
    # =============================================================================
    ylim0 = axs[0].get_ylim()
    axs[0].fill_between((pd.to_datetime('2018-09-05 21:30'),
                         pd.to_datetime('2018-09-06 02:30')),
                         (ylim0[0],ylim0[0]),(ylim0[1],ylim0[1]),
                         facecolor="none", hatch="/", edgecolor="k", linewidth=0.5)
    
    axs[2].fill_between((pd.to_datetime('2018-09-05 21:30'),
                         pd.to_datetime('2018-09-06 02:30')),
                         (ylim1[0],ylim1[0]),(ylim1[1],ylim1[1]),
                         facecolor="none", hatch="/", edgecolor="k", linewidth=0.5)
    
    
    axs[0].set_ylim(ylim0)
    axs[2].set_ylim(ylim1)

