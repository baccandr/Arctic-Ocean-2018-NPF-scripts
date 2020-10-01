# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:46:17 2019

@author: baccarini_a
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker


#%% Plotting functions

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def format_axes(axs):
    for j in axs:
        j.set_xscale('log')
        j.set_yscale('log')
        j.set_ylim((0.1,1e4))


#%%matplotlib parameters
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['font.size'] = 22

#%% Main

def Figure4(MergedPSD, DMPS, DMPS_CVI, Res_CPC, Visibility):
    '''
    Input:
    MergedPSD data: https://doi.org/10.17043/ao2018-aerosol-merged-psd
    DMPS: https://doi.org/10.17043/ao2018-aerosol-dmps
    DMPS_CVI and Res_CPC: https://doi.org/10.17043/baccarini-2020-new-particle-formation
    Visibility can be obtained from: https://doi.org/10.17043/ao2018-misu-weather-2
    '''
    
    # Prepare PSD data
    cols = MergedPSD.columns[:-1]
    DpPSD=np.array([s.split('Diameter[nm] : ',1)[1] for s in cols], dtype=float)
    PSD=MergedPSD.iloc[:,:-1].copy() 
    PSD.columns=DpPSD
    PSD_clean = PSD[MergedPSD['Pollution flag']==1]
    
    # Prepare DMPS
    DMPS = DMPS.iloc[:,1:]
    cols = DMPS.columns
    DpDMPS = np.array([s.split(' ')[2] for s in cols], dtype=float)*1e9
    DMPS.columns = DpDMPS
    # Set index to start time
    DMPS.index = DMPS.index - pd.Timedelta('4.5min')
    
    # Prepare DMPS_CVI
    DMPS_CVI = DMPS_CVI.iloc[:,3:] # first 3 Dp bins not trustworthy
    cols = DMPS_CVI.columns
    DpDMPS_CVI = np.array([s.split('Diameter[nm] : ',1)[1] for s in cols], dtype=float)
    DMPS_CVI.columns = DpDMPS_CVI
    
    # Select data
    PSDsel = PSD_clean['2018-09-06']
    VisibilitySel = Visibility['2018-09-06']
    DMPSsel = DMPS['2018-09-06']
    
    # Integrate particle concentration
    # Add Particles
    AccMode = DMPSsel.apply(lambda x: np.trapz(x[DpDMPS>70],np.log10(DpDMPS[DpDMPS>70])),axis=1)
    AitMode = DMPSsel.apply(lambda x: np.trapz(x[(DpDMPS<70)&(DpDMPS>38)],
                                                np.log10(DpDMPS[(DpDMPS<70)&(DpDMPS>38)])),axis=1)
    
    # Calculate statistics of Res_CPC
    CPC_CVI_mean = Res_CPC.resample('10min').median()
    CPC_CVI_q25 = Res_CPC.resample('10min').quantile(0.25)
    CPC_CVI_q75 = Res_CPC.resample('10min').quantile(0.75)
    
    # =============================================================================
    # identify Fog periods
    # =============================================================================
    
    VisibilitySel=Visibility['2018-09-05 12:00':'2018-09-06 23:00']
    Fog=VisibilitySel[VisibilitySel<1000]
    delta_t=Fog.reset_index().diff(-1).DateTime/ np.timedelta64(1, 'm')
    
    Fog_idxend=pd.Series(np.append(Fog.iloc[delta_t[delta_t<-5].index].index.values[1:],
                                   np.datetime64(Fog.index[-1])))+pd.Timedelta('1min')
    
    Fog_idxstart=pd.Series(Fog.iloc[delta_t[delta_t<-5].index+1].index)-pd.Timedelta('1min')


    # =============================================================================
    # Prepare individual PSDS
    # =============================================================================
    PSD_ev0=PSD['2018-09-06 03:30':'2018-09-06 04:20'].median()
    CVI_avg_ev0=DMPS_CVI['2018-09-06 03:30':'2018-09-06 04:20'].median()
    
    PSD_ev1=PSD['2018-09-06 07:50':'2018-09-06 09:15'].median()
    CVI_avg_ev1=DMPS_CVI['2018-09-06 07:50':'2018-09-06 09:15'].median()
    
    PSD_ev2=PSD['2018-09-06 09:20':'2018-09-06 10:25'].median()
    CVI_avg_ev2=DMPS_CVI['2018-09-06 09:20':'2018-09-06 10:25'].median()
    
    PSD_ev3=PSD['2018-09-06 16:00':'2018-09-06 17:25'].median()
    CVI_avg_ev3=DMPS_CVI['2018-09-06 16:00':'2018-09-06 17:25'].median()
    
    #q25
    PSD_ev0q25=PSD['2018-09-06 03:30':'2018-09-06 04:20'].quantile(0.25)
    CVI_avg_ev0q25=DMPS_CVI['2018-09-06 03:30':'2018-09-06 04:20'].quantile(0.25)
    
    PSD_ev1q25=PSD['2018-09-06 07:50':'2018-09-06 09:15'].quantile(0.25)
    CVI_avg_ev1q25=DMPS_CVI['2018-09-06 07:50':'2018-09-06 09:15'].quantile(0.25)
    
    PSD_ev2q25=PSD['2018-09-06 09:20':'2018-09-06 10:25'].quantile(0.25)
    CVI_avg_ev2q25=DMPS_CVI['2018-09-06 09:20':'2018-09-06 10:25'].quantile(0.25)
    
    PSD_ev3q25=PSD['2018-09-06 16:00':'2018-09-06 17:25'].quantile(0.25)
    CVI_avg_ev3q25=DMPS_CVI['2018-09-06 16:00':'2018-09-06 17:25'].quantile(0.25)
    
    #q75
    PSD_ev0q75=PSD['2018-09-06 03:30':'2018-09-06 04:20'].quantile(0.75)
    CVI_avg_ev0q75=DMPS_CVI['2018-09-06 03:30':'2018-09-06 04:20'].quantile(0.75)
    
    PSD_ev1q75=PSD['2018-09-06 07:50':'2018-09-06 09:15'].quantile(0.75)
    CVI_avg_ev1q75=DMPS_CVI['2018-09-06 07:50':'2018-09-06 09:15'].quantile(0.75)
    
    PSD_ev2q75=PSD['2018-09-06 09:20':'2018-09-06 10:25'].quantile(0.75)
    CVI_avg_ev2q75=DMPS_CVI['2018-09-06 09:20':'2018-09-06 10:25'].quantile(0.75)
    
    PSD_ev3q75=PSD['2018-09-06 16:00':'2018-09-06 17:25'].quantile(0.75)
    CVI_avg_ev3q75=DMPS_CVI['2018-09-06 16:00':'2018-09-06 17:25'].quantile(0.75)



    # =============================================================================
    # Figure
    # =============================================================================
    fig = plt.figure(figsize=(18, 12))
    plt.subplots_adjust(left=0.06, right=0.9, bottom=0.065) 
    plt.subplots_adjust(wspace=0.02, hspace=0.2)
    
    # Generate axis
    ax0 = fig.add_subplot(2,4,5)
    ax1 = fig.add_subplot(2,4,6)
    ax2 = fig.add_subplot(2,4,7)
    ax3 = fig.add_subplot(2,4,8)
    ax4 = fig.add_subplot(2,1,1)
    ax5 = fig.add_subplot(13,1,1)
    
    #adjust position of PSD and colro bar
    ax4.set_position(matplotlib.transforms.Bbox([[0.06, 0.48], [0.9, 0.86]]))
    ax4pos=ax4.get_position()
    ax5.set_position(matplotlib.transforms.Bbox([[0.06, 0.9], [0.9, 0.915]]))
    
    # create axis for common label
    ax_lab = fig.add_subplot(2,1,2,frameon=False)
    ax_lab.tick_params(left=False, right=False, labelleft=False, labelright=False,
                    bottom=False, labelbottom=False)
    ax_lab.set_xlabel('Diameter [nm]',labelpad=25)
    ax_lab.grid(False)
    
    format_axes([ax0,ax1,ax2,ax3])
    
    
    # Axis 0
    l1,=ax0.plot(DpPSD,PSD_ev0,lw=3,label='NPF event')
    ax0.fill_between(DpPSD,PSD_ev0q25,PSD_ev0q75,color=l1.get_color(),alpha=0.5)
    
    l2,=ax0.plot(DpDMPS_CVI,CVI_avg_ev0,color='#7bb274',lw=3,label='Residuals')
    ax0.fill_between(DpDMPS_CVI,CVI_avg_ev0q25,CVI_avg_ev0q75,color='#7bb274',alpha=0.5)
    
    ax0.set_ylabel(r'dN/dlogDp [cm$^{-3}$]',labelpad=-5)
    ax0.set_title('03:30 - 04:30',fontsize=16)
    
    ax0.legend((l1,l2),('Total PSD','Residuals'),loc='upper right')
    
    # Axis 1
    ax1.plot(DpPSD,PSD_ev1,lw=3,label='NPF event')
    ax1.fill_between(DpPSD,PSD_ev1q25,PSD_ev1q75,color=l1.get_color(),alpha=0.5)
    
    ax1.plot(DpDMPS_CVI,CVI_avg_ev1,color='#7bb274',lw=3,label='Residuals')
    ax1.fill_between(DpDMPS_CVI,CVI_avg_ev1q25,CVI_avg_ev1q75,color='#7bb274',alpha=0.5)
    
    ax1.set_title('07:50 - 09:15',fontsize=16)
    ax1.get_yaxis().set_ticklabels([])
    
    
    # Axis 2
    ax2.plot(DpPSD,PSD_ev2,lw=3,label='NPF event')
    ax2.fill_between(DpPSD,PSD_ev2q25,PSD_ev2q75,color=l1.get_color(),alpha=0.5)
    
    ax2.plot(DpDMPS_CVI,CVI_avg_ev2,color='#7bb274',lw=3,label='Residuals')
    ax2.fill_between(DpDMPS_CVI,CVI_avg_ev2q25,CVI_avg_ev2q75,color='#7bb274',alpha=0.5)
    
    ax2.set_title('09:25 - 10:30',fontsize=16)
    ax2.get_yaxis().set_ticklabels([])
    
    # Axis 3
    ax3.plot(DpPSD,PSD_ev3,lw=3,label='NPF event')
    ax3.fill_between(DpPSD,PSD_ev3q25,PSD_ev3q75,color=l1.get_color(),alpha=0.5)
    
    ax3.plot(DpDMPS_CVI,CVI_avg_ev3,color='#7bb274',lw=3,label='Residuals')
    ax3.fill_between(DpDMPS_CVI,CVI_avg_ev3q25,CVI_avg_ev3q75,color='#7bb274',alpha=0.5)
    
    ax3.set_title('16:00 - 17:30',fontsize=16)
    ax3.get_yaxis().set_ticklabels([])
    
    # PSD plot
    PSDplot=PSDsel
    matrix=np.array(PSDplot)
    matrix[np.where(matrix<=0)]=0.01 #replace the zeros for graphical purposes (cannot handle log scale)
    size=np.shape(matrix)
    Tmatrix=np.zeros((size[1],size[0]))
    
    for k in range(size[1]):
        Tmatrix[k,:]=matrix[:,k]
    
    pcm=ax4.pcolormesh(PSDplot.index,DpPSD,Tmatrix,norm=LogNorm(vmin=1,vmax=10000),cmap='viridis')
    
    ax4.set_yscale('log')
    ax4.set_ylabel('Diameter [nm]')
    ax4.set_xlim(pd.to_datetime(['2018-09-06 03:00','2018-09-06 21:00']))
    ax4.set_yticks([3,5,7,20,30,50,70,200,300,500,700], minor=True)
    ax4.yaxis.set_minor_formatter(ticker.FixedFormatter(['3','5','7','2','3','5','7','2','3','5','7']))
    ax4.tick_params(axis='y', which='minor', labelsize=16)
    ax4.set_yticks([10,100])
    ax4.yaxis.set_major_formatter(ticker.FixedFormatter(['10','100']))
    ax4.xaxis.tick_top()
    ax4.tick_params(axis='x', which='major', pad=-3)
    
    cbar=fig.colorbar(pcm,cax=ax5, extend='both', orientation='horizontal')
    cbar.ax.set_title('dN/dlogDp [cm$^{-3}$]',fontsize=22)
    cbar.ax.xaxis.tick_top()
    
    ylim=ax4.get_ylim()
    for j in range(len(Fog_idxstart)):
        ax4.fill_between((Fog_idxstart[j],Fog_idxend[j]),
                            (ylim[0],ylim[0]),
                            (ylim[1],ylim[1]),
                            color='#d8dcd6',alpha=0.6)
    
    
    # # Add Fog
    ax6=ax4.twinx()
    fog=shipdata.Visibility['2018-09-06'].resample('1min').mean()/1000
    ax6.plot(fog,color='k',lw=3.5)
    l1,=ax6.plot(fog,color='grey',lw=3)
    #ax6.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax6.set_ylabel('Visibility [km]')
    ax6.set_position(ax4pos)
    ax6.set_ylim((-0.1,5))
    
    
    # =============================================================================
    # Plot CVI CPC
    # =============================================================================
    
    
    ax7=ax4.twinx()
    ax7.fill_between(CPC_CVI_q25.index,CPC_CVI_q25.iloc[:,0],CPC_CVI_q75.iloc[:,0],color='#d9544d',alpha=0.5)
    ax7.plot(CPC_CVI_mean,'-',color='k',lw=3)
    l4,=ax7.plot(CPC_CVI_mean,'-',color='#d9544d',lw=2.5)
    
    
    l2,=ax7.plot(AccMode,'o',color='#feb308',markeredgecolor='k',markersize=8)
    l3,=ax7.plot(AitMode,'o',color='#39ad48',markeredgecolor='k',markersize=8)
    
    ax7.legend((l1,l2,l3,l4),('Visibility',r'DMPS [70-900 nm]',r'DMPS [35-70 nm]','Residuals'),
               ncol=4,loc='upper left',handletextpad=0.1)
    
    #ax7.set_yscale('log')
    ax7.set_ylim((-0.5,20))
    ax7.set_ylabel(r'Particle conc. [cm$^{-3}$]')
    ax7.set_position(ax4pos)
    ax7.spines["right"].set_position(("axes", 1.055))
    make_patch_spines_invisible(ax7)
    ax7.spines["right"].set_visible(True)
    
