# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:18:33 2020

@author: baccarini_a
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

import seaborn as sns
from math import atan2,degrees

#%% Define deposition velocity
def friction_velocity(Wspeed,height):
    k=0.4
    Z0=0.001
    u_fric=k*Wspeed/(np.log(height/Z0))
    return(u_fric)

def Aerod_Res(Fvel,height):
    k=0.4
    Z0=0.001
    A_Res=1/(k*Fvel)*np.log(height/Z0)
    return(A_Res)

def Schmidt_num(T,P,RH):
    # Dry and wet air density
    rho_d,rho_w=Air_density(T,P,RH)
    Sc=Air_viscosity(T)/(rho_w*SA_DiffCoeff(T,P)*1e-4) #1e4 factor is conversion from cm2 to m2
    return(Sc)

def Qlam_Res(Sc,FricVel):
    Q_Res=5*Sc**(2/3)/FricVel
    return Q_Res

def Deposition_Vel(Wspeed,Temperature,Pressure,RH,height):
    # Temperature must be in kelvin
    
    FricVel=friction_velocity(Wspeed,height)
    ARes=Aerod_Res(FricVel,height)
    ScN=Schmidt_num(Temperature,Pressure,RH)
    QRes=Qlam_Res(ScN,FricVel)
    
    Dvel=1/(ARes+QRes)
    return Dvel
#%% Define plotting functions

#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)
        

#%%matplotlib parameters
mpl.rcParams['xtick.labelsize'] = 32
mpl.rcParams['ytick.labelsize'] = 32
mpl.rcParams['font.size'] = 37


#%% Iodic acid model
def Iodic_acid_model(CIMS, BLH, CondSink, Visib, Weather_data, Diagn_figures = True):
    
    # =============================================================================
    # Prepare data
    # =============================================================================
    master_time = pd.date_range(start='2018-08-01 14:00:00', end='2018-09-19 21:00:00', freq='10min')
    
    # =============================================================================
    # Meteorological Data
    # =============================================================================
    # Visibility
    Visibility=Visib.resample('10min').mean().reindex(master_time)
    # Wind speed
    WindSpeed=Weather['ws_tru'].resample('10min').mean().reindex(master_time)
    Temperature=Weather['temp'].resample('10min').mean().reindex(master_time)+273.15
    Pressure=Weather['press'].resample('10min').mean().reindex(master_time)
    
    # =============================================================================
    # Boundary layer height
    # =============================================================================
    # Create inversion base timeseries and resample
    invbase=BLH.sfmlbase.copy()
    invbase.index=invbase.index.round('h')
    
    # Modify BLH to account for wrong mixed layer height on 17 Sept. 
    # (personal communication with Jutta Vuellers)
    invbasers_10min=invbase.resample('10min').interpolate(method='time').dropna().reindex(master_time)
    invbasers_10min.name='BLH [m]'
    
    # =============================================================================
    # Prepare iodic acid data
    # =============================================================================
    HIO3=CIMS[CIMS['Pollution mask']==1]['IO3-'].reindex(master_time)
    # Keep data without fog
    HIO3_clear = HIO3[Visibility>4000].reindex(master_time)
    # Keep only fall data
    HIO3fall_clear=HIO3_clear['2018-08-27':].reindex(master_time)
    
    # =============================================================================
    # Prepare model data
    # =============================================================================
    DepoVel=Deposition_Vel(WindSpeed,Temperature,Pressure,90,10)
    Lsum=100*(DepoVel+invbasers_10min*CS)

    # =============================================================================
    # Automatic segmentation of the data
    # =============================================================================

    # resample data to fill nan up to 30min
    HIO3fall_rs = HIO3fall_clear.interpolate(method='time',limit=3,
                                             limit_direction='both',limit_area='inside')
    
    # Identify periods that are separated by more than 30min (this part is useless)
    idx = pd.Series(HIO3fall_rs.dropna().index)
    min_diff = idx.diff().dt.seconds/60
    idx_periodgen_start = idx[min_diff>30]
    idx_periodgen_end = idx.loc[idx_periodgen_start.index-1][1:]
    idx_periodgen_end[1357] = idx.iloc[-1]
    
    
    # Rolling mean
    HIO3_smooth = HIO3fall_rs.rolling(6,center=True,win_type='hamming',min_periods=1).mean()
    # remove data that are padded by the rolling function
    HIO3_smooth[HIO3fall_rs.isna()] = np.nan
    
    # =============================================================================
    # Derivative
    # =============================================================================
    HIO3_deriv = pd.Series(np.gradient(HIO3_smooth),index=HIO3_smooth.index)
    # Identify data where deriv/signal is below treshold
    '''
    I do the following:
        - calculate the ratio
        - take the index of values that are below the treshold
        - look at the time difference between consecutive index
        - keep all values where the difference is <= 30 min, in this
        way I'm keeping also the data with single spikes above the treshold
        - take groups of data where the number of points is larger than 90min
    '''
    Ratio = HIO3_deriv.abs()/HIO3_smooth
    Ratio_smooth = Ratio.rolling(6,center=True,win_type='hamming',min_periods=1).mean()
    Ratio_smooth[Ratio.isna()] = np.nan
    
    treshold = 0.05
    idx_sel = pd.Series(Ratio_smooth[Ratio_smooth<treshold].index)
    min_diff = idx_sel.diff().dt.seconds/60
    val_add = idx_sel.loc[min_diff[min_diff==20].index]-pd.Timedelta('10min')
    val_add2 = idx_sel.loc[min_diff[min_diff==30].index]-pd.Timedelta('10min')
    val_add3 = idx_sel.loc[min_diff[min_diff==30].index]-pd.Timedelta('20min')
    val_add_all = np.append(val_add,(val_add2,val_add3))
    idx_sel = pd.Series(np.append(idx_sel.values,val_add_all)).sort_values()
    
    
    # Create group of data and keep only those longer than 90 min
    min_diff2 = idx_sel.diff().dt.seconds/60
    idx_period_start = idx_sel[min_diff2>10]
    idx_period_end = idx_sel.loc[idx_period_start.index-1][1:]
    idx_period_end[1027] = idx_sel.iloc[-1]
    
    Duration = np.array((idx_period_end.reset_index(drop=True)-idx_period_start.reset_index(drop=True))/np.timedelta64(1, 'h'))
    idx_period_start_sel = idx_period_start[Duration>1.5]
    idx_period_end_sel = idx_period_end[Duration>1.5]
    
    ref=Ratio.index.values
    i,j=np.where((ref[:,None]>=idx_period_start_sel.values)&(ref[:,None]<=idx_period_end_sel.values))
    Ratio_sel = Ratio_smooth.iloc[i]
    
    HIO3_sel_ver1 = HIO3fall_rs.iloc[i]
    
    # =============================================================================
    # Checking Lsum on the same groups
    # =============================================================================
    
    # recalculating Lsum after interpolating selected data
    # all the nans outside the selceted periods will be removed, hence
    # the interpolation will only have a minor effect
    
    ref=CS.index.values
    i,j=np.where((ref[:,None]>=idx_period_start_sel.values)&(ref[:,None]<=idx_period_end_sel.values))
    CS_sel = CS.iloc[i].interpolate(method='time')
    DepoVel_sel = DepoVel.iloc[i].interpolate(method='time')
    invbasers_10min_sel = invbasers_10min.iloc[i].interpolate(method='time')
    
    Lsum_sel=100*(DepoVel_sel+invbasers_10min_sel*CS_sel)
    
    # =============================================================================
    # Applying the same procedure as for HIO3 to Lsum
    # =============================================================================
    Lsum_smooth = Lsum_sel.rolling(6,center=True,win_type='hamming',min_periods=1).mean()
    Lsum_smooth[Lsum_sel.isna()] = np.nan
    
    Lsum_deriv = pd.Series(np.gradient(Lsum_smooth),index=Lsum_smooth.index)
    Ratio_Lsum = Lsum_deriv.abs()/Lsum_smooth
    Ratio_Lsum_smooth = Ratio_Lsum.rolling(6,center=True,win_type='hamming',min_periods=1).mean()
    
    treshold = 0.05
    idx_sel = pd.Series(Ratio_Lsum_smooth[Ratio_Lsum_smooth<treshold].index)
    min_diff = idx_sel.diff().dt.seconds/60
    val_add = idx_sel.loc[min_diff[min_diff==20].index]-pd.Timedelta('10min')
    val_add2 = idx_sel.loc[min_diff[min_diff==30].index]-pd.Timedelta('10min')
    val_add3 = idx_sel.loc[min_diff[min_diff==30].index]-pd.Timedelta('20min')
    val_add_all = np.append(val_add,(val_add2,val_add3))
    idx_sel = pd.Series(np.append(idx_sel.values,val_add_all)).sort_values()
    
    # Create group of data and keep only those longer than 90 min
    min_diff2 = idx_sel.diff().dt.seconds/60
    idx_period_start_Lsum = idx_sel[min_diff2>10]
    idx_period_end_Lsum = idx_sel.loc[idx_period_start_Lsum.index-1][1:]
    idx_period_end_Lsum[448] = idx_sel.iloc[-1]
    
    Duration = np.array((idx_period_end_Lsum.reset_index(drop=True)-idx_period_start_Lsum.reset_index(drop=True))/np.timedelta64(1, 'h'))
    idx_period_start_Lsum_sel = idx_period_start_Lsum[Duration>1.5]
    idx_period_end_Lsum_sel = idx_period_end_Lsum[Duration>1.5]
    
    ref=Ratio_Lsum.index.values
    k,z=np.where((ref[:,None]>=idx_period_start_Lsum_sel.values)&(ref[:,None]<=idx_period_end_Lsum_sel.values))
    Ratio_Lsum_sel = Ratio_Lsum_smooth.iloc[k]
    
    # =============================================================================
    # Create dataframe with HIO3 values, Lsum and temperature
    # =============================================================================
    HIO3_aggregate = pd.DataFrame({'HIO3': HIO3_sel_ver1.iloc[k],
                                   'Lsum': Lsum_sel.iloc[k],
                                   'Tmean': Temperature.loc[Lsum_sel.index]})
    
    idx_range = pd.IntervalIndex.from_arrays(idx_period_start_Lsum_sel,idx_period_end_Lsum_sel,closed='both')
    #groupby df2 based on these intervals and calculate the mean
    HIO3_grouped = HIO3_aggregate.groupby(pd.cut(HIO3_aggregate.index,idx_range))
    
    # =============================================================================
    # Diagnostic Figures
    # =============================================================================
    if Diagn_figures:
        fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,
                                figsize=(22,13),constrained_layout=True)
        axs[0].plot(HIO3fall_rs,'-o')
        axs[0].plot(HIO3_smooth)
            
        axs[1].plot(Ratio_smooth,'-')
        axs[1].plot(Ratio_smooth[idx_sel],'o')
        axs[1].plot(Ratio_sel,'s')
        xlim= axs[1].get_xlim()
        axs[1].plot(xlim,(0.05,0.05),'--')
        
        
        fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,
                                figsize=(22,13),constrained_layout=True)
        axs[0].plot(Lsum,'-')
        axs[0].plot(Lsum_sel,'o')
        axs[0].plot(Lsum_sel.iloc[k],'*')
            
        axs[1].plot(Ratio_smooth,'-')
        axs[1].plot(Ratio_smooth[idx_sel],'o')
        axs[1].plot(Ratio_sel,'s')
        xlim= axs[1].get_xlim()
        axs[1].plot(xlim,(0.05,0.05),'--')
        
        for j in range(len(idx_period_start_Lsum_sel)):
            fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(18,12),constrained_layout=True)
            axs.plot(HIO3fall_rs[idx_period_start_Lsum_sel.iloc[j]:idx_period_end_Lsum_sel.iloc[j]],'-o')
            ax2 = axs.twinx()
            ax2.plot(Lsum_sel[idx_period_start_Lsum_sel.iloc[j]:idx_period_end_Lsum_sel.iloc[j]],'r-*')
        
        fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(18,12),constrained_layout=True)
        axs.plot(HIO3fall_rs[idx_period_start_Lsum_sel.iloc[4]:idx_period_end_Lsum_sel.iloc[4]],'-o')
        axs.plot(HIO3_smooth[idx_period_start_Lsum_sel.iloc[4]:idx_period_end_Lsum_sel.iloc[4]],'s')    
        
        for j in range(len(idx_period_start_sel)):
            fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(18,12),constrained_layout=True)
            axs.plot(HIO3fall_rs[idx_period_start_sel.iloc[j]:idx_period_end_sel.iloc[j]],'-o')
            ax2 = axs.twinx()
            ax2.plot(Lsum_sel[idx_period_start_sel.iloc[j]:idx_period_end_sel.iloc[j]],'r-*')


    # =============================================================================
    # # Equilibrium figure with the new data
    # =============================================================================
    sns.set(style="ticks",font_scale=3.3,palette=sns.color_palette("deep",n_colors=11),
            font = 'DejaVu Sans')
    
    # =============================================================================
    # Normal
    # =============================================================================
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(20,12),
                            constrained_layout=True)
    valid_markers = mpl.markers.MarkerStyle.filled_markers
    
    # =============================================================================
    # Standard plot
    # =============================================================================
    axs.plot((0,0),(5.6,7.2),'k--',lw=2)
    j = 0
    for name, group in HIO3_grouped:
        axs.plot(-np.log10(group['Lsum']),np.log10(group['HIO3']),
                        valid_markers[j],markersize=14,markeredgecolor='k')
        j+=1
    
    ParamVal=np.log10([1e6,2e6,3.5e6,6e6,1e7,1.5e7])#np.linspace(6.2,7.2,6)
    for j in ParamVal:
        l1,=axs.plot((-1,0.65),(-1+j,0.65+j),lw=2)
        if j<7:
            labelLine(l1,0.25,label='E=%.1e'%(10**j),fontsize=34)
        else:
           labelLine(l1,7.1-j,label='E=%.1e'%(10**j),fontsize=34) 
    
    axs.set_xlim((-0.97,0.6))
    axs.set_ylim((5.6,7.2))
    
    axs.set_xlabel(r'-log$_{10}$(v$_{d}$+$h$ $CS$)')
    axs.set_ylabel('log$_{10}$(HIO$_3$)')
    

    # =============================================================================
    # Distribution plot
    # =============================================================================
    df_g = HIO3_grouped.apply(lambda x: x).dropna()
    
    Evalues = np.log10(df_g['Lsum']*df_g['HIO3'])
    
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(14,12),
                                constrained_layout=True,gridspec_kw={"height_ratios": (.15, .85)})
    
    sns.boxplot(Evalues, ax=ax_box,linewidth=3.5)
    sns.distplot(Evalues,norm_hist=True, ax=ax_hist,
                 bins=15,kde_kws={'linewidth':4})
    
    ax_box.set(yticks=[])
    ax_box.set(xlabel=None)
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    ax_hist.set_xlabel(r'log$_{10}$(E)')