# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:14:29 2020
Creation of a mask to separate clean from polluted data
The particle derivative signal and black carbon 
concentration are used as proxies.

@author: baccarini_a
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
import pandas as pd
from scipy import stats
import seaborn as sns
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FixedFormatter

#%%matplotlib parameters
matplotlib.rcParams['xtick.labelsize'] = 29
matplotlib.rcParams['ytick.labelsize'] = 29
matplotlib.rcParams['font.size'] = 33
#%% Load data
fpath=Path('D:/2018_MOCCHA/Analysis/hdfData/py3_hdf')

fname=fpath / 'PSM_clean.hdf'
PSM=pd.read_hdf(fname)
PSM.columns = ['PSM']

fname=fpath / 'UCPC_clean.hdf'
UCPC=pd.read_hdf(fname)
UCPC.columns = ['UCPC']

# UCPC processed
fname = fpath / 'UCPC_1min_Tcorr_Pollflag_03Jul20.hdf'
UCPC_proc = pd.read_hdf(fname)

# CIMS
fname = fpath / 'CIMS_10min_Pollflag_03Jul20.hdf'
CIMS = pd.read_hdf(fname)

# NAIS
fname = fpath / 'NAIS_Tcorr_Pollflag_03Jul20.hdf'
NAIS = pd.read_hdf(fname)

#load MAAP
fpath = Path('D:/2018_MOCCHA/Data')
fname = fpath / 'MAAP.csv'
MAAP=pd.read_csv(fname,index_col=0,header=None,parse_dates=True,squeeze=True)

def Pollution_mask(PSM, UCPC, MAAP, Integ_mask_clean, Integ_mask_dirty):
    '''
    Input: 
        PSM data with 1s time resolution
        UCPC data with 1s time resolution
        MAAP data with 1s time resolution
        Integ_mask_clean and Integ_mask_dirts periods to integrate mask
    '''
    #%% Process data
    MAAPres = MAAP.resample('1min').nearest() # Resample to regular 1minute grid 

    # =============================================================================
    # Calculate derivative
    # =============================================================================
    # consider only valid values
    PSM_temp = PSM[PSM>0].dropna()
    PSMder = np.abs(np.gradient(PSM_temp['PSM']))
    
    # Remove zeros (about 0.01% of the data)
    PSM_temp = PSM_temp[PSMder>0] 
    PSMder = PSMder[PSMder>0]
    
    # Repeat with UCPC
    UCPC_temp = UCPC[UCPC>0].dropna()
    UCPCder = np.abs(np.gradient(UCPC_temp['UCPC']))
    
    #remove the zeros 
    UCPC_temp=UCPC_temp[UCPCder>0]
    UCPCder=UCPCder[UCPCder>0]
    
    # =============================================================================
    # Create new dataframe 
    # =============================================================================
    PSM_df = pd.DataFrame({'PSM conc': PSM_temp['PSM'],
                           'PSM der': PSMder}).resample('1min').mean()
    UCPC_df = pd.DataFrame({'UCPC conc': UCPC_temp['UCPC'],
                           'UCPC der': UCPCder}).resample('1min').mean()

    # =============================================================================
    # Create mask    
    # =============================================================================
    # Parameters
    a = 0.2
    b = 0.53
    PSM_treshold = 1.12
    UCPC_treshold = 2.8
    
    # Normalization
    NormDiffPSM = np.log10(PSM_df['PSM der']/(a*PSM_df['PSM conc']**b))
    NormDiffUCPC = np.log10(UCPC_df['UCPC der']/(a*UCPC_df['UCPC conc']**b))
    # Mask
    maskPSM = NormDiffPSM < np.log10(PSM_treshold)
    maskUCPC = NormDiffUCPC < np.log10(UCPC_treshold)
    
    # Reindex
    #reindexing
    master_time = pd.DatetimeIndex(start='2018-08-01 14:00:00', end='2018-09-19 21:00:00', freq='1min')

    maskPSM = maskPSM.reindex(master_time) 
    maskPSM = maskPSM.fillna(False)

    maskUCPC = maskUCPC.reindex(master_time) 
    maskUCPC = maskUCPC.fillna(False)
    
    # Additional criteria
    # Boundary adjustment
    maskPSM[-1] = True
    maskUCPC[0] = True
    
    maskPSM02 = maskPSM.copy()
    maskUCPC02 = maskUCPC.copy()

    # Remove points before/after each contaminated point
    maskPSM02[np.where(maskPSM==False)[0]-1]=False
    maskPSM02[np.where(maskPSM==False)[0]+1]=False
    
    maskUCPC02[np.where(maskUCPC==False)[0]-1]=False
    maskUCPC02[np.where(maskUCPC==False)[0]+1]=False

    #Set boundaries back to their original value
    maskPSM[-1] = False
    maskUCPC[0] = False
    maskPSM02[-1] = False
    maskUCPC02[0] = False
    
    # Filtering of data in time period with a lot of pollution (window of 30 minutes)
    # Remove points within a 30-min window when more than 10 consecutive data are polluted
    
    maskPSM03 = maskPSM02.copy()
    maskUCPC03 = maskUCPC02.copy()

    for j in range(15,len(maskPSM03)-15):
        if maskPSM02[j-15:j+16].sum()<10:
            maskPSM03[j] = False

    for j in range(15,len(maskUCPC03)-15):
        if maskUCPC02[j-15:j+16].sum()<10:
            maskUCPC03[j] = False
        
    # =============================================================================
    # Integrate mask 
    # =============================================================================
    # This is based on visual inspection of the data using external proxies like
    # CO2 and wind direction
    
    maskPSM04=maskPSM03.copy()
    maskUCPC04=maskUCPC03.copy()

    for j in range(len(cleanmask.index)):
        maskPSM04[Integ_mask_clean['Start'][j]:Integ_mask_clean['End'][j]]= True
        maskUCPC04[Integ_mask_clean['Start'][j]:Integ_mask_clean['End'][j]]= True
    
    maskPSM05=maskPSM04.copy()
    maskUCPC05=maskUCPC04.copy()
    
    for j in range(len(dirtymask.index)):   
        maskPSM05[Integ_mask_dirty['Start'][j]:Integ_mask_dirty['End'][j]]= False
        maskUCPC05[Integ_mask_dirty['Start'][j]:Integ_mask_dirty['End'][j]]= False
    
    # =============================================================================
    # Merge UCPC and PSM masks
    # =============================================================================
    # The PSM provide the basic mask and the UCPC is used for integration when 
    # no PSM data are available.
    
    #I Drop NaNs
    idxPSM = PSM1min.dropna().index
    idxUCPC = UCPC1min.dropna().index
    
    # idx2 correspond to periods were only UCPC data are available
    idx2 = idxUCPC[~idxUCPC.isin(idxPSM)]
    
    #create the combined mask
    maskcombo = maskPSM05.copy()
    maskcombo[idx2] = maskUCPC05[idx2]
    
    # adjust last missing values
    idxmerge = idxPSM.append(idx2) #this index correspond to all available measurements (PSM+UCPC)
    idxmerge = idxmerge.sort_values()
    
    idxmissing = maskcombo.index[~maskcombo.index.isin(idxmerge)]
    
    # Create lower resolution mask
    mask2min = maskcombo.resample('2min').min()
    mask5min = maskcombo.resample('5min').min()
    mask10min = maskcombo.resample('10min').min()

    #%% Plot
    # =============================================================================
    # Figure 1, original data
    # =============================================================================
    fig, axs = plt.subplots(figsize=(20,12),ncols=2,constrained_layout=True,
                            gridspec_kw={'width_ratios': [20,.5]})
    
    pcm = axs[0].scatter(PSM_df['PSM conc'],PSM_df['PSM der'],marker='.',
                         c=MAAPres.loc[PSM_df.index],norm=LogNorm(vmin=0.001,vmax=10))
    
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].grid()
    axs[0].set_xlabel(r'Particle number concentration [cm$^{-3}$]')
    axs[0].set_ylabel(r'|Concentration derivative [cm$^{-3}$ s$^{-1}$]|')
    
    cbar=fig.colorbar(pcm,cax=axs[1], extend='neither', orientation='vertical')
    cbar.set_label(r'Black carbon [$\mu$g m$^{-3}$]',labelpad=-2)
    
    # =============================================================================
    # Figure 2, normalized data
    # =============================================================================
    a = 0.2
    b = 0.53
    
    # Hist fit
    NormDiffPSM=np.log10(PSM_df['PSM der']/(a*PSM_df['PSM conc']**b))
    x=np.linspace(NormDiffPSM.min(),NormDiffPSM.max(),500)
    mu, std = stats.norm.fit(NormDiffPSM[NormDiffPSM<0.05])
    
    # Plot
    fig, axs = plt.subplots(figsize=(20,12),ncols=2,nrows=2,constrained_layout=True,
                            gridspec_kw={'width_ratios': [20,10],'height_ratios': [0.5,20]})
    
    pcm = axs[1,0].scatter(PSM_df['PSM conc'],PSM_df['PSM der']/(a*PSM_df['PSM conc']**b),
                         marker='.',c=MAAPres.loc[PSM_df.index],norm=LogNorm(vmin=0.001,vmax=10))
    
    axs[1,0].set_yscale('log')
    axs[1,0].set_xscale('log')
    axs[1,0].grid()
    axs[1,0].set_xlabel(r'Particle number concentration [cm$^{-3}$]')
    axs[1,0].set_ylabel(r'|Normalized concentration derivative|')
    
    cbar=fig.colorbar(pcm,cax=axs[0,0], extend='neither', orientation='horizontal')
    cbar.set_label(r'Black carbon [$\mu$g m$^{-3}$]')
    axs[0,0].xaxis.set_label_position('top')
    axs[0,0].xaxis.set_ticks_position('top')
    
    axs[1,1].hist(NormDiffPSM,orientation='horizontal',bins=200,density=True)
    p = stats.norm.pdf(x, mu, std)
    axs[1,1].plot(p,x, '#feb308', linewidth=3,label='Log normal fit')
    axs[1,1].legend()
    
    ylim = axs[1,0].get_ylim()
    axs[1,1].set_ylim(np.log10(ylim))
    axs[1,1].set_yticks([])
    
    axs[0,1].axis('off')

    return(maskcombo, mask2min, mask5min, mask10min, idxmissing)


