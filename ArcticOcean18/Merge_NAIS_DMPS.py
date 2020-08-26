# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:01:25 2020

@author: baccarini_a
"""


import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import splrep, splev, splint


#%% Main

def Merged_PSD(NAIS, DMPS):
    '''
    NAIS data can be find: doi:https://doi.org/10.17043/ao2018-aerosol-nais
    DMPS data can be find: doi:https://doi.org/10.17043/ao2018-aerosol-dmps
    '''
    
    # Prepare NAIS data
    cols = NAIS.columns[1:-1]
    DpNAIS=np.array([s.split('Diameter[nm] : ',1)[1] for s in cols], dtype=float)
    NAIS_PSD=NAIS.iloc[:,1:-1].copy() 
    NAIS_PSD.columns=NAIS_PSD.columns.astype(float)
    NAIS_PSD[NAIS_PSD<0]=np.nan
    NAIS_PSD[NAIS_PSD<0.01]=0

    # Prepare DMPS
    cols = DMPS.columns[:-1]
    DpDMPS=np.array([s.split('Diameter[nm] : ',1)[1] for s in cols], dtype=float)
    DMPS_PSD=DMPS.iloc[:,:-1].copy()

    # Average Nais over DMPS
    ref=NAIS_PSD.index
    i,j=np.where((ref[:,None]>DMPS.index.values)&(ref[:,None]<(DMPS.index+pd.Timedelta('9min')).values))
    dfres=pd.DataFrame(NAIS_PSD.iloc[i].reset_index(drop=True).values,index=DMPS.index[j])
    NAISrs = dfres.groupby(dfres.index).mean()
    NAISrs.columns=DpNAIS

    #Diameters
    DpTransition=DpNAIS[(DpNAIS>16)&(DpNAIS<40)] #for merged PSD
    DpmergedPSD=np.concatenate((DpNAIS[DpNAIS<40],DpDMPS[DpDMPS>40]))

    # create empty df
    Merged_PSD=pd.DataFrame(index=DMPS_PSD.index,columns=DpmergedPSD) 
    
    # Create linear weights for PSD merging
    WeightsNAIS=np.interp(DpTransition,[16,40],[1,0])
    WeightsDMPS=np.interp(DpTransition,[16,40],[0,1])
    WeightsArr=np.column_stack((WeightsNAIS,WeightsDMPS))


    for j in (DMPS_PSD.index):
    
        if j in NAISrs.index:
            try:
                fspline=splrep(DpDMPS,DMPS_PSD.loc[j],k=3)
            except:
                break
            DMPS_interp=splev(DpTransition,fspline)
            NAIS_match=np.array(NAISrs.loc[j][(DpNAIS>16)&(DpNAIS<40)]).astype(float) #NAIS subset for overlapping region
            MergedArr=np.column_stack((NAIS_match,DMPS_interp)) #prepare array for average
            #use masked array to remove nan from averaging
            MaskArr=np.ma.masked_array(MergedArr, np.isnan(MergedArr))
            MergedArr_avg=pd.Series(np.ma.average(MaskArr,axis=1,weights=WeightsArr).filled(np.nan),index=DpTransition)
    
            #concatenate data
            Merged_PSD.loc[j]=pd.concat([NAISrs.loc[j][DpNAIS<16],
                                             MergedArr_avg,DMPS_PSD.loc[j][DpDMPS>40]])
        else:
            # This is when there are no NAIS data
            # to keep diameter consistent I have to interpolate DMPS
            DMPS_interp=np.interp(DpNAIS[(DpNAIS>=16)&(DpNAIS<40)],DpDMPS,DMPS_PSD.loc[j])
            DMPS_interp=pd.Series(DMPS_interp,index=DpNAIS[(DpNAIS>=16)&(DpNAIS<40)])
            Merged_PSD.loc[j][DpmergedPSD>=16]=pd.concat([DMPS_interp,DMPS_PSD.loc[j][DpDMPS>40]])

    # Remove artificially negative values from spline interpolation
    Merged_PSD[Merged_PSD<0]=0
    
    # Adjust column name
    Merged_PSD.columns = DpmergedPSD.round(2)
    Merged_PSD.index.name = 'Datetime'
    
    # Append pollution flag
    Merged_PSD['Pollution mask'] = DMPS['Pollution mask']

    return(Merged_PSD)
