# -*- coding: utf-8 -*-
"""
Script used to:
    - Read Lagranto backtrajectory files
    - Produce structured pandas dataframes
    - Isolate trajecteories in the Boundary Layer
    - Some simple plotting

@author: baccarini_a
"""

import numpy as np
import pandas as pd
from pathlib import Path
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

    
def Unpack_traj(fileN): 
    """ Function to unpack backtrajectory file given the fullpath of the file.
    it returns the dataframe with the trajectory and a datatime object
    correspondinf to the release time of the trajectories
    The only difference from the func above is the string length to determine the datetimeindex"""
    strpath = str(fileN)
    Trjtime = pd.to_datetime(strpath[32:36]+'-'+strpath[36:38]+'-'+strpath[38:40]+' '+strpath[41:43]+':00')
    
    Trj = pd.read_csv(fileN, delim_whitespace=True, skiprows=(0,3),skip_blank_lines=True,engine='python')
    
    #Removing nan
    Trj.replace(-999.99,np.nan,inplace=True)
    Trj.replace(-1000,np.nan,inplace=True)
    #Trj.dropna(inplace=True)
    
    #Creating a multiindex to account for the different trajectories
    thrddim = len(np.where(Trj['time']==0)[0]) #number of trajectories
    
    #here I simply create a sequential index (each trajectory has the same integer number)
    indexcol = np.zeros(len(Trj.index))
    for j in range(thrddim):
        indexcol[(81*j):((j+1)*81)]=j
    
    newidx = [indexcol,Trj['time']] #create new double index that will be the new multindex df
    Trj02 = Trj.iloc[:,1:] #Remove time column because it will be used as index
    Trj02.index = newidx #create this new multindex
    
    return(Trj02,Trjtime)
#%%matplotlib parameters
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['font.size'] = 18

plt.close('all')

#%% Load data
# Unpack all the trajectories and create a 3 level multindexed dataframe,
# the first index is associated to the release time, the second 
# to the different trajectorie and the third to the airparcel time (negative)

fpath = Path('D:/2018_MOCCHA/Trajectories/')
files = fpath.glob('lsl*')

Trjtot = []
Trj_index = pd.DatetimeIndex([])

for k, j in enumerate(files):
    Trjtemp,Trjtemp_name = Unpack_traj(j)
    
    Trj_index = Trj_index.append(pd.DatetimeIndex([Trjtemp_name]))
    Trjtot.append(Trjtemp)
    print(j)

# concatenate the trajectories in a multindex
Trjtot = pd.concat(Trjtot, keys=Trj_index)

# assign name to multindex
Trjtot.index.names=['Release time','Trajectory','Air parcel time']

#%% Get only BL trajectories

Trj_copy = Trjtot.copy()
groupby_date_range = Trj_copy.groupby(['Release time', 'Trajectory'])
Trj_copy["cumcount"] = groupby_date_range.cumcount()

first_col1_lt_col2 = defaultdict(lambda: len(Trj_copy), 
                                 Trj_copy[Trj_copy['p'] < Trj_copy['BLHP']]\
                                     .groupby(['Release time', 'Trajectory'])\
                                         ["cumcount"].min().to_dict())

TrjBL = Trj_copy[Trj_copy.apply(lambda row: row["cumcount"] < first_col1_lt_col2\
                                [row.name[:2]], axis=1)].drop(columns="cumcount")

#%%Save data
fpath='E:\\2018_MOCCHA\\Analysis\\hdfData\\'
Trjtot.to_hdf(fpath+'MOCCHA_trajectories_all.hdf','w')
TrjBL.to_hdf(fpath+'MOCCHA_trajectories_BL.hdf','w')

#%% Summer and Fall plot
idx = pd.IndexSlice
TrjBL5d = TrjBL.loc[idx[TrjBL.index.get_level_values(2)>-5*24]]

index1 = pd.date_range(start='2018-08-14 00:00',end='2018-08-27 00:00',freq='1H')
index2 = pd.date_range(start='2018-08-27 00:00',end='2018-09-15 00:00',freq='1H')

Trj_sub1 = TrjBL5d.loc[index1]
Trj_sub2 = TrjBL5d.loc[index2]


aMBL1 = Trj_sub1.groupby(['lat','lon']).size()
aMBL2 = Trj_sub2.groupby(['lat','lon']).size()

lont1 = aMBL1.index.get_level_values('lon')
latt1 = aMBL1.index.get_level_values('lat')

lont2 = aMBL2.index.get_level_values('lon')
latt2 = aMBL2.index.get_level_values('lat')


fig = plt.figure(figsize=(24,12))
ax = plt.subplot(121,projection=ccrs.NorthPolarStereo())
ax.set_extent([0, 359, 60, 90], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.gridlines()

ax.plot(lont1,latt1,marker='.',markersize=1,alpha=0.08,transform=ccrs.PlateCarree(),label='All Trajectories',linestyle="None")
ax.set_title('Summer Period')

ax2 = plt.subplot(122,projection=ccrs.NorthPolarStereo())
ax2.set_extent([0, 359, 60, 90], crs=ccrs.PlateCarree())
ax2.coastlines(resolution='50m')
ax2.gridlines()

ax2.plot(lont2,latt2,marker='.',markersize=1,alpha=0.08,transform=ccrs.PlateCarree(),label='MBL trajectories',linestyle="None")
ax2.set_title('Fall Period')
plt.tight_layout()
