# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:27:48 2018

@author: baccarini_a
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import matplotlib


#%% matplotlib parameters
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['font.size'] = 22

#%% Functions
def Read_navigation(fpath):
    navig = pd.read_csv(fpath)
    navig.index = pd.to_datetime('2018-1-1') + pd.to_timedelta(navig.doy, unit='D')
    navig_rs = navig.resample('1min').mean()
    
    return(navig_rs[['lon','lat']])

#%% Main
def PlotMap(GPStrackfpath, NPFeventfpath, Seaicefpath):
    
    # Read gps track
    gps_data = Read_navigation(GPStrackfpath)
    gps_rs = gps_data.resample('1h').mean()
    
    gpsdrift = gps_rs[(gps_rs.index>'2018-08-13')&(gps_rs.index<'2018-09-14')] # Select only drift period


    # Loaf NPF periods
    NPFlist = pd.read_csv(NPFeventfpath)

    NPF_start = pd.to_datetime(NPFlist.Date.astype('str')+' '+NPFlist['Start time'].astype('str'))
    NPF_end = pd.to_datetime(NPFlist.Date.astype('str')+' '+NPFlist['End time'].astype('str'))
    
    NPF_mean = pd.to_datetime((NPF_start.values.astype(np.int64)+NPF_end.values.astype(np.int64))/2).floor('h')
    gps_NPF = gps_rs.loc[NPF_mean]
    
    # Seaice data
    #based on https://ocefpaf.github.io/python4oceanographers/blog/2015/04/20/arctic_sea_ice_concentration/
    with open(Seaicefpath, 'rb') as fr:
        hdr = fr.read(300)
        ice = np.fromfile(fr, dtype=np.uint8)
    
    ice = ice.reshape(448, 304)
    ice = ice / 2.50
    ice = np.ma.masked_greater(ice, 100.0)
    ice = np.ma.masked_equal(ice, 0.0)
    
    dx = dy = 25000
    
    x = np.arange(-3850000, +3750000, +dx)
    y = np.arange(+5850000, -5350000, -dy)
    
    # =============================================================================
    # Figure    
    # =============================================================================
    
    fig = plt.figure(figsize=(14,12))
    ax = plt.subplot(111, projection = ccrs.NorthPolarStereo())
    ax.set_extent([0, 359, 76, 90], crs = ccrs.PlateCarree())
    
    # Set map features
    land = cartopy.feature.NaturalEarthFeature(category='physical',name='land',scale='50m')   
    ax.add_feature(cartopy.feature.OCEAN, zorder = 0)
    ax.add_feature(land,zorder = 0, edgecolor = 'black', facecolor = cartopy.feature.COLORS['land'])
    ax.coastlines(resolution='50m', linewidth=0.5)
    
    # Plot GPS track and NPF events
    ax.plot(gps_rs.iloc[:,1], gps_rs.iloc[:,0], sns.xkcd_rgb["amber"], lw=4,
            transform=ccrs.PlateCarree(), label='AO18 Track')
    ax.plot(gpsdrift.iloc[:,1], gpsdrift.iloc[:,0], sns.xkcd_rgb["pale red"], lw=3,
            transform=ccrs.PlateCarree(), label='Ice Drift')
    ax.plot(gps_NPF.iloc[:,1],gps_NPF.iloc[:,0],'o',color=sns.xkcd_rgb["leaf green"],
            markersize=8,markeredgecolor='k',transform=ccrs.PlateCarree(),label='NPF events')
    
    # Seaice plot
    kw = dict(central_latitude=90, central_longitude=-45, true_scale_latitude=70)
    cs=ax.pcolormesh(x, y, ice, cmap=plt.cm.Blues,
                       transform=ccrs.Stereographic(**kw))
    
    ax.legend(fontsize=25)
    
    # Plot parallels
    ax.text(-5, 78, u'78°', transform=ccrs.PlateCarree(), fontsize=22)
    ax.text(-6, 80, u'80°', transform=ccrs.PlateCarree(), fontsize=22)
    ax.text(-8, 82, u'82°', transform=ccrs.PlateCarree(), fontsize=22)
    ax.text(-10, 84, u'84°', transform=ccrs.PlateCarree(), fontsize=22)
    ax.text(-15, 86, u'86°', transform=ccrs.PlateCarree(), fontsize=22)
    ax.text(-30, 88, u'88°', transform=ccrs.PlateCarree(), fontsize=22)
    
    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = False,
                      linewidth = 2, color = 'gray', alpha = 0.5, linestyle = '--')
    gl.ylocator = mticker.FixedLocator(np.linspace(70, 90, 11, dtype=int))
    
    # Colorbar
    cbar_ax = fig.add_axes([0.755, 0.11, 0.03, 0.77])
    cbar = fig.colorbar(cs, cax=cbar_ax, extend='neither', orientation='vertical')  
    cbar.set_label('Sea ice concentration [%]')
    
    # Adjust Figure margins
    plt.subplots_adjust(hspace = 0.1)
    fig.subplots_adjust(left=0.06)
    fig.subplots_adjust(right=0.88)


