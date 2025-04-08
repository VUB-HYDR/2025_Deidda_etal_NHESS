# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:01:55 2023

@author: Cristina
"""

#%%  ----------------------------------------------------------------
# import and path
# ----------------------------------------------------------------

import xarray as xr
import pickle as pk
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import mapclassify as mc
from copy import deepcopy as cp
import os
import matplotlib.pyplot as plt
#import cartopy as cr
#import cartopy.crs as ccrs
import geopandas as gpd
import seaborn as sns
import rioxarray as rio

scriptsdir = os.getcwd()


#%% ----------------------------------------------------------------
# flags
# ----------------------------------------------------------------

# extreme event
global flags

flags = {}
flags['extr'] = 'tropicalcyclonedarea' # 0: all
                                # 1: burntarea
                                # 2: cropfailedarea
                                # 3: driedarea
                                # 4: floodedarea
                                # 5: heatwavedarea
                                # 6: tropicalcyclonedarea
                                # 7: waterscarcity
flags['mode'] = 'airports' # 0: all
                                # 1: airports
                                # 2: iww
                                # 3: ports
                                # 4: railways
                                # 5: roads
                                # 6: rrt
                                # 7: urban_nodes  
flags['run'] = 0          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                                # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['exposure_modes'] = 0     # 0: do not run analysis to calculate exposure for each transport type
                                # 1: run exposure analysis

flags['test1'] = 1       # 0: do not save the figure for each gcm and RCP for selected trasport mode as test
                                   # 1: save the figure

flags['Multi-model_median']= 0 # 0: run the calculation of multimodel mean for the two time windows and the three year: 2030,2040,2050
                                # 1: import pickles with processed data
                                   
flags['gmt'] = 'ar6'        # original: use Wim's stylized trajectory approach with max trajectory a linear increase to 3.5 deg                               
                            # ar6: substitute the linear max wth the highest IASA c7 scenario (increasing to ~4.0), new lower bound, and new 1.5, 2.0, NDC (2.8), 3.0

flags['rm'] = 'rm'       # no_rm: no smoothing of RCP GMTs before mapping
                          # rm: 21-year rolling mean on RCP GMTs 

# TODO: add rest of flags
#%%
# Directory
data_dir="C:/Users/Cristina/OneDrive - Vrije Universiteit Brussel/Documents/Project 2022/European Commission/Task 2/Code/lifetime_exposure_isimip-emergence"

os.chdir(data_dir)

#%% ----------------------------------------------------------------
# settings
# ----------------------------------------------------------------

from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
set_extremes(flags)


#%% ----------------------------------------------------------------
# load and manipulate demographic, GMT and ISIMIP data
# ----------------------------------------------------------------

from load_manip import *

# --------------------------------------------------------------------
# Load global mean temperature projections
global df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj

df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj, GMT_indices = load_GMT(
    year_start,
    year_end,
    year_range,
    flags,
)

#%% --------------------------------------------------------------------
# Select area of interest: Europe

global grid_area
grid_area = xr.open_dataarray('./data/isimip/clm45_area.nc4')
Europe=gpd.read_file('./data/EU_shape/Europe.shp')  

# Clip just in Europe
lon_min, lon_max = -35.0, 40.0
lat_min, lat_max = 25.0, 72.0

# Clip the dataset to the specified latitude and longitude bounds
grid_EU = grid_area.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min)) 

#%% --------------------------------------------------------------------
# load ISIMIP model data
from Exposure_transport import *

global grid_area
grid_area = xr.open_dataarray('./data/isimip/clm45_area.nc4')



d_isimip_meta, GMT_rcp = load_isimipdata(
    extremes,
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,
    df_GMT_strj,
    flags,
)


# Information about the run: gcm and rcp

sim_data = {}

# Iterate through keys in d_isimip_meta
for i in list(d_isimip_meta.keys()):
    # Initialize an empty dictionary for each key in sim_data
    sim_data[i] = {}
    
    # Populate the nested dictionary with values from d_isimip_meta
    sim_data[i]['gcm'] = d_isimip_meta[i]['gcm']
    sim_data[i]['rcp'] = d_isimip_meta[i]['rcp']
    sim_data[i]['model'] = d_isimip_meta[i]['model']

#%% ----------------------------------------------------------------
# compute exposure per each transport modes
# ------------------------------------------------------------------

# --------------------------------------------------------------------
# Clip the exposure data for each transport nodes and each run
# Compute the mean on all the run

year_range=np.arange(1861, 2100)


if flags['exposure_modes']: 
     
    # calculate lifetime exposure per country and per region and save data
   all_modes=calc_exposure_for_tr_modes(
       d_isimip_meta,
       year_range,
       grid_EU,
       flags )
    

else: # load processed exposure data

    print('Loading processed exposure for each transport mode')

    # load lifetime exposure pickle
    with open('./data/pickles/{}/Transport_modes/0_Allmode_dataset_{}.pkl'.format(flags['extr'],flags['extr']),'rb') as f:
        all_modes = pk.load(f)  

#%% ----------------------------------------------------------------
# Event selection: if extreme is flood or wildfire
# ------------------------------------------------------------------

# Transform percentage in binary
tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

if flags['extr'] in ['floodedarea', 'tropicalcyclonedarea', 'burntarea']:
    
    #Selection threshold for defining the event
    thr_sel=0.01
        
    #Substituting exposure variable to binary data
    for trmode in tr_modes:
        all_modes[trmode]['percentage'] = all_modes[trmode]['exposure'] 
        all_modes[trmode]['exposure'] = xr.where( all_modes[trmode]['exposure'] >= thr_sel,
        1,  xr.where(all_modes[trmode]['exposure'] < thr_sel, 0, np.nan))

print(flags['extr'], ': selected threshold',thr_sel)   


#%% ----------------------------------------------------------------
# Test 1
# Plot sum of events for all transport modes for each gcm and RCP
# ------------------------------------------------------------------

if flags['test1']: 
    
    from Test_Plot_gcm_RCP2 import *
    tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']
    #tr_mode_sel='railways'
    
    for tr_mode_sel in tr_modes:
       
        plot_gcm_rcp(all_modes,flags,tr_mode_sel,sim_data,d_isimip_meta,"Rolmean")
        plt.close()

#%% ----------------------------------------------------------------
# Median values if needed
# ------------------------------------------------------------------

# keys run
keys_rcp26 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp26']
keys_rcp60 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp60']
keys_rcp85 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp85']


Mode26_median=median_CI_allmodes(all_modes,keys_rcp26,d_isimip_meta,grid_EU)
Mode60_median=median_CI_allmodes(all_modes,keys_rcp60,d_isimip_meta,grid_EU)
Mode85_median=median_CI_allmodes(all_modes,keys_rcp85,d_isimip_meta,grid_EU)

 
#%%
#plt.figure()
#all_modes['airports']['percentage'].sel(time=2090,run=80).plot()

#%% ----------------------------------------------------------------
#%%%%%%%%%%%   NOT AUTOMATIC CODE SAVE PICK FOR ALL A SELECTED YEAR AND TIME WINDOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Select year and time window 

# Compute probability in past present and future

list_year= [2050,2075]

y_selected=list_year[1]

y_past=1861
y_pres=2005
tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

time_window=51

#Probability of exposure
ratio_allmodes_2050, ref_past, ref_pres, ref_2050 =pb_exposure_allmodes(time_window, 2050, all_modes,d_isimip_meta,grid_EU)
ratio_allmodes_2075, ref_past, ref_pres, ref_2075 =pb_exposure_allmodes(time_window, 2075, all_modes,d_isimip_meta,grid_EU)
#ratio_allmodes_2050=pb_exposure_allmodes(time_window, 2050, all_modes,d_isimip_meta,grid_EU)


#%%
# Median over the rcp, here calculate also the ratio of the probability given the mean pb over all models

Median_RCP_2050= median_ratio_rcp_allmodes( rcp_list,ratio_allmodes_2050,d_isimip_meta,grid_EU)
Median_RCP_2075= median_ratio_rcp_allmodes( rcp_list,ratio_allmodes_2075,d_isimip_meta,grid_EU)
#Mean_RCP_2050= mean_ratio_rcp_allmodes( rcp_list,ratio_allmodes_2050,d_isimip_meta,grid_EU)


#%% ----------------- Plots  -----------------#%%
from Final_Plots import *
dir_plot='./Preliminary_results/MeanRCP/'+flags['extr']+'/'


# Import Europe shapefile
eu_shap_wgs = './data/EU_shape/Europe.shp'
eu_wgs = gpd.read_file(eu_shap_wgs)
#eu_wgs.crs


# Import Europe Coastline shapefile
eu_coast_shap = './data/EU_shape/EU_coast_wgs84.shp'
eu_coast = gpd.read_file(eu_coast_shap)


Median_RCP_YY=Median_RCP_2050
var_list = ['ratio_pres', 'ratio_past']

maxv=10

for var_sel in var_list:

    #Plot Year 2050    
    plot_var_RCP(Median_RCP_2050,var_sel,2050,dir_plot)
    #composite_Nevent_pastfut(Median_RCP_2050,2050,dir_plot, ref_pres, ref_past)

    #Plot Year 2075     
    plot_var_RCP(Median_RCP_2075,var_sel,2075,dir_plot)
    #composite_Nevent_pastfut(Median_RCP_2075,2075,dir_plot, ref_pres, ref_past)


#%%###############   Save intermediate data              #%%#######3#############  

#Airp_2030_rcp=Mean_RCP_2030['airports'].sel(rcp=85)

#Airp_2030_rcp.to_netcdf("./data/pickles/Preliminary_results/Airp_2030_rcp85.nc")
#Mean_RCP_2030.to_netcdf("./data/pickles/Preliminary_results/Mean_RCP_2030.nc")


# Save pickles
with open('./Preliminary_results/pickles/{}_Median_RCP_2050.pkl'.format(flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(Median_RCP_2050,f)   
        
# Save pickles
with open('./Preliminary_results/pickles/{}_Median_RCP_2075.pkl'.format(flags['extr'],flags['extr']),'wb') as f:
        pk.dump(Median_RCP_2075,f) 
        
        
if thr_sel==0.01:
    with open('./Preliminary_results/pickles/TH1/{}_Median_RCP_2050.pkl'.format(flags['extr'],flags['extr']), 'wb') as f:
            pk.dump(Median_RCP_2050,f)   
            
    # Save pickles
    with open('./Preliminary_results/pickles/TH1/{}_Median_RCP_2075.pkl'.format(flags['extr'],flags['extr']),'wb') as f:
            pk.dump(Median_RCP_2075,f) 
            
    
    
       
#%%###########################

# Get the colormap from the m1_plot
cmap = m1_plot.get_cmap()

# Add a colorbar at the bottom using the colormap from m1_plot
cax = fig.add_axes([0.2, 0.1, 0.6, 0.02])  # Adjust the position and size as needed
norm = colors.Normalize(vmin=0, vmax=max_val)  # Adjust vmin and vmax as needed
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='max')
cbar.set_label('Ratio present')  # Set your colorbar label

    
#%% ----------------------------------------------------------------
#%%%%%%%%%%%  PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dir_plot='./Preliminary_results/MeanRCP/'+flags['extr']+'/'
ipc_list = [26, 60, 85]

Mean_RCP_YY=Median_RCP_2050


# Import Europe shapefile
eu_shap_wgs = './data/EU_shape/Europe.shp'
eu_wgs = gpd.read_file(eu_shap_wgs)
#eu_wgs.crs


# Import Europe Coastline shapefile
eu_coast_shap = './data/EU_shape/EU_coast_wgs84.shp'
eu_coast = gpd.read_file(eu_coast_shap)

#% # SINGLE PLOT ##################

Median_RCP_YY=Median_RCP_2050  
Y_sel=2050 

for mode_sel in tr_modes:
       
   plt.figure()
    
   fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    
   # Plot Past and present
    
   m_past=Mean_RCP_YY[mode_sel]['x_past'].sel(rcp=26)
   m_pres=Mean_RCP_YY[mode_sel]['x_pres'].sel(rcp=26)
    
   eu_coast.plot(ax=axs[0], color='lightgrey')
   Europe.plot(ax=axs[0], color='darkgrey')
  # m_past.plot(ax=axs[0], cmap='autumn_r', add_colorbar=True)
   m_past.plot(ax=axs[0], cmap='YlOrRd', add_colorbar=True) 
   axs[0].set_title(f'Past {ref_past}')
    
   eu_coast.plot(ax=axs[1], color='lightgrey')
   Europe.plot(ax=axs[1], color='darkgrey')
   m_pres.plot(ax=axs[1], cmap='YlOrRd', add_colorbar=True)     
   axs[1].set_title(f'Present {ref_pres}')
        
           
   # Add text to the plot
   #        text_content = f'Time window: {y_selected-(time_window-1)/2} to {y_selected+(time_window-1)/2}'
   
   #        axs[2].text(0.5, 0.95,text_content, ha='center', va='center', transform=axs[2].transAxes,
   #                        bbox=dict(facecolor='white', alpha=0.7))
    
   plt.suptitle(f'Past and present, window_period: {time_window}')
   plt.savefig(dir_plot + mode_sel + str(y_selected)+'_PastPresent_timew_'+ str(time_window)+ ".png", dpi=900)
   plt.close()
     
   # Plot future

   var_sel='x_fut'
    
   plt.figure()
    
   fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharex=True, sharey=True)     
   
   for i, ii in enumerate(ipc_list):
               
       m1=Median_RCP_YY[mode_sel][var_sel].sel(rcp=ii)
       max_val=Median_RCP_YY[mode_sel][var_sel].sel(rcp=85).max()
        
       eu_coast.plot(ax=axs[i], color='lightgrey')
       Europe.plot(ax=axs[i], color='darkgrey')
       m1.plot(ax=axs[i], cmap='YlOrRd', add_colorbar=True, vmin=0, vmax=max_val)
 
   plt.suptitle(f'{mode_sel}: {var_sel},{y_selected}, window_period: {time_window} days')
   plt.savefig(dir_plot + mode_sel + str(y_selected)+'_'+var_sel+".png", dpi=900)
   plt.close()

#%%

var_sel='ratio_pres'
 
plt.figure()

for mode_sel in tr_modes: 
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharex=True, sharey=True)     
    
    for i, ii in enumerate(ipc_list):
                
        m1=Median_RCP_YY[mode_sel][var_sel].sel(rcp=ii)
        max_val=Median_RCP_YY[mode_sel][var_sel].sel(rcp=85).max()
         
        eu_coast.plot(ax=axs[i], color='lightgrey')
        Europe.plot(ax=axs[i], color='darkgrey')
        m1.plot(ax=axs[i], cmap='YlOrRd', add_colorbar=True, vmin=0, vmax=10)
        #cbar = fig.colorbar(m1, ax=axs[i], extend='both')
        #cbar.set_clim(0, max_val)  # Set the colorbar limits
    plt.suptitle(f'{mode_sel}: {var_sel},{y_selected}, window_period: {time_window} days')
    plt.savefig(dir_plot + mode_sel + str(y_selected)+'_'+var_sel+".png", dpi=900)
    plt.close()


#%%

list_year= [2050,2075]
tw_list=[51]
ipc_list = [26, 60, 85]
dir_pick='./Preliminary_results/MeanRCP/'+flags['extr']+'/'

tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

Median_RCP_YY= Median_RCP_2050

for mode_sel in tr_modes:
    
    med=data[mode_sel]
    
    #Plot future
    plt.figure()
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 5), sharex=True, sharey=True)
        
    eu_coast.plot(ax=axs[i,i], color='lightgrey')
    Europe.plot(ax=axs[i,i], color='navajowhite')
    
    mpast=Median_RCP_YY[mode_sel]['x_past'].sel(rcp=26).plot(ax=axs[0], cmap='hot_r', add_colorbar=True, vmin=0, vmax=max_val)
    mpres=Median_RCP_YY[mode_sel]['x_pres'].sel(rcp=26) .plot(ax=axs[1], cmap='hot_r', add_colorbar=True, vmin=0, vmax=max_val)  
    
    m1.plot(ax=axs[i], cmap='hot_r', add_colorbar=True, vmin=0, vmax=max_val)
    
    
    for i, ii in enumerate(ipc_list):    
    
    
            m1=Mean_RCP_YY[mode_sel][var_sel].sel(rcp=ii)
            #max_val=Mean_RCP_YY[mode_sel][var_sel].sel(rcp=85).max()
            max_val = Mean_RCP_YY[mode_sel][var_sel].sel(rcp=85).where(Mean_RCP_YY[mode_sel][var_sel].sel(rcp=85) < 80).max() 
          
          
            eu_coast.plot(ax=axs[i], color='lightgrey')
            Europe.plot(ax=axs[i], color='navajowhite')
            m1.plot(ax=axs[i], cmap='hot_r', add_colorbar=True, vmin=0, vmax=max_val)
        
        # Add text to the plot
    #        text_content = f'Time window: {y_selected-(time_window-1)/2} to {y_selected+(time_window-1)/2}'
    
    #        axs[2].text(0.5, 0.95,text_content, ha='center', va='center', transform=axs[2].transAxes,
    #                        bbox=dict(facecolor='white', alpha=0.7))
    
        plt.suptitle(f'{mode_sel}: {var_sel},{y_selected}, window_period: {time_window} years {ref_fut}')
        plt.savefig(dir_pick + mode_sel + str(y_selected)+'_'+ var_sel+'_timew_'+ str(time_window)+ ".png", dpi=900)
        plt.close()
    



if flags['Multi-model_median']: 
    
    for y_selected in list_year:
        
        y_past=1861
        y_pres=2005
        tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']
        
        for time_window in tw_list:
            time_half=(time_window-1)/2
            ratio_allmodes_YY, ref_past, ref_pres, ref_fut=pb_exposure_allmodes(time_window, y_selected, all_modes,d_isimip_meta,grid_EU)
            Mean_RCP_YY= median_ratio_rcp_allmodes( rcp_list,ratio_allmodes_YY,d_isimip_meta,grid_EU)
            
            #save pickles
           # print('Saving pickles for year selected:',y_selected,', time window:',time_window, ' ', ref_fut)
           # with open('./Preliminary_results/MeanRCP/{}/MeanRCP_{}_timew_{}.pkl'.format(flags['extr'],str(y_selected),str(time_window)), 'wb') as f:
            #       pk.dump(Mean_RCP_YY,f)     
                  

            # Plot Multi-Model mean for years and selected variables
            
            mode_sel='railways'
            var_sel='x_past'
            
            #Plot future
            plt.figure()
        
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        
            for i, ii in enumerate(ipc_list):
        
                m1=Mean_RCP_YY[mode_sel][var_sel].sel(rcp=ii)
                #max_val=Mean_RCP_YY[mode_sel][var_sel].sel(rcp=85).max()
                max_val = Mean_RCP_YY[mode_sel][var_sel].sel(rcp=85).where(Mean_RCP_YY[mode_sel][var_sel].sel(rcp=85) < 80).max() 
              
              
                eu_coast.plot(ax=axs[i], color='lightgrey')
                Europe.plot(ax=axs[i], color='navajowhite')
                m1.plot(ax=axs[i], cmap='hot_r', add_colorbar=True, vmin=0, vmax=max_val)
            
            # Add text to the plot
        #        text_content = f'Time window: {y_selected-(time_window-1)/2} to {y_selected+(time_window-1)/2}'
        
        #        axs[2].text(0.5, 0.95,text_content, ha='center', va='center', transform=axs[2].transAxes,
        #                        bbox=dict(facecolor='white', alpha=0.7))
        
            plt.suptitle(f'{mode_sel}: {var_sel},{y_selected}, window_period: {time_window} years {ref_fut}')
            plt.savefig(dir_pick + mode_sel + str(y_selected)+'_'+ var_sel+'_timew_'+ str(time_window)+ ".png", dpi=900)
            plt.close()


else: # load processed multimodel data

    print('Loading processed data: multi-model mean for: 2030,2040,2050 and time window:21 and 51 years')

    # load pickled multimodel mean data for all timewindow and years

    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2030_timew_21.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2030_tw21 = pk.load(f)
    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2030_timew_51.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2030_tw51 = pk.load(f)

    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2040_timew_21.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2040_tw21 = pk.load(f)
    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2040_timew_51.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2040_tw51 = pk.load(f)
        
    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2050_timew_21.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2050_tw21 = pk.load(f)
    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2050_timew_51.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2050_tw51 = pk.load(f)
 
                    
                # with open('./data/pickles/{}/isimip_pic_metadata_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
                #     d_pic_meta = pk.load(f)        
                # with open('./data/pickles/GMT_rcp.pkl', 'rb') as f:
                #     GMT_rcp = pk.load(f)  

#%%        
        #Plot Past and present
        
        plt.figure()

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

        m_past=Mean_RCP_YY[mode_sel]['x_past'].sel(rcp=26)
        m_pres=Mean_RCP_YY[mode_sel]['x_pres'].sel(rcp=26)
        
        eu_coast.plot(ax=axs[0], color='lightgrey')
        Europe.plot(ax=axs[0], color='navajowhite')
        m_past.plot(ax=axs[0], cmap='hot_r', add_colorbar=True)
        axs[0].set_title(f'Past {ref_past}')
        
        eu_coast.plot(ax=axs[1], color='lightgrey')
        Europe.plot(ax=axs[1], color='navajowhite')
        m_pres.plot(ax=axs[1], cmap='hot_r', add_colorbar=True)     
        axs[1].set_title(f'Present {ref_pres}')
            
               
        # Add text to the plot
#        text_content = f'Time window: {y_selected-(time_window-1)/2} to {y_selected+(time_window-1)/2}'

#        axs[2].text(0.5, 0.95,text_content, ha='center', va='center', transform=axs[2].transAxes,
#                        bbox=dict(facecolor='white', alpha=0.7))

        plt.suptitle(f'Past and present, window_period: {time_window}')
        plt.savefig(dir_pick + mode_sel + str(y_selected)+'_PastPresent_timew_'+ str(time_window)+ ".png", dpi=900)
        plt.close()



#%%
mode_sel='airports'
var_sel='x_fut'

plt.figure()

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for i, ii in enumerate(ipc_list):

    m1=Median_RCP_YY[mode_sel][var_sel].sel(rcp=ii)
    max_val=Median_RCP_YY[mode_sel][var_sel].sel(rcp=85).max()

    eu_coast.plot(ax=axs[i], color='lightgrey')
    Europe.plot(ax=axs[i], color='navajowhite')
    m1.plot(ax=axs[i], cmap='hot_r', add_colorbar=True, vmin=0, vmax=max_val)
    
    plt.suptitle(f'{mode_sel}: {var_sel},{y_selected}, window_period: {time_window} days')



#%%--------------------------------------------------------------------
#Save figure of clipped maps for each mode
#%%
#Save figure of clipped maps for each mode

# Import Europe shapefile
eu_shap_wgs = './data/EU_shape/Europe.shp'
eu_wgs = gpd.read_file(eu_shap_wgs)
#eu_wgs.crs


# Import Europe Coastline shapefile
eu_coast_shap = './data/EU_shape/EU_coast_wgs84.shp'
eu_coast = gpd.read_file(eu_coast_shap)
#eu_coast.crs

#Plot per probability

# Plot per ratio
mode_rcp_mean=Mean_RCP_2030

#%% Multiple plot #######

Europe = gpd.read_file('./data/EU_shape/Europe.shp')
plot_dir = './Plot/Mean_ratio/'
ipc_list = [26, 60, 85]
var_sel = 'ratio_past'

# Assuming tr_modes, mode_rcp_mean, and eu_coast are defined

for trmode in tr_modes:
    # Create a figure and axis for the entire row of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    max_values = []  # To store the maximum value for each IPC

    for i, ii in enumerate(ipc_list):
        x1 = mode_rcp_mean[trmode][var_sel].sel(rcp=ii)

        max_val = mode_rcp_mean[trmode][var_sel].sel(rcp=85).where(mode_rcp_mean[trmode][var_sel].sel(rcp=85) < 80).max()
#       max_val = x1.where(x1 < 80).max()
#       max_val=mode_rcp_mean[trmode][var_sel].sel(rcp=85).max()
        #max_values.append(max_val)  # Store the maximum value for each IPC

        # Plot the base map (Europe)
        eu_coast.plot(ax=axs[i], color='lightgrey')
        Europe.plot(ax=axs[i], color='navajowhite')

        # Plot the data
        x1.plot(ax=axs[i], cmap='hot_r', add_colorbar=True, vmin=0, vmax=max_val)
#       x1.plot(ax=axs[i], cmap='viridis', add_colorbar=True, vmin=0, vmax=90)
#       x1.plot(ax=axs[i], cmap='viridis', add_colorbar=True, vmin=0, vmax=max_val)
#       x1.where(x1 >= 90).plot(ax=axs[i], cmap='flare', add_colorbar=True)

        axs[i].set_title(f'RPC={ii}')
        axs[i].set_xlim([-35, 35])
        axs[i].set_ylim([25, 72])
        #print(i)
        
    # Create a colorbar for the entire row of subplots
    #cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, max(max_values)), cmap='viridis'), ax=axs, orientation='vertical', label='Values')

    plt.suptitle(f'{var_sel} {trmode}')
    plt.savefig(plot_dir + var_sel + '/' +  str(time_window)+ '_' + var_sel + '_'+ trmode + "21dhotr2.png", dpi=900)
    plt.close()


#%% Single plot #######

if  flags['exposure_fig']:
    
    
    Europe=gpd.read_file('./data/EU_shape/Europe.shp')  
    #plot_dir='./Plot/Transport_mode/'
    plot_dir='./Plot/Mean_ratio/'
    ipc_list=26,60,85
#    var_sel_list=ratio_pres,ratio_past,pb_past,pb_pres,pb_fut
    var_sel='ratio_past'  
    
    for trmode in tr_modes:
         
        for ii in ipc_list:
                x1=mode_rcp_mean[trmode][var_sel].sel(rcp=ii)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                eu_coast.plot(ax=ax, color='lightgrey')
                # Plot the base map (Europe)
                Europe.plot(ax=ax, color='navajowhite')
                
                max_val=x1.where(x1 < 100).max()
                #x1.plot(cmap='viridis', add_colorbar=True, vmin=0, vmax=50) 
                x1.plot(cmap='viridis', add_colorbar=True, vmin=0, vmax=max_val)
                x1.where(x1 >= 100).plot(cmap='flare', add_colorbar=False)

                # Overlay the second shapefile (eu_coast) on the same axis
                # Set the limits of the axis

                ax.set_xlim([-35, 35])  # Replace min_x and max_x with your desired limits
                ax.set_ylim([25, 72])

                #plt.title('Europe with Coastline Overlay')
                #plt.show()                
                
                plt.title(var_sel+trmode+' ipc='+str(ii))
                plt.savefig(plot_dir+var_sel+'/'+'_'+var_sel+trmode+' ipc='+str(ii)+"2.png", dpi=900)

##################%%%%%%%%%%%          SENSITIVITY ANALYSIS          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%###################
# Import Europe shapefile
eu_shap_wgs = './data/EU_shape/Europe.shp'
eu_wgs = gpd.read_file(eu_shap_wgs)
#eu_wgs.crs


# Import Europe Coastline shapefile
eu_coast_shap = './data/EU_shape/EU_coast_wgs84.shp'
eu_coast = gpd.read_file(eu_coast_shap)


list_year= [2050,2075]
tw_list=[51]
ipc_list = [26, 60, 85]
dir_pick='./Preliminary_results/MeanRCP/'+flags['extr']+'/'


data_dir





if flags['Multi-model_median']: 
    
    for y_selected in list_year:
        
        y_past=1861
        y_pres=2005
        tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']
        
        for time_window in tw_list:
            time_half=(time_window-1)/2
            ratio_allmodes_YY, ref_past, ref_pres, ref_fut=pb_exposure_allmodes(time_window, y_selected, all_modes,d_isimip_meta,grid_EU)
            Mean_RCP_YY= median_ratio_rcp_allmodes( rcp_list,ratio_allmodes_YY,d_isimip_meta,grid_EU)
            
            #save pickles
           # print('Saving pickles for year selected:',y_selected,', time window:',time_window, ' ', ref_fut)
           # with open('./Preliminary_results/MeanRCP/{}/MeanRCP_{}_timew_{}.pkl'.format(flags['extr'],str(y_selected),str(time_window)), 'wb') as f:
            #       pk.dump(Mean_RCP_YY,f)     
                  

            # Plot Multi-Model mean for years and selected variables
            
            mode_sel='railways'
            var_sel='x_past'
            
            #Plot future
            plt.figure()
        
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        
            for i, ii in enumerate(ipc_list):
        
                m1=Mean_RCP_YY[mode_sel][var_sel].sel(rcp=ii)
                #max_val=Mean_RCP_YY[mode_sel][var_sel].sel(rcp=85).max()
                max_val = Mean_RCP_YY[mode_sel][var_sel].sel(rcp=85).where(Mean_RCP_YY[mode_sel][var_sel].sel(rcp=85) < 80).max() 
              
              
                eu_coast.plot(ax=axs[i], color='lightgrey')
                Europe.plot(ax=axs[i], color='navajowhite')
                m1.plot(ax=axs[i], cmap='hot_r', add_colorbar=True, vmin=0, vmax=max_val)
            
            # Add text to the plot
        #        text_content = f'Time window: {y_selected-(time_window-1)/2} to {y_selected+(time_window-1)/2}'
        
        #        axs[2].text(0.5, 0.95,text_content, ha='center', va='center', transform=axs[2].transAxes,
        #                        bbox=dict(facecolor='white', alpha=0.7))
        
            plt.suptitle(f'{mode_sel}: {var_sel},{y_selected}, window_period: {time_window} years {ref_fut}')
            plt.savefig(dir_pick + mode_sel + str(y_selected)+'_'+ var_sel+'_timew_'+ str(time_window)+ ".png", dpi=900)
            plt.close()


else: # load processed multimodel data

    print('Loading processed data: multi-model mean for: 2030,2040,2050 and time window:21 and 51 years')

    # load pickled multimodel mean data for all timewindow and years

    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2030_timew_21.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2030_tw21 = pk.load(f)
    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2030_timew_51.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2030_tw51 = pk.load(f)

    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2040_timew_21.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2040_tw21 = pk.load(f)
    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2040_timew_51.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2040_tw51 = pk.load(f)
        
    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2050_timew_21.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2050_tw21 = pk.load(f)
    with open('./Preliminary_results/MeanRCP/{}/MeanRCP_2050_timew_51.pkl'.format(flags['extr']), 'rb') as f:
        MeanRCP_2050_tw51 = pk.load(f)
 
                    
                # with open('./data/pickles/{}/isimip_pic_metadata_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
                #     d_pic_meta = pk.load(f)        
                # with open('./data/pickles/GMT_rcp.pkl', 'rb') as f:
                #     GMT_rcp = pk.load(f)  

#%%        





#%%--------------------------------------------------------------------



#%%--------------------------------------- OLD PLOTS  -----------------------------------------------------#############

x1 = mode_rcp_mean['airports'].sel(rcp=26).to_dataframe()

x2 = mode_rcp_mean['airports'].sel(rcp=60).to_dataframe()

x3 = mode_rcp_mean['airports'].sel(rcp=85).to_dataframe()
        
#%%--------------------------------------------------------------------
#Save figure of clipped maps for each mode

mode_rcp_mean=Mean_RCP_2030

if  flags['exposure_fig']:
    
    tr_modes=['airports','ports','railways',"rrt",'urban_nodes','iww']
    Europe=gpd.read_file('./data/EU_shape/Europe.shp')  
    #plot_dir='./Plot/Transport_mode/'
    plot_dir='./Plot/Mean_ratio/'
    ipc_list=26,60,89
    
    for trmode in tr_modes:
        year_sel=1960,2100
        
        for ii in ipc_list:
            for tt in year_sel:
                x1=mode_rcp_mean[trmode]['mean_exp'].sel(time=tt,rcp=ii)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                eu_coast.plot(ax=ax, color='lightgrey')
                # Plot the base map (Europe)
                Europe.plot(ax=ax, color='navajowhite')
                x1.plot(cmap='viridis', add_colorbar=True, vmin=0, vmax=50) 
                x1.where(x1 == 100).plot(cmap='flare', add_colorbar=False)

                # Overlay the second shapefile (eu_coast) on the same axis
                # Set the limits of the axis

                ax.set_xlim([-35, 35])  # Replace min_x and max_x with your desired limits
                ax.set_ylim([25, 72])

                #plt.title('Europe with Coastline Overlay')
                plt.show()
                
                
                
                # plt.figure()
                # Europe.plot(color='navajowhite')
                # x1.plot(cmap='viridis', add_colorbar=True, vmin=0, vmax=50) 
                # x1.where(x1 == 100).plot(cmap='flare', add_colorbar=False)
                # plt.title(trmode+' ipc='+str(ii)+'- year:'+str(tt))
                # plt.savefig(plot_dir+trmode+' ipc='+str(ii)+'year'+str(tt)+".png", dpi=900)


                
                # plt.figure()
                # Europe.plot(color='navajowhite')
                # x1.plot(vmin=0, vmax=1)
                # plt.title(trmode+' ipc='+str(ii)+'- year:'+str(tt))
                # plt.savefig(plot_dir+trmode+' ipc='+str(ii)+'year'+str(tt)+".png", dpi=900)
        
#%%
if  flags['exposure_fig']:
    
    
    Europe=gpd.read_file('./data/EU_shape/Europe.shp')  
    #plot_dir='./Plot/Transport_mode/'
    plot_dir='./Plot/Mean_ratio/'
    ipc_list=26,60,85
    #var_sel_list=ratio_pres,ratio_past,pb_past,pb_pres,pb_fut
    var_sel='pb_past'  
    
    for trmode in tr_modes:
         
        for ii in ipc_list:
                x1=mode_rcp_mean[trmode][var_sel].sel(rcp=ii)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                eu_coast.plot(ax=ax, color='lightgrey')
                # Plot the base map (Europe)
                Europe.plot(ax=ax, color='navajowhite')
                
                #max_val=x1.where(x1 < 5000).max()
                #x1.plot(cmap='viridis', add_colorbar=True, vmin=0, vmax=50) 
                x1.plot(cmap='viridis', add_colorbar=True)
                #x1.where(x1 > 5000).plot(cmap='flare', add_colorbar=False)

                # Overlay the second shapefile (eu_coast) on the same axis
                # Set the limits of the axis

                ax.set_xlim([-35, 35])  # Replace min_x and max_x with your desired limits
                ax.set_ylim([25, 72])

                #plt.title('Europe with Coastline Overlay')
                #plt.show()                
                
                plt.title(var_sel+trmode+' ipc='+str(ii))
                plt.savefig(plot_dir+var_sel+'/'+'_'+var_sel+trmode+' ipc='+str(ii)+"2.png", dpi=900)



#%%
for trmode in tr_modes:
    for ii in [0,1,2]:
        
        plt.figure()
        Europe.plot(color='navajowhite')
        x1=mode_rcp_mean[trmode]['mean_exp'].isel(rcp=ii,time=30)
        x1.plot(vmin=0, vmax=1)
        plt.savefig(plot_dir+trmode+' ipc='+str(ii)+'year'+str(30)+".png", dpi=900)
        
        plt.figure()
        Europe.plot(color='navajowhite')
        x2=mode_rcp_mean[trmode]['mean_exp'].isel(rcp=ii,time=100)
        x2.plot(vmin=0, vmax=1)
        plt.savefig(plot_dir+trmode+' ipc='+str(ii)+'year'+str(100)+".png", dpi=900)
    
    
    





plt.figure()
Europe.plot()
ds_europe.isel(time=1).plot()

plt.figure()
Europe.plot()
mean_exposure.isel(time=10).plot()

uu=mean_exposure.sel(time=2100)
plt.figure()
Europe.plot()
uu.plot()




#Extract information database of exposure for each tr node













