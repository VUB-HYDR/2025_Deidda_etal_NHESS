# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:42:25 2024

@author: Cristina
"""
import numpy as np
from matplotlib import colors
import matplotlib as mpl
import geopandas as gpd
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import pickle as pk
from matplotlib.colors import ListedColormap

#%%
# Directory

data_dir="C:/Users/Cristina/OneDrive - Vrije Universiteit Brussel/Documents/Project 2022/European Commission/Task 2/Code/lifetime_exposure_isimip-emergence"

os.chdir(data_dir)

#%% ----------------------------------------------------------------
#%%%%  Shapefile #%%%%

# Import Europe shapefile
eu_shap_wgs = './data/EU_shape/Europe.shp'
eu_wgs = gpd.read_file(eu_shap_wgs)
#eu_wgs.crs


# Import Europe Coastline shapefile
eu_coast_shap = './data/EU_shape/EU_coast_wgs84.shp'
eu_coast = gpd.read_file(eu_coast_shap)

Europe = gpd.read_file('./data/EU_shape/Europe.shp')


#%%%% Load climate projection data
global flags

flags = {}
flags['extr'] = 'driedarea' # 0: all
                                # 1: burntarea
                                # 3: driedarea
                                # 4: floodedarea
                                # 5: heatwavedarea
                                # 6: tropicalcyclonedarea

with open('./Preliminary_results/pickles_v1/{}_Median_RCP_2050.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
      Median_RCP_2050 =  pk.load(f)  
      
with open('./Preliminary_results/pickles_v1/{}_Median_RCP_2075.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
      Median_RCP_2075 =  pk.load(f) 

th_sel=0.05
TH_val= round(th_sel*100)      
#%%%
# th_sel=0.05
# TH_val= round(th_sel*100)
          
# if th_sel==0.01:
    
#     with open('./Preliminary_results/pickles/TH{}/{}_Median_RCP_2050_{}.pkl'.format(TH_val,flags['extr'],TH_val), 'rb') as f:
#           Median_RCP_2050 =  pk.load(f)  
          
#     with open('./Preliminary_results/pickles/TH{}/{}_Median_RCP_2050_{}.pkl'.format(TH_val,flags['extr'],TH_val), 'rb') as f:
#           Median_RCP_2075 =  pk.load(f)        
# else :
#    with open('./Preliminary_results/pickles/{}_Median_RCP_2050.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
#          Median_RCP_2050 =  pk.load(f)  
         
#    with open('./Preliminary_results/pickles/{}_Median_RCP_2075.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
#          Median_RCP_2075 =  pk.load(f) 
    
#%%%%

#plot_dir='./Final_results/composite/'+flags['extr']+'/'

excmap= 'YlOrRd'    # 1: burntarea
            # 3: driedarea
            # 4: floodedarea
      #'YlOrRd'     # 5: heatwavedarea
            # 6: tropicalcyclonedarea

ipc_list = [26, 60, 85]
ipc_tit=['RCP 2.6','RCP 6.0','RCP 8.5']
var_sel = 'ratio_pres'

#tr_modesup = ['airports', 'ports', 'railways', 'roads', 'iww']
#tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

if th_sel==0.01:
    plot_dir='./Final_results/composite/May/TH1/'
else:
    plot_dir='./Final_results/composite/May/Review/'


tr_modes=['airports', 'ports', 'railways', 'roads', 'iww']
tr_names=['Airports', 'Ports', 'Railways', 'Roads', 'IWWs']

var_sel='ratio_pres'


# Define your column names
column_names = ['2024-2075', '2049-2100']  # Adjust as needed


for i, ii in enumerate(ipc_list):

    # Create a figure and axis for the entire row of subplots
    fig, axs = plt.subplots(5, 2, figsize=(10, 45), sharex=True, sharey=True)
    #fig.suptitle(ipc_tit[i], fontsize=16)
    fig.subplots_adjust(hspace=0.1, wspace=0.05)
    # Add column names
#    for col, name in enumerate(column_names):
#        fig.text(0.2 * (col + 1), 0.96, name, ha='left', fontsize=14)
    plt.gcf().text(0.265,0.92,'2024-2075', fontsize=13)
    plt.gcf().text(0.685,0.92,'2049-2100', fontsize=13)
    
    x0 = Median_RCP_2050['railways'][var_sel].sel(rcp=ii)
    max_val = x0.where(x0 < 80).max()
    
    for j, trmode in enumerate(tr_modes):

        eu_coast.plot(ax=axs[j, 0], color='lightgrey')
        Europe.plot(ax=axs[j, 0], color='darkgrey')
        
        mode_shap='./data/TENT_shapefiles/{}_GL2017_EU.shp'.format(trmode)
        mode = gpd.read_file(mode_shap)
        mode=mode.to_crs("EPSG:4326") 
        
        # Plot the data for 2050
        x1 = Median_RCP_2050[trmode][var_sel].sel(rcp=ii)
        #max_val_2050 = x1.where(x1 < 80).max()
        yy=x1.to_dataframe()
        yy = yy.reset_index()
        squares = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for _, (lat, lon) in yy[['lat', 'lon']].iterrows()]
        gdf_squares = gpd.GeoDataFrame(geometry=squares, crs='epsg:4326')
            
        gdf_squares['GridIndex']=gdf_squares.index
        gdf_squares['lat'] = yy['lat']
        gdf_squares['lon'] = yy['lon']
        gdf_squares['ratio_pres'] =yy['ratio_pres']
        
        Shapef =gpd.sjoin(mode, gdf_squares,how='left', op='intersects')
        Shapef['ratio_pres1']=Shapef['ratio_pres'].fillna(1)
        Shapef=Shapef.to_crs("EPSG:4326") 
        
        # Define the discrete intervals for the legend
        num_intervals = 8
        intervals = np.linspace(1, max_val, num_intervals + 1)

        # Create a colormap with discrete colors
        colors = plt.cm.get_cmap(excmap, num_intervals)
        cmap_discrete = ListedColormap([colors(i) for i in range(num_intervals)])


        Shapef.plot(column='ratio_pres1',ax=axs[j, 0],  cmap=cmap_discrete, legend=False, markersize=0.7, vmin=1, vmax=max_val)
        Shapef.plot(column='ratio_pres',ax=axs[j, 0],  cmap=cmap_discrete, legend=False, markersize=0.7, vmin=1, vmax=max_val)
        
        axs[j, 0].set_aspect('equal', adjustable='box')
        #axs[j, 0].set_title(f'{tr_names[j]}', loc='left', fontsize=13)
        axs[j, 0].set_xlim([-15, 35])
        axs[j, 0].set_ylim([30, 75])
        
                # Hide spines and tick labels
        axs[j, 0].spines['top'].set_visible(False)
        axs[j, 0].spines['right'].set_visible(False)
        axs[j, 0].spines['left'].set_visible(False)
        axs[j, 0].spines['bottom'].set_visible(False)
        axs[j, 0].xaxis.set_visible(False)
        axs[j, 0].yaxis.set_visible(False)


        # Plot the base map (Europe) for the second column
        eu_coast.plot(ax=axs[j, 1], color='lightgrey')
        Europe.plot(ax=axs[j, 1], color='darkgrey')
        
        
        # Plot the data for 2075
        x2 = Median_RCP_2075[trmode][var_sel].sel(rcp=ii)
        yy2=x2.to_dataframe()
        yy2 = yy2.reset_index()
        gdf_squares['ratio_pres'] =yy2['ratio_pres']
        max_val_2075 = x2.where(x2 < 80).max()
        #max_val_2075 = 5
        
        Shapef2 =gpd.sjoin(mode, gdf_squares,how='left', op='intersects')       
        Shapef2['ratio_pres1']=Shapef2['ratio_pres'].fillna(1)
        Shapef2.plot(column='ratio_pres1',ax=axs[j, 1],  cmap=cmap_discrete, legend=False, markersize=0.7, vmin=1, vmax=max_val)
        Shapef2.plot(column='ratio_pres',ax=axs[j, 1],  cmap=cmap_discrete, legend=False, markersize=0.7, vmin=1, vmax=max_val)
        Shapef2=Shapef2.to_crs("EPSG:4326") 
        
        axs[j, 1].set_aspect('equal', adjustable='box')
        #axs[j, 1].set_title(f'{trmode}')
        axs[j, 1].set_xlim([-15, 35])
        axs[j, 1].set_ylim([30, 75])
        
                # Hide spines and tick labels
        axs[j, 1].spines['top'].set_visible(False)
        axs[j, 1].spines['right'].set_visible(False)
        axs[j, 1].spines['left'].set_visible(False)
        axs[j,1].spines['bottom'].set_visible(False)
        axs[j, 1].xaxis.set_visible(False)
        axs[j, 1].yaxis.set_visible(False)
        
        # Add subplot title in the middle of the row
        fig.text(0.5, 0.9 - j * 0.16, f'{tr_names[j]}', ha='center', fontsize=13)


    # Add a color bar at the end
    cax = fig.add_axes([0.2, 0.085, 0.6, 0.014])  # Adjust the position and size as needed
    norm = plt.Normalize(vmin=1, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap_discrete, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='max')
    cbar.set_label('Exposure multiplication factor')
    
    plt.savefig(plot_dir +"Compos_RCP_"+str(ii) + flags['extr']+"_TH" + str(TH_val)+ "350.png", dpi=900)
    
#%%    


#plot_dir='./Final_results/composite/'+flags['extr']+'/'

excmap= 'YlOrRd'    # 1: burntarea
            # 3: driedarea
            # 4: floodedarea
      #'YlOrRd'     # 5: heatwavedarea
            # 6: tropicalcyclonedarea

ipc_list = [26, 60, 85]
ipc_tit=['RCP 2.6','RCP 6.0','RCP 8.5']
var_sel = 'ratio_pres'

#tr_modesup = ['airports', 'ports', 'railways', 'roads', 'iww']
#tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

#if th_sel==0.01:
#    plot_dir='./Final_results/composite/May/TH1/'
#else:
    
plot_dir='./Final_results/composite/May/urbannodes'


#tr_modes=['airports', 'ports', 'railways', 'roads', 'iww']
#tr_names=['Airports', 'Ports', 'Railways', 'Roads', 'IWW']

tr_modes=['urban_nodes', 'rrt']
tr_names=['Urban nodes', 'RRT']


var_sel='ratio_pres'


# Define your column names
column_names = ['2024-2075', '2049-2100']  # Adjust as needed


for i, ii in enumerate(ipc_list):

    # Create a figure and axis for the entire row of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle(ipc_tit[i], fontsize=16)
    fig.subplots_adjust(hspace=0.1, wspace=0.0001)
    # Add column names
#    for col, name in enumerate(column_names):
#        fig.text(0.2 * (col + 1), 0.96, name, ha='left', fontsize=14)
    plt.gcf().text(0.265,0.92,'2024-2075', fontsize=13)
    plt.gcf().text(0.685,0.92,'2049-2100', fontsize=13)
    
    x0 = Median_RCP_2075['urban_nodes'][var_sel].sel(rcp=ii)
    max_val = x0.where(x0 < 80).max()
    
    for j, trmode in enumerate(tr_modes):

        eu_coast.plot(ax=axs[j, 0], color='lightgrey')
        Europe.plot(ax=axs[j, 0], color='darkgrey')
        
        mode_shap='./data/TENT_shapefiles/{}_GL2017_EU.shp'.format(trmode)
        mode = gpd.read_file(mode_shap)
        mode=mode.to_crs("EPSG:4326") 
        
        # Plot the data for 2050
        x1 = Median_RCP_2050[trmode][var_sel].sel(rcp=ii)
        #max_val_2050 = x1.where(x1 < 80).max()
        yy=x1.to_dataframe()
        yy = yy.reset_index()
        squares = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for _, (lat, lon) in yy[['lat', 'lon']].iterrows()]
        gdf_squares = gpd.GeoDataFrame(geometry=squares, crs='epsg:4326')
            
        gdf_squares['GridIndex']=gdf_squares.index
        gdf_squares['lat'] = yy['lat']
        gdf_squares['lon'] = yy['lon']
        gdf_squares['ratio_pres'] =yy['ratio_pres']
        
        Shapef =gpd.sjoin(mode, gdf_squares,how='left', op='intersects')
        Shapef['ratio_pres1']=Shapef['ratio_pres'].fillna(1)
        Shapef=Shapef.to_crs("EPSG:4326") 
        
        # Define the discrete intervals for the legend
        num_intervals = 5
        intervals = np.linspace(1, max_val, num_intervals + 1)

        # Create a colormap with discrete colors
        colors = plt.cm.get_cmap(excmap, num_intervals)
        cmap_discrete = ListedColormap([colors(i) for i in range(num_intervals)])


        Shapef.plot(column='ratio_pres1',ax=axs[j, 0],  cmap=cmap_discrete, legend=False, markersize=0.7, vmin=1, vmax=max_val)
        Shapef.plot(column='ratio_pres',ax=axs[j, 0],  cmap=cmap_discrete, legend=False, markersize=0.7, vmin=1, vmax=max_val)
        
        axs[j, 0].set_aspect('equal', adjustable='box')
        #axs[j, 0].set_title(f'{tr_names[j]}', loc='left', fontsize=13)
        axs[j, 0].set_xlim([-15, 35])
        axs[j, 0].set_ylim([30, 75])
        
                # Hide spines and tick labels
        axs[j, 0].spines['top'].set_visible(False)
        axs[j, 0].spines['right'].set_visible(False)
        axs[j, 0].spines['left'].set_visible(False)
        axs[j, 0].spines['bottom'].set_visible(False)
        axs[j, 0].xaxis.set_visible(False)
        axs[j, 0].yaxis.set_visible(False)


        # Plot the base map (Europe) for the second column
        eu_coast.plot(ax=axs[j, 1], color='lightgrey')
        Europe.plot(ax=axs[j, 1], color='darkgrey')
        
        
        # Plot the data for 2075
        x2 = Median_RCP_2075[trmode][var_sel].sel(rcp=ii)
        yy2=x2.to_dataframe()
        yy2 = yy2.reset_index()
        gdf_squares['ratio_pres'] =yy2['ratio_pres']
        max_val_2075 = x2.where(x2 < 80).max()
        #max_val_2075 = 5
        
        Shapef2 =gpd.sjoin(mode, gdf_squares,how='left', op='intersects')       
        Shapef2['ratio_pres1']=Shapef2['ratio_pres'].fillna(1)
        Shapef2.plot(column='ratio_pres1',ax=axs[j, 1],  cmap=cmap_discrete, legend=False, markersize=0.7, vmin=1, vmax=max_val)
        Shapef2.plot(column='ratio_pres',ax=axs[j, 1],  cmap=cmap_discrete, legend=False, markersize=0.7, vmin=1, vmax=max_val)
        Shapef2=Shapef2.to_crs("EPSG:4326") 
        
        axs[j, 1].set_aspect('equal', adjustable='box')
        #axs[j, 1].set_title(f'{trmode}')
        axs[j, 1].set_xlim([-15, 35])
        axs[j, 1].set_ylim([30, 75])
        
                # Hide spines and tick labels
        axs[j, 1].spines['top'].set_visible(False)
        axs[j, 1].spines['right'].set_visible(False)
        axs[j, 1].spines['left'].set_visible(False)
        axs[j,1].spines['bottom'].set_visible(False)
        axs[j, 1].xaxis.set_visible(False)
        axs[j, 1].yaxis.set_visible(False)
        
        # Add subplot title in the middle of the row
        fig.text(0.5, 0.9 - j * 0.40, f'{tr_names[j]}', ha='center', fontsize=13)


    # Add a color bar at the end
    cax = fig.add_axes([0.2, 0.085, 0.6, 0.014])  # Adjust the position and size as needed
    norm = plt.Normalize(vmin=1, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap_discrete, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='max')
    cbar.set_label('Exposure multiplication factor')
    
    plt.savefig(plot_dir +"Compos_RCP_"+str(ii) + flags['extr']+"_TH" + str(TH_val)+ ".png", dpi=900)
    
   
#%%%%%%%%%    

# Define the discrete colormap with specified boundaries
    
