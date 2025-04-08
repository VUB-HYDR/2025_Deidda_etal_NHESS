# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:33:18 2024

@author: Cristina
"""

#%%%
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:49:47 2024

@author: Cristina
"""
from matplotlib import colors
import matplotlib as mpl
import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import pickle as pk



data_dir="C:/Users/Cristina/OneDrive - Vrije Universiteit Brussel/Documents/Project 2022/European Commission/Task 2/Code/lifetime_exposure_isimip-emergence"

os.chdir(data_dir)


#%%
# Directory


flags = {}
flags['extr'] = 'heatwavedarea' # 0: all
                                # 1: burntarea
                                # 3: driedarea
                                # 4: floodedarea
                                # 5: heatwavedarea
                                # 6: tropicalcyclonedarea

with open('./Preliminary_results/pickles/{}_Median_RCP_2050.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
      Median_RCP_2050 =  pk.load(f)  
      
#with open('./Preliminary_results/pickles/{}_Median_RCP_2075.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
#     Median_RCP_2075 =  pk.load(f)      


# #%%%


# # Assuming 'ratio_pres' is the variable you want to analyze
# ratio_pres_data = airports_data.ratio_pres

# ratio_pres = airports_data.ratio_pres.to_dataframe()


# x_fut_data=airports_data.x_fut.to_dataframe()
# x_pres_data=airports_data.x_pres.to_dataframe()



tr_modes=['airports','ports','railways','roads','iww']

for trmode in tr_modes:
    airports_data = Median_RCP_2050[trmode].sel(rcp=60)
    ratio_pres_data = airports_data.ratio_pres
    
    # Compute statistics
    mean_ratio_pres = ratio_pres_data.mean().values
    median_ratio_pres = ratio_pres_data.median().values
    std_ratio_pres = ratio_pres_data.std().values
    max_ratio_pres = ratio_pres_data.max().values
    min_ratio_pres = ratio_pres_data.min().values    
    q10_ratio_pres = ratio_pres_data.quantile(0.1).values
    q80_ratio_pres = ratio_pres_data.quantile(0.8).values    
        
    
    print('Extr:', flags['extr'], trmode,': Mean is',mean_ratio_pres, ', median is:', median_ratio_pres, 
          ', q10:',q10_ratio_pres, 'q80',q80_ratio_pres )

#%%%
    xx=ratio_pres_data.to_dataframe()
    plt.figure()
    plt.hist(xx.ratio_pres)
    
    tit_test=flags['extr']+'-'+trmode
    
    plt.title(tit_test)
    
    # Displaying the plot
    plt.show()  
      
#%%%


#%%%
ex_list=['heatwavedarea','floodedarea', 'driedarea', 'burntarea', 'tropicalcyclonedarea']

for extreme in ex_list:
    #Load data
    
    flags = {}
    flags['extr'] = extreme # 0: all
                                    # 1: burntarea
                                    # 3: driedarea
                                    # 4: floodedarea
                                    # 5: heatwavedarea
                                    # 6: tropicalcyclonedarea
    
    with open('./Preliminary_results/pickles/{}_Median_RCP_2050.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
          Median_RCP_2050 =  pk.load(f)  
          
    with open('./Preliminary_results/pickles/{}_Median_RCP_2075.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
          Median_RCP_2075 =  pk.load(f)  
    
    
    #Loop trmode
    
    tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']
    
    for trmode in tr_modes:
        
        #read shapefile
        mode_shap='./data/TENT_shapefiles/{}_GL2017_EU.shp'.format(trmode)
        mode = gpd.read_file(mode_shap)
        mode=mode.to_crs("EPSG:4326") 
        
        #Shapefile
        mode_data = Median_RCP_2050[trmode].sel(rcp=60).ratio_pres
        yy=mode_data.to_dataframe()
        yy = yy.reset_index()
        squares = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for _, (lat, lon) in yy[['lat', 'lon']].iterrows()]
        gdf_squares = gpd.GeoDataFrame(geometry=squares, crs='epsg:4326')
            
        gdf_squares['GridIndex']=gdf_squares.index
        gdf_squares['lat'] = yy['lat']
        gdf_squares['lon'] = yy['lon']
        gdf_squares['ratio_pres'] =yy['ratio_pres']
    
        Shapef =gpd.sjoin(mode, gdf_squares,how='left', op='intersects')
        Shapef['ratio_pres1']=Shapef['ratio_pres'].fillna(0)
        Shapef=Shapef.to_crs("EPSG:4326") 
        
        #############
        count_100 = (Shapef['ratio_pres'] == 100).sum()
        ratio_pres_data=Shapef['ratio_pres'] 
        
        # Compute statistics
        mean_ratio_pres = ratio_pres_data.mean()
        median_ratio_pres = ratio_pres_data.median()
        std_ratio_pres = ratio_pres_data.std()
        max_ratio_pres = ratio_pres_data.max()
        min_ratio_pres = ratio_pres_data.min()   
        q10_ratio_pres = ratio_pres_data.quantile(0.1)
        q90_ratio_pres = ratio_pres_data.quantile(0.9)   
        
        perc_100=count_100/len(ratio_pres_data)*100
        
        print(extreme, trmode,': Mean is',mean_ratio_pres, ', median is:', median_ratio_pres, 
              ', std', std_ratio_pres, ',max is:', max_ratio_pres,', min is:', min_ratio_pres, 
              ', q10:',q10_ratio_pres, 'q80',q90_ratio_pres, '. The number of elements for which we have 100:', perc_100 )
        
        titl= extreme+ '-'+ trmode
        plt.figure()
        ratio_pres_data.hist()
        plt.title(titl)
    
        # Show the plot
        plt.show()
#%%%


# Plot the data for 2050
x1 = Median_RCP_2050[trmode][var_sel].sel(rcp=ii)
max_val_2050 = x1.where(x1 < 80).max()
yy=x1.to_dataframe()
yy = yy.reset_index()
squares = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for _, (lat, lon) in yy[['lat', 'lon']].iterrows()]
gdf_squares = gpd.GeoDataFrame(geometry=squares, crs='epsg:4326')
    
gdf_squares['GridIndex']=gdf_squares.index
gdf_squares['lat'] = yy['lat']
gdf_squares['lon'] = yy['lon']
gdf_squares['ratio_pres'] =yy['ratio_pres']

Shapef =gpd.sjoin(mode, gdf_squares,how='left', op='intersects')
Shapef['ratio_pres1']=Shapef['ratio_pres'].fillna(0)
Shapef=Shapef.to_crs("EPSG:4326") 



#%%%

for i, ii in enumerate(ipc_list):

    # Create a figure and axis for the entire row of subplots
    fig, axs = plt.subplots(5, 2, figsize=(10, 45), sharex=True, sharey=True)
    #fig.suptitle(ipc_tit[i], fontsize=16)
    fig.subplots_adjust(hspace=0.1, wspace=0.05)
    
    for j, trmode in enumerate(tr_modes):

        eu_coast.plot(ax=axs[j, 0], color='lightgrey')
        Europe.plot(ax=axs[j, 0], color='darkgrey')
        
        mode_shap='./data/TENT_shapefiles/{}_GL2017_EU.shp'.format(trmode)
        mode = gpd.read_file(mode_shap)
        mode=mode.to_crs("EPSG:4326") 
        
        # Plot the data for 2050
        x1 = Median_RCP_2050[trmode][var_sel].sel(rcp=ii)
        max_val_2050 = x1.where(x1 < 80).max()
        yy=x1.to_dataframe()
        yy = yy.reset_index()
        squares = [box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25) for _, (lat, lon) in yy[['lat', 'lon']].iterrows()]
        gdf_squares = gpd.GeoDataFrame(geometry=squares, crs='epsg:4326')
            
        gdf_squares['GridIndex']=gdf_squares.index
        gdf_squares['lat'] = yy['lat']
        gdf_squares['lon'] = yy['lon']
        gdf_squares['ratio_pres'] =yy['ratio_pres']
        
        Shapef =gpd.sjoin(mode, gdf_squares,how='left', op='intersects')
        Shapef['ratio_pres1']=Shapef['ratio_pres'].fillna(0)
        Shapef=Shapef.to_crs("EPSG:4326") 
        
#%%%