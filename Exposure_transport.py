# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:05:25 2023

@author: Cristina

Functions for Transport Mode extreme event exposure

"""

from rasterio.features import geometry_mask
from shapely.geometry import mapping
from pyproj import Transformer
from operator import index
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import pickle as pk
from scipy import interpolate
from scipy import stats as sts
import regionmask as rm
import glob
import time
import matplotlib.pyplot as plt
from copy import deepcopy as cp
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

rcp_list=[26,60,85]
tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

#%% ----------------------------------------------------------------   
# Function to open isimip data array and read years from filename
# (the isimip calendar "days since 1661-1-1 00:00:00" cannot be read by xarray datetime )
# this implies that years in file need to correspond to years in filename
def open_dataarray_isimip(file_name): 
    
    begin_year = int(file_name.split('_')[-2])
    end_year = int(file_name.split('_')[-1].split('.')[0])
    
    # some files contain extra var 'time_bnds', first try reading for single var
    try:
        
        da = xr.open_dataarray(file_name, decode_times=False)
        
    except:
        
        da = xr.open_dataset(file_name, decode_times=False).exposure
    
    da['time'] = np.arange(begin_year,end_year+1)
    
    return da

#%% ----------------------------------------------------------------
# Load ISIMIP model data
def load_isimipdata(
    extremes, 
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,
    df_GMT_strj,
    flags,
): 
    
    if flags['run']: 

        print('Processing isimip')

        # initialise counter, metadata dictionary, pic list, pic meta, and 
        i = 1
        d_isimip_meta = {}
        pic_list = []
        #d_pic_meta = {}
        GMT_rcp= {}
        
        # loop over extremes
        for extreme in extremes:

            # define all models
            models = model_names[extreme]

            # loop over models
            for model in models: 

                # store all files starting with model name
                file_names = sorted(glob.glob('./data/isimip/'+flags['extr']+'/'+model.lower()+'/'+model.lower()+'*rcp*landarea*2099*'))

                for file_name in file_names: 

                    print('Loading '+file_name.split('\\')[-1]+' ('+str(i)+')')

                    # load rcp data (AFA: Area Fraction Affected) - and manually add correct years
                    da_AFA_rcp = open_dataarray_isimip(file_name)

                    # save metadata
                    d_isimip_meta[i] = {
                        'model': file_name.split('_')[0].split('\\')[-1],
                        'gcm': file_name.split('_')[1],
                        'rcp': file_name.split('_')[2],
                        'extreme': file_name.split('_')[3]
                    }

                    #load associated historical variable
                    file_name_his = glob.glob('./data/isimip/'+flags['extr']+'/'+model.lower()+'/'+model.lower()+'*'+d_isimip_meta[i]['gcm']+'*hist*landarea*')[0]
                    da_AFA_his = open_dataarray_isimip(file_name_his)

                    # concatenate historical and future data
                    da_AFA = xr.concat([da_AFA_his,da_AFA_rcp], dim='time')
                    
                    
                    # # load GMT for rcp and historical period - note that these data are in different files
                    #file_names_gmt = glob.glob('./data/isimip/DerivedInputData/globalmeans/tas/'+d_isimip_meta[i]['gcm'].upper()+'/'+'*.fldmean.yearmean.txt') # ignore running mean files
                    
                    file_names_gmt = glob.glob('./data/isimip/DerivedInputData/globalmeans/tas/'+'*FirstYearOverThreshold.csv') # ignore running mean files
                    
                    # Create a list with GMT values and years of crossing it for each rcp. The file are one for each model
                    
                    for rcp in rcp_list:
                    
                        file_name_gmt = glob.glob('./data/isimip/DerivedInputData/globalmeans/tas/*'+str(rcp)+'*.csv') 
                    
                        GMT_rcp[rcp]= pd.read_csv(file_name_gmt[0]).set_index('threshold')
  
                        
                    # Pi-control data
                    # adding this to avoid duplicates of da_AFA_pic in pickles
                    # if '{}_{}'.format(d_isimip_meta[i]['model'],d_isimip_meta[i]['gcm']) not in pic_list:

                    #     # load associated picontrol variables (can be from up to 4 files)
                    #     file_names_pic  = glob.glob('./data/isimip/'+flags['extr']+'/'+model.lower()+'/'+model.lower()+'*'+d_isimip_meta[i]['gcm']+'*_picontrol_*landarea*')

                    #     if  isinstance(file_names_pic, str): # single pic file 
                    #         da_AFA_pic  = open_dataarray_isimip(file_names_pic)
                    #     else: # concat pic files
                    #         das_AFA_pic = [open_dataarray_isimip(file_name_pic) for file_name_pic in file_names_pic]
                    #         da_AFA_pic  = xr.concat(das_AFA_pic, dim='time')
                            
                    #     # save AFA field as pickle
                    #    with open('./data/pickles/{}/isimip_AFA_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],str(i)), 'wb') as f: # added extreme to string of pickle
                     #        pk.dump(da_AFA_pic,f)
                            
                    #     pic_list.append('{}_{}'.format(d_isimip_meta[i]['model'],d_isimip_meta[i]['gcm']))
                        
                        # # save metadata for picontrol
                        # d_pic_meta[i] = {
                        #     'model': d_isimip_meta[i]['model'], 
                        #     'gcm': d_isimip_meta[i]['gcm'],              
                        #     'extreme': file_name.split('_')[3], 
                        #     'years': str(len(da_AFA_pic.time)),
                        # }
                            
                    # save AFA field as pickle
                    with open('./data/pickles/{}/isimip_AFA_{}_{}.pkl'.format(flags['extr'],flags['extr'],str(i)), 'wb') as f: # added extreme to string of pickle
                        pk.dump(da_AFA,f)

                    # update counter
                    i += 1
        
            # save metadata dictionary as a pickle
            print('Saving metadata')
            with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],flags['gmt'],flags['rm']), 'wb') as f:
                pk.dump(d_isimip_meta,f)
            with open('./data/pickles/GMT_rcp.pkl', 'wb') as f:
                pk.dump(GMT_rcp,f)
            #with open('./data/pickles/{}/isimip_pic_metadata_{}.pkl'.format(flags['extr'],flags['extr']), 'wb') as f:
            #   pk.dump(d_pic_meta,f)

    else: 
        
        # loop over extremes
        print('Loading processed isimip data')
        # loac pickled metadata for isimip and isimip-pic simulations

        with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)
       # with open('./data/pickles/{}/isimip_pic_metadata_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
       #     d_pic_meta = pk.load(f)        
        with open('./data/pickles/GMT_rcp.pkl', 'rb') as f:
            GMT_rcp = pk.load(f)      
            
    return d_isimip_meta, GMT_rcp

#%% ----------------------------------------------------------------

# convert Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span
def calc_exposure_for_tr_modes(
    d_isimip_meta,
    year_range,
    grid_EU,
    flags ):

   
#%% 
    #tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']
    Europe=gpd.read_file('./data/EU_shape/Europe.shp')  
    #NUTS2=gpd.read_file('./data/EU_shape/NUTS2.shp')
     
    # Selecting the mode of transport
    
    print('Loading processed isimip data')
    # loac pickled metadata for isimip and isimip-pic simulations

    with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
        d_isimip_meta = pk.load(f)
  #  with open('./data/pickles/{}/isimip_pic_metadata_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
    #    d_pic_meta = pk.load(f)  

    mode_datasets = {}
  
    
    for trmode in tr_modes: 

    # load AFA data of that run
    
        print('Selected transport mode:{}'.format(trmode)  )
        #print('Selected transport mode:{}'.format(flags['mode']))    
        
        mode_shap='./data/TENT_shapefiles/{}_GL2017_EU.shp'.format(trmode)
        mode = gpd.read_file(mode_shap)    
        
                
        ## Dataset for exposure values for each climate model ########
    
        ds_e = xr.Dataset(
        data_vars={
            'exposure': (['run', 'time', 'lat', 'lon'], np.full((len(list(d_isimip_meta.keys())), len(year_range), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
        },
        coords={
            'run': ('run', list(d_isimip_meta.keys())),
            'time': year_range,
            'lat': grid_EU.lat,  # Replace with your actual latitudes
            'lon': grid_EU.lon,  # Replace with your actual longitudes
            }
            )
            
        # loop over simulations
        for i in list(d_isimip_meta.keys()): 
    
            print('simulation {} of {}'.format(i,len(d_isimip_meta)))
    
            
            # load AFA data of that run
            with open('./data/pickles/{}/isimip_AFA_{}_{}.pkl'.format(flags['extr'],flags['extr'],str(i)), 'rb') as f:
                da_AFA = pk.load(f)  
            
            da_AFA.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            da_AFA.rio.write_crs("epsg:4326", inplace=True)
            clip = da_AFA.rio.clip(mode.geometry.apply(mapping), mode.crs, drop=False)
            
            # Clip just in Europe
            lon_min, lon_max = -35.0, 40.0
            lat_min, lat_max = 25.0, 72.0
    
            # Clip the dataset to the specified latitude and longitude bounds
            ds_europe = clip.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))
            
            # ADD FLAG
            # Save pickles
            with open('./data/pickles/{}/Transport_modes/{}_isimip_AFA_{}_{}.pkl'.format(flags['extr'],str(trmode),flags['extr'],str(i)), 'wb') as f:
               pk.dump(ds_europe,f) 
            
            # Fill the dataset
            
            ds_e['exposure'].loc[{'run': i}] = ds_europe
            
        
        mode_datasets[trmode] = ds_e     
        

        # Save pickles
    with open('./data/pickles/{}/Transport_modes/0_Allmode_dataset_{}.pkl'.format(flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(mode_datasets,f)     
                                          
                              
    return mode_datasets

#%%##########################################################################################################################################################
# Calculate probability of occurrence

def pb_exposure_allmodes(
    time_window,
    y_selected,
    all_modes,
    d_isimip_meta,
    grid_EU):
    
    y_past=1861
    y_pres=2005
    time_half=(time_window-1)/2
    
    ref_past= y_past,y_past+time_window
    ref_pres= y_pres-time_window,y_pres
    ref_fut= round(y_selected-time_half-1),round(y_selected+time_half)
    tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']
    
    
    mode_ratio = {}
  
    for trmode in tr_modes: 
    
        ds=all_modes[trmode]['exposure']
        
        ## Dataset for exposure values for each climate model ########
    
        ds_ratio = xr.Dataset(
        data_vars={
            'ratio_pres': (['run', 'lat', 'lon'], np.full((len(list(d_isimip_meta.keys())), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'ratio_past': (['run', 'lat', 'lon'], np.full((len(list(d_isimip_meta.keys())), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'pb_past': (['run', 'lat', 'lon'], np.full((len(list(d_isimip_meta.keys())), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'pb_pres': (['run', 'lat', 'lon'], np.full((len(list(d_isimip_meta.keys())), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'pb_fut': (['run', 'lat', 'lon'], np.full((len(list(d_isimip_meta.keys())), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'x_fut': (['run', 'lat', 'lon'], np.full((len(list(d_isimip_meta.keys())), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'x_past': (['run', 'lat', 'lon'], np.full((len(list(d_isimip_meta.keys())), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'x_pres': (['run', 'lat', 'lon'], np.full((len(list(d_isimip_meta.keys())), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
        },
        coords={
            'run': ('run', list(d_isimip_meta.keys())),
            #'time': np.arange(year_start, year_end + 1, 1),
            'lat': grid_EU.lat,  # Replace with your actual latitudes
            'lon': grid_EU.lon,  # Replace with your actual longitudes
            }
            )
        
          ##  
        
        for ii in ds.run:
            
            ds_run=ds.sel(run=ii)
                            
            subset_future = ds_run.sel(time=slice(ref_fut[0], ref_fut[1]))
            x_future=subset_future.sum(dim=('time'), skipna=False)
            pb_fut=x_future/len(subset_future.time)
            
            
            subset_pres = ds_run.sel(time=slice(ref_pres[0], ref_pres[1]))
            x_pres=subset_pres.sum(dim=('time'), skipna=False)
            pb_pres=x_pres/len(subset_pres.time)
            
            subset_past = ds_run.sel(time=slice(ref_past[0], ref_past[1]))
            x_past=subset_past.sum(dim=('time'), skipna=False)
            pb_past=x_past/len(subset_past.time)
            
            #save probability
            ds_ratio['pb_fut'].loc[{'run': ii}] = pb_fut
            ds_ratio['pb_past'].loc[{'run': ii}] = pb_past            
            ds_ratio['pb_pres'].loc[{'run': ii}] = pb_pres            
            
            #save nb events
            ds_ratio['x_fut'].loc[{'run': ii}] = x_future
            ds_ratio['x_past'].loc[{'run': ii}] = x_past            
            ds_ratio['x_pres'].loc[{'run': ii}] = x_pres                

            #save moltiplicator factor
            ds_ratio['ratio_past'].loc[{'run': ii}] = x_future/x_past
            ds_ratio['ratio_pres'].loc[{'run': ii}] = x_future/x_pres        
            
            
        mode_ratio[trmode]=ds_ratio

    return mode_ratio, ref_past, ref_pres, ref_fut


#%%##########################################################################################################################################################
# Calculate statistics

def EMF_stats(
    ratio_chosen, variab, keys_rcp,d_isimip_meta,grid_EU):
    
    tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

    Statistics_rcp = {}
    
   # keys_rcp26 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp26']
   # keys_rcp60 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp60']
   # keys_rcp85 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp85']
    

    for trmode in tr_modes: 
        
    ## Dataset for exposure values for each climate model ########
        
        rcp_stat = xr.Dataset(
        data_vars={
            'median': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'q02': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'q08': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)), 
            'q025': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'q075': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)), 
        },
        coords={
          #  'rcp': [26,60,85],
            'lat': grid_EU.lat,  # Replace with your actual latitudes
            'lon': grid_EU.lon,  # Replace with your actual longitudes
            }
            )
        
        
        rcp_stat['median']=ratio_chosen[trmode][variab].sel(run=keys_rcp).median(dim='run')
        
        mat_clean=ratio_chosen[trmode][variab].fillna(1)
        data_clean=mat_clean.groupby('run').map(lambda x: x.where(~np.isinf(x), 100000))
        
        
        rcp_stat['min']=mat_clean.sel(run=keys_rcp).min(dim='run') 
        rcp_stat['max']=mat_clean.sel(run=keys_rcp).max(dim='run')
        
        
        rcp_stat['q02']=data_clean.sel(run=keys_rcp).quantile([0.2], dim='run')   
        rcp_stat['q08']=data_clean.sel(run=keys_rcp).quantile([0.8], dim='run')  
        
        rcp_stat['q05']=data_clean.sel(run=keys_rcp).quantile([0.5], dim='run') 
        
        rcp_stat['q025']=data_clean.sel(run=keys_rcp).quantile([0.25], dim='run')   
        rcp_stat['q075']=data_clean.sel(run=keys_rcp).quantile([0.75], dim='run')   
        
        count_100 = (rcp_stat['median'] > 100).sum().item()
        count_0 = (rcp_stat['median'] == 0).sum().item()
        
        perc_100=count_100/len(rcp_stat['median'])*100
        perc_0=count_0/len(rcp_stat['median'])*100
        
        medmed= rcp_stat['median'].median(dim=['lon', 'lat']).item()
        Q2med= rcp_stat['q02'].median(dim=['lon', 'lat']).item()
        Q8med= rcp_stat['q08'].median(dim=['lon', 'lat']).item()
        

        
        print(extreme, trmode,': Median is:', medmed, ', count100 is:', count_100, ', count0 is:', count_0,
             ', Q 0.2 is:' , Q2med,  ', Q 0.8 is:', Q8med)

#       rcp_stat['mean']=ratio_chosen[trmode][variab].sel(run=keys_rcp).quantile([0.2], dim='run')   
#       rcp_stat['q08']=ratio_chosen[trmode][variab].sel(run=keys_rcp).dropna([0.8], dim='run') 

  
                
        Statistics_rcp[trmode] = rcp_stat
    
    return Statistics_rcp


#%%##########################################################################################################################################################

#%% ----------------------------------------------------------------

#rcp_list=26,60,85 

# median of EMF over the rcp and all modes

def median_ratio_rcp_allmodes(
    rcp_list,
    ratio_chosen,d_isimip_meta,grid_EU):
    
    tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']
    
    keys_rcp26 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp26']
    keys_rcp60 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp60']
    keys_rcp85 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp85']
    
    Ratio_mode_rcp_median = {}
    

    for trmode in tr_modes: 
        
    ## Dataset for exposure values for each climate model ########
    
        # rcp_mean = xr.Dataset(
        # data_vars={
        #     'median_exp': (['rcp', 'time', 'lat', 'lon'], np.full((len(rcp_list), len(year_range), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
        # },
        # coords={
        #     'rcp': [26,60,85],
        #     'time': np.arange(year_start, year_end + 1, 1),
        #     'lat': grid_EU.lat,  # Replace with your actual latitudes
        #     'lon': grid_EU.lon,  # Replace with your actual longitudes
        #     }
        #     )
        
        rcp_median = xr.Dataset(
        data_vars={
            'ratio_pres': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'ratio_past': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'pb_past': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'pb_pres': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'pb_fut': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'x_fut': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'x_past': (['rcp', 'lat', 'lon'],  np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'x_pres': (['rcp', 'lat', 'lon'],  np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            
        },
        coords={
            'rcp': [26,60,85],
            'lat': grid_EU.lat,  # Replace with your actual latitudes
            'lon': grid_EU.lon,  # Replace with your actual longitudes
            }
            )
        
        ## Doing the median over the models for selected rcp
        
        median_rcp26=ratio_chosen[trmode].sel(run=keys_rcp26).median(dim='run')
        median_rcp60=ratio_chosen[trmode].sel(run=keys_rcp60).median(dim='run')
        median_rcp85=ratio_chosen[trmode].sel(run=keys_rcp85).median(dim='run')

        # median_rcp26=ratio_chosen[trmode]['ratio'].sel(run=keys_rcp26).median(dim='run')
        # median_rcp60=ratio_chosen[trmode]['ratio'].sel(run=keys_rcp60).median(dim='run')
        # median_rcp85=ratio_chosen[trmode]['ratio'].sel(run=keys_rcp85).median(dim='run')
        
     
        # Use loc to assign data to the specific 'rcp' coordinate
        rcp_median.loc[{'rcp': 26}]=median_rcp26
        rcp_median.loc[{'rcp': 60}]=median_rcp60        
        rcp_median.loc[{'rcp': 85}]=median_rcp85        
        
        #Calculate the ratio for present and future
        #rcp_median['ratio_pres']=rcp_median.pb_fut/rcp_median.pb_pres
        #rcp_median['ratio_past']=rcp_median.pb_fut/rcp_median.pb_past
      
        #Setting probability ratio =100 if infinite, exposure moltiplication factor
        rcp_median['ratio_pres'] = xr.where(np.isinf(rcp_median.ratio_pres), 100, rcp_median.ratio_pres)
        rcp_median['ratio_past'] = xr.where(np.isinf(rcp_median.ratio_past), 100, rcp_median.ratio_past)
            
       # ds_ratio['ratio_pres'].loc[{'run': ii}] = ratio_pres
        #ds_ratio['ratio_past'].loc[{'run': ii}] = ratio_past
        
        
        Ratio_mode_rcp_median[trmode] = rcp_median
    
    return Ratio_mode_rcp_median

#%%
# quantile of EMF over the rcp and all modes

def quant_ratio_rcp_allmodes(
    rcp_list,
    ratio_chosen,d_isimip_meta,grid_EU, qval):
    
    tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']
    
    keys_rcp26 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp26']
    keys_rcp60 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp60']
    keys_rcp85 = [key for key, value in d_isimip_meta.items() if value['rcp'] == 'rcp85']
    
    Ratio_mode_rcp_quant = {}
    

    for trmode in tr_modes: 
        
    ## Dataset for exposure values for each climate model ########
           
        rcp_quant = xr.Dataset(
        data_vars={
            'ratio_pres': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            
        },
        coords={
            'rcp': [26,60,85],
            'lat': grid_EU.lat,  # Replace with your actual latitudes
            'lon': grid_EU.lon,  # Replace with your actual longitudes
            }
            )
        
        ## Doing the quantiles over the models for selected rcp
        
        data_sel= ratio_chosen[trmode]['ratio_pres']
        
        #Clean the infinite since it is not supported for the quantile calculation        
        data_clean=data_sel.groupby('run').map(lambda x: x.where(~np.isinf(x), 100000))
  
        # Calculate quantile for selected one
        quant_rcp26=data_clean.sel(run=keys_rcp26).quantile(qval, dim='run', skipna=True )
        quant_rcp60=data_clean.sel(run=keys_rcp60).quantile(qval, dim='run', skipna=True )
        quant_rcp85=data_clean.sel(run=keys_rcp85).quantile(qval, dim='run', skipna=True )

        
     
        # Use loc to assign data to the specific 'rcp' coordinate
        rcp_quant['ratio_pres'].loc[{'rcp': 26}]=quant_rcp26
        rcp_quant['ratio_pres'].loc[{'rcp': 60}]=quant_rcp60        
        rcp_quant['ratio_pres'].loc[{'rcp': 85}]=quant_rcp85        
        

        # Add in the big xarray
        Ratio_mode_rcp_quant[trmode] = rcp_quant
    
    return Ratio_mode_rcp_quant

#%% ----------------------------------------------------------------


def median_CI_allmodes(
    ratio_chosen,keys_rcp,d_isimip_meta,grid_EU):
    
    tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

    Statistics_rcp_median = {}
    

    for trmode in tr_modes: 
        
    ## Dataset for exposure values for each climate model ########
        
        rcp_stat = xr.Dataset(
        data_vars={
            'median': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'q25': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'q975': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),          
        },
        coords={
          #  'rcp': [26,60,85],
            'lat': grid_EU.lat,  # Replace with your actual latitudes
            'lon': grid_EU.lon,  # Replace with your actual longitudes
            'time': np.arange(year_start, year_end + 1, 1),
            }
            )
          
        
        rcp_stat['median']=ratio_chosen[trmode].exposure.sel(run=keys_rcp).median(dim='run')
        rcp_stat['q25']=ratio_chosen[trmode].exposure.sel(run=keys_rcp).quantile([0.025], dim='run')   
        rcp_stat['q975']=ratio_chosen[trmode].exposure.sel(run=keys_rcp).quantile([0.975], dim='run')         
        
                
        Statistics_rcp_median[trmode] = rcp_stat
    
    return Statistics_rcp_median

#%%

def median_CI_allmodes(
    ratio_chosen,keys_rcp,d_isimip_meta,grid_EU):
    
    tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

    Statistics_rcp_median = {}
    

    for trmode in tr_modes: 
        
    ## Dataset for exposure values for each climate model ########
        
        rcp_stat = xr.Dataset(
        data_vars={
            'median': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'q25': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'q975': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),          
        },
        coords={
          #  'rcp': [26,60,85],
            'lat': grid_EU.lat,  # Replace with your actual latitudes
            'lon': grid_EU.lon,  # Replace with your actual longitudes
            'time': np.arange(year_start, year_end + 1, 1),
            }
            )
          
        
        rcp_stat['median']=ratio_chosen[trmode].exposure.sel(run=keys_rcp).median(dim='run')
        rcp_stat['q25']=ratio_chosen[trmode].exposure.sel(run=keys_rcp).quantile([0.025], dim='run')   
        rcp_stat['q975']=ratio_chosen[trmode].exposure.sel(run=keys_rcp).quantile([0.975], dim='run')         
        
                
        Statistics_rcp_median[trmode] = rcp_stat
    
    return Statistics_rcp_median

#%%

def median_CI_allmodes(
    ratio_chosen,keys_rcp,d_isimip_meta,grid_EU):
    
    tr_modes=['airports','ports','railways','roads',"rrt",'urban_nodes','iww']

    Statistics_rcp_median = {}
    

    for trmode in tr_modes: 
        
    ## Dataset for exposure values for each climate model ########
        
        rcp_stat = xr.Dataset(
        data_vars={
            'median': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'q25': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),
            'q975': (['rcp', 'lat', 'lon'], np.full((len(rcp_list), len(grid_EU.lat), len(grid_EU.lon)), fill_value=np.nan)),          
        },
        coords={
          #  'rcp': [26,60,85],
            'lat': grid_EU.lat,  # Replace with your actual latitudes
            'lon': grid_EU.lon,  # Replace with your actual longitudes
            'time': np.arange(year_start, year_end + 1, 1),
            }
            )
          
        
        rcp_stat['median']=ratio_chosen[trmode].exposure.sel(run=keys_rcp).median(dim='run')
        rcp_stat['q25']=ratio_chosen[trmode].exposure.sel(run=keys_rcp).quantile([0.025], dim='run')   
        rcp_stat['q975']=ratio_chosen[trmode].exposure.sel(run=keys_rcp).quantile([0.975], dim='run')         
        
                
        Statistics_rcp_median[trmode] = rcp_stat
    
    return Statistics_rcp_median






#%%

#plt.figure()
#Median_RCP_2050['airports']['ratio_pres'].loc[{'rcp': 85}].plot()

#%%

# M10=rcp_mean.loc[{'rcp': 26}]
# M1d=M10.to_dataframe()


# M12=rcp_mean.loc[{'rcp': 60}]
# M2d=M12.to_dataframe()

# M13=rcp_mean.loc[{'rcp': 85}]
# M3d=M13.to_dataframe()

# #%% ----
# plt.figure()
# M1=rcp_mean['ratio_pres'].loc[{'rcp': 26}]
# plt.title('Pb 2.6') 
# M1.plot()

# plt.figure()
# M1=rcp_mean['ratio_pres'].loc[{'rcp': 60}]
# plt.title('Pb 6.0') 
# M1.plot()

# plt.figure()
# M1=rcp_mean['ratio_pres'].loc[{'rcp': 85}]
# plt.title('Pb 8.5') 
# M1.plot()

# #%% ----

# plt.figure()
# mean_rcp26.plot(vmin=0, vmax=1)
# plt.title('Pb 2.6') 

# plt.figure()
# mean_rcp60.plot(vmin=0, vmax=1)
# plt.title('Pb 6') 


# plt.figure()
# mean_rcp85.plot(vmin=0, vmax=1)
# plt.title('Pb 8.5') 


# #%% 

# plt.figure()
# mean_rcp26['ratio_pres'].plot()

# plt.figure()
# uu.plot()

# #%% 
# uu=ratio_chosen[trmode]['ratio_pres'].sel(run=keys_rcp60).mean(dim='run')

# plt.figure()
# mean_rcp26['ratio_pres'].plot()

# plt.figure()
# uu.plot()

# #%% 
# plt.figure()
# rcp_mean['mean_exp'].isel(rcp=0,time=1).plot()

# plt.figure()
# rcp_mean['mean_exp'].isel(rcp=0,time=200).plot()

# plt.figure()
# rcp_mean['mean_exp'].isel(rcp=0,time=1).plot()

# plt.figure()
# rcp_mean['mean_exp'].isel(rcp=0,time=200).plot()

# plt.figure()
# rcp_mean['mean_exp'].isel(rcp=1,time=100).plot()

# plt.figure()
# rcp_mean['mean_exp'].isel(rcp=2,time=100).plot()

#%% ----

# year_sel=1960,2100

# for ii in [0,1,2]:
    
#     plt.figure()
#     Europe.plot(color='navajowhite')
#     x1=rcp_mean['mean_exp'].isel(rcp=ii,time=1)
#     x1.plot(vmin=0, vmax=1)
#     plt.savefig(plot_dir+trmode+' ipc='+str(ii)+'year'+str(1)+".png", dpi=900)
    
#     plt.figure()
#     Europe.plot(color='navajowhite')
#     x2=rcp_mean['mean_exp'].isel(rcp=ii,time=100)
#     x2.plot(vmin=0, vmax=1)
#     plt.savefig(plot_dir+trmode+' ipc='+str(ii)+'year'+str(100)+".png", dpi=900)



    
#     for tt in year_sel:
#         x1=rcp_mean['mean_exp'].sel(time=tt,rcp=ii)

#         plt.figure()
#         Europe.plot(color='navajowhite')
#         x1.plot(vmin=0, vmax=1)
#         plt.title(trmode+' ipc='+str(ii)+'- year:'+str(tt))
#         plt.savefig(plot_dir+trmode+' ipc='+str(ii)+'year'+str(tt)+".png", dpi=900)
        
        
        

# plt.figure()
# rcp_mean['mean_exp']['rcp'==60].isel(time=100).plot()

# plt.figure()
# rcp_mean['mean_exp']['rcp'==85].isel(time=100).plot()


# def calc_exposure_for_tr_modes(
#     d_isimip_meta,
#     year_range,
#     grid_EU,
#     flags ):
 
#     for key, value in d_isimip_meta.items():
#     if value['rcp'] == 'rcp85':
#         print(f"Element {key}")    
    
#     mean_exposure[trmode] = ds_e['exposure'].mean(dim='run')
#         with open('./data/pickles/{}/Transport_modes/1_Allmode_mean_exposure_{}.pkl'.format(flags['extr'],flags['extr']), 'wb') as f:
#            pk.dump(mean_exposure,f)     

#         # Create an empty list to store extracted data
#         extracted_data = []
        
#         for index, point in mode.iterrows():
#             # Get the coordinates of the point
#             lon, lat = point.geometry.x, point.geometry.y
        
#             # Use xarray's `.sel()` method to extract data at the point's location
#             extracted_value = ds_europe.sel(lat=lat, lon=lon, method='nearest')
        
#             # Append the extracted data to the list
#             extracted_data.append({
#                 'PointID': point['ID'],  # Customize this based on your shapefile's attribute
#                 'Latitude': lat,
#                 'Longitude': lon,
#                 'ExtractedData': extracted_value,
#                 "DESCRIPTIO":point['DESCRIPTIO']
#             })