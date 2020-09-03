"""

compute normalization weights of input data using xarray and dask. This can be done on a single
node, because dask candeal with data that is larger then available RAM
"""


import os

import xarray as xr
import numpy as np
import dask as da

from dask.diagnostics import ProgressBar
ProgressBar().register()

basepath = '/proj/bolinc/users/x_sebsc/weather-benchmark/2.5deg/'
outfolder = '/proj/bolinc/users/x_sebsc/nn_/benchmark/normalization_weights/'

os.system(f'mkdir -p {outfolder}')

modelname='era5_2.5deg'
train_startyear=1979
train_endyear=2016
time_resolution_hours = 6



variables = ['geopotential_500']

outname=f'{outfolder}/normalization_weights_{modelname}_{train_startyear}-{train_endyear}_tres{time_resolution_hours}_' \
    + '_'.join([str(e) for e in variables])

def read_e5_data(startyear, endyear, time_resolution_hours=1,
                 variables='all'):
    all_variables = ['2m_temperature', 'mean_sea_level_pressure',
                     '10m_u_component_of_wind', '10m_v_component_of_wind']
    years = list(range(startyear, endyear+1))
    if variables=='all':
        variables = all_variables

    combined_ds = []
    for variable in variables:
        ifiles = [f'{basepath}/{variable}/{variable}_{year}_2.5deg.nc' for year in years]
        ds = xr.open_mfdataset(ifiles, chunks={'time':1}) # we need to chunk time by 1 to get efficient
        # reading of data whenrequesting lower time-resolution
        # this is now a dataset. we want to have a dataarray
        da = ds.to_array()
        # now the first dimension is the variable dimension, which should have length 1 (because we have only
        # 1 variable per file
        assert(da.shape[0]==1)
        # remove this dimension
        da = da.squeeze()
        # now there are 2 possibilities. If we loaded data on mulciple pressure levels, then there
        # is a "lev" dimension. if we loaded surface variables, we dont have this dimensions, but we need it.
        if len(da.shape) == 4:
            # case with pressure level present
            combined_ds.append(da)
        elif len(da.shape) == 3:
            da = da.expand_dims(dim='lev', axis=1)
            combined_ds.append(da)
        else:
            raise Exception("we should never get here, data has wrong dimension")

    ds = xr.concat(combined_ds, dim='lev')

    # for kears/tensorflow, we need to have the lev dimension as last one
    ds = ds.transpose('time','lat','lon','lev')

    # now reduce to desired time-resolution
    ds = ds[::time_resolution_hours]
    return ds

ds = read_e5_data(train_startyear, train_endyear, time_resolution_hours=time_resolution_hours,
                  variables=variables)

# to avoid precision issues, convert to float64
ds = ds.astype('float64')
norm_mean = ds.mean(('time','lat','lon'))
norm_std = ds.std(('time','lat','lon'))
# conver back to float32
norm_mean = norm_mean.astype('float32')
norm_std = norm_std.astype('float32')
norm_mean.to_netcdf(outname+'_mean.nc')
norm_std.to_netcdf(outname+'_std.nc')

