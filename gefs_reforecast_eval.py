
"""

TODO: check [rpb;e,m with lead time at ltime=4
"""

import xarray as xr
import numpy as np
import properscoring
import pickle

from dask.diagnostics import ProgressBar
import pandas as pd
ProgressBar().register()
ifile = '/climstorage/sebastian/nn_ensemble_nwp/gefs_reforecast/hgt_pres_latlon_all_20170101_20181231_NH.nc'
data = xr.open_dataset(ifile, chunks={'time':100})['Geopotential_height']

data = data.sel(time=slice('20170101', '20181231'))
# remove empty pressure dimension
data = data.squeeze()

# we only want fulls of 24 h as leadtime( because for this we have the gefs reforecast analysis as "truth")
data = data.sel(fhour=[pd.to_timedelta(str(d)+'d') for d in range(0,6)])

data.load()
# first "member" is ctrl
truth_all = data.sel(fhour=0,ens=0)


for n_ens in range(1,10+1):

    leadtimes = []
    res_mse = []
    res_ensvar = []
    res_crps = []
    res_ctrl_mse = []
    for ifhour in range(len(data.fhour)):
        print(ifhour)
        fc = data.isel(fhour=ifhour)
        fc.load()
        # 'time' is init titme, change to init time
        fc['time'] = fc['time'] + fc['fhour']

        if ifhour > 0:
            fc = fc[:-ifhour]
            truth = truth_all[ifhour:]
        else:
            truth = truth_all
        assert(len(fc)==len(truth))
        # remove ctrl member
        fc_ens = fc.isel(ens=range(1,1+n_ens))
        fc_ensmean = fc_ens.mean('ens')
        mse_per_fc = ((truth-fc_ensmean)**2).mean(('lat','lon'))
        spread_per_fc = fc_ens.std('ens').mean(('lat','lon'))
        mse_ctrl_per_fc = ((fc.isel(ens=0)-truth)**2).mean(('lat','lon'))
        crps_per_fc = properscoring.crps_ensemble(truth, fc_ens, axis=1).mean(axis=(1,2))



        leadtimes.append(fc['fhour'].values/np.timedelta64(1,'h'))
        res_mse.append(mse_per_fc)
        res_ctrl_mse.append(mse_ctrl_per_fc)
        res_ensvar.append(spread_per_fc)
        res_crps.append(crps_per_fc)


    out = {'leadtime': leadtimes,
           'mse_ensmean': res_mse,
           'spread': res_ensvar,
           'crps': res_crps,
           'mse_ctrl': res_ctrl_mse,
           'n_ens':n_ens,
           }

    pickle.dump(out,open(f'/home/sebastian/nn_ensemble_nwp/revision1/gefs/gefs_reforecast_scores_n_ens{n_ens}.pkl','wb'))
