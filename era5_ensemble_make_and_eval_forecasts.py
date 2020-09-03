#! /pfs/nobackup/home/s/sebsc/miniconda3/envs/tf2-env/bin/python

#SBATCH -A SNIC2019-3-611
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:k80:1

"""
needs tensorflow 2.0
run on kebnekaise
"""


import os
import sys
import pickle
import matplotlib
matplotlib.use('agg')
from pylab import plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
import properscoring
from dask.diagnostics import ProgressBar

ProgressBar().register()


datadir='data'
outdir='output'
os.system(f'mkdir -p {datadir}')
os.system(f'mkdir -p {outdir}')

# basepath for input data
if 'NSC_RESOURCE_NAME' in os.environ and os.environ['NSC_RESOURCE_NAME']=='tetralith':
    basepath = '/proj/bolinc/users/x_sebsc/weather-benchmark/2.5deg/'
    norm_weights_folder = '/proj/bolinc/users/x_sebsc/nn_/benchmark/normalization_weights/'
else:
    basepath = '/home/s/sebsc/pfs/weather_benchmark/2.5deg/'
    norm_weights_folder = '/home/s/sebsc/pfs/nn_ensemble/era5/normalization_weights/'


lead_time = 1   # in steps
N_gpu=0
load_data_lazily = True
modelname='era5_2.5deg_weynetal'
train_startyear=1979
train_endyear=2016
test_startyear = 2017
test_endyear = 2018
time_resolution_hours = 6
variables = ['geopotential_500']
invariants = ['lsm', 'z']
valid_split = 0.02


n_svs=100
n_ens=10 # this was only used for filename of jacobian
pert_scale=1.0 # only for filename of jacobian
n_resample = 1  # factor to reduce the output resolution/diension
svd_leadtime = int(sys.argv[1])

norm_weights_filenamebase = f'{norm_weights_folder}/normalization_weights_era5_2.5deg_{train_startyear}-{train_endyear}_tres{time_resolution_hours}_' \
    + '_'.join([str(e) for e in variables])


## parameters for the neural network

# fixed (not-tuned params)
batch_size = 32
num_epochs = 200
drop_prob=0


def read_e5_data(startyear, endyear,
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
        if not load_data_lazily:
            da.load()
        combined_ds.append(da)


    return combined_ds


# lazily load the whole dataset
ds_whole = read_e5_data(test_startyear, test_endyear, variables=variables)
# this is now a lazy dask array. do not do any operations on this array outside the data generator below.
# if we do operations before, it will severly slow down the data loading throughout the training.

# load normalization weights
norm_mean = xr.open_dataarray(norm_weights_filenamebase+'_mean.nc').values
norm_std = xr.open_dataarray(norm_weights_filenamebase+'_std.nc').values


n_data = ds_whole[0].shape[0]
N_train =  n_data// time_resolution_hours
n_valid = int(N_train * valid_split)

Nlat,Nlon,=ds_whole[0].shape[1:3]

Nlat = Nlat//2 # only NH

n_channels_out = len(variables)

n_channels_in = n_channels_out

param_string = f'{modelname}_{train_startyear}-{train_endyear}'


time_indices_all = np.arange(0,n_data-time_resolution_hours*lead_time,time_resolution_hours)
data_train = np.array(ds_whole[0][time_indices_all], dtype='float32')
data_train = (data_train - norm_mean)/norm_std

x_train = data_train[:-lead_time]
y_train = data_train[lead_time:]

# add (empty) channel dimension
x_train = np.expand_dims(x_train,axis=-1)
y_train = np.expand_dims(y_train,axis=-1)
x_train = x_train[:,-Nlat:]
y_train = y_train[:,-Nlat:]


class PeriodicPadding(keras.layers.Layer):
    def __init__(self, axis, padding, **kwargs):
        """
        layer with periodic padding for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along
        padding: number of cells to pad
        """

        super(PeriodicPadding, self).__init__(**kwargs)

        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(padding, int):
            padding = (padding,)

        self.axis = axis
        self.padding = padding

    def build(self, input_shape):
        super(PeriodicPadding, self).build(input_shape)

    # in order to be able to load the saved model we need to define
    # get_config
    def get_config(self):
        config = {
            'axis': self.axis,
            'padding': self.padding,

        }
        base_config = super(PeriodicPadding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, input):

        tensor = input
        for ax, p in zip(self.axis, self.padding):
            # create a slice object that selects everything form all axes,
            # except only 0:p for the specified for right, and -p: for left
            ndim = len(tensor.shape)
            ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
            ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
            right = tensor[ind_right]
            left = tensor[ind_left]
            middle = tensor
            tensor = tf.concat([right, middle, left], axis=ax)
        return tensor

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        for ax, p in zip(self.axis, self.padding):
            output_shape[ax] += 2 * p
        return tuple(output_shape)

# as main net we select net2
modelfile='data_mem2/trained_model_era5_2.5deg_weynetal_1979-2016_mem2.h5'
model = keras.models.load_model(modelfile,
    custom_objects={'PeriodicPadding': PeriodicPadding})

print(model.summary())



lat = ds_whole[0].lat.values[-Nlat:]


data = data_train[:,-Nlat:]
if n_channels_in == 1:
    data = np.expand_dims(data, -1)

# reduce to 2-daily resolution (to save computation time)
tres_factor = 8
data_reduced = data[::tres_factor]


svd_params = f'n_svs{n_svs}_n_ens{n_ens}_pertscale{pert_scale}_nresample{n_resample}_svdleadtime{svd_leadtime}'


def create_pertubed_states_svd(x, svecs, n_svs):

    assert(svecs[0].shape==(Nlat*Nlon*n_channels_in,))
    svecs = svecs[:n_svs]
    x_perts = []
    for i in range(n_ens//2):

        # get sample of size n_svs from truncated gaussian distribution (truncated at 3)
        p = np.random.normal(size=n_svs)
        p[p>3]=3
        p[p<-3]=-3
        # now convert to distribution with desired scale
        pert_scales = pert_scale * p
        pert = np.sum([pert_scales[ii] * svecs[ii] for ii in range(n_svs)], axis=0)
        # this is now flattened. reshape
        pert = pert.reshape((Nlat,Nlon,n_channels_in))
        # "positive" and "negative" perturbation
        x_pert1 = x + pert
        x_pert2 = x - pert

        x_perts.append(x_pert1)
        x_perts.append(x_pert2)

    x_perts = np.array(x_perts)
    return x_perts


def create_pertubed_states_rand(x):
    perts = np.random.normal(0, scale=pert_scale, size=(n_ens,Nlat,Nlon,n_channels_in))
    # since this is a finite sample from the distribution,
    # it does not necessarily have 0 mean and pert_scale variance (section 4.5 in Tellus paper)
    # and we therefore rescale them for every channel and gridpoint
    perts = (perts - perts.mean(axis=0)) / perts.std(axis=0) * pert_scale
    # perts is (n_ens,N), and x is (N), so we can use standard numpy broadcasting
    # to create n_ens states
    x_pert = x + perts
    assert(np.allclose(np.mean(x_pert-x, axis=0),0.))
    assert(np.allclose(np.std(x_pert, axis=0),pert_scale))
    return x_pert

# prediction
target_max_leadtime = 5*24
max_forecast_steps = target_max_leadtime // (lead_time*time_resolution_hours)



def compute_mse(x,y):
    '''mse per sample/forecast'''
    assert(x.shape == y.shape)
    assert(x.shape[1:] == (Nlat,Nlon,n_channels_in))
    return np.mean((x-y)**2,axis=(1,2)) * norm_std**2


# load precomputed singular vectors
svecs_all =np.load(f'{outdir}/jacobians_{param_string}_{test_startyear}-{test_endyear}_{svd_params}_tres{tres_factor}.npy')
assert(len(data_reduced) == len(svecs_all))


# network member ensemble
member_selection_scores = np.load('mse_per_mem.npy')
# select only the 20 best members (at lead time index 10)
best_mem_idcs = member_selection_scores[10,:].argsort()[:20][::-1]
best_mem_idcs = best_mem_idcs +1
network_ensemble_all = []
for imem in best_mem_idcs:
    _modelfile=f'data_mem{imem}/trained_model_era5_2.5deg_weynetal_1979-2016_mem{imem}.h5'
    _model = keras.models.load_model(_modelfile,    custom_objects={'PeriodicPadding': PeriodicPadding})
    network_ensemble_all.append(_model)



x_init_ctrl = data_reduced[:-int(max_forecast_steps//tres_factor)]



# loop over different parameters for th eensemble.
# in each loop iteration, all forecasts are made and evaluated
for n_ens in [2,4,10,20,100]:

    if n_ens <=20:
        # network member ensemble

        # initialize (via repeating x_init_ctrl)
        y_pred_netens = np.zeros((len(x_init_ctrl), n_ens, Nlat, Nlon, n_channels_in))
        for i in range(n_ens):
            y_pred_netens[:,i] = x_init_ctrl

        res_mse_netens = []
        res_ensvar_netens = []
        res_mse_netens_permember = []
        res_crps_netens = []
        leadtimes = []

        for ilead, fc_step in enumerate(range(0, max_forecast_steps)):
            print(ilead)
            leadtime = fc_step * (lead_time * time_resolution_hours)
            leadtimes.append(leadtime)
            y_pred_ensmean_netens = np.mean(y_pred_netens, axis=1)
            y_pred_ensvar_netens_2d = np.var(y_pred_netens, axis=1)

            # get right target data for this leadtime
            if fc_step < max_forecast_steps:
                truth = data[fc_step * lead_time:-(max_forecast_steps - fc_step * lead_time):tres_factor]
            else:
                truth = data[fc_step * lead_time::tres_factor]

            # compute ensemble mean error
            mse_ensmean_netens = compute_mse(y_pred_ensmean_netens, truth)
            ensvar_mean_netens = np.mean(y_pred_ensvar_netens_2d, axis=(1, 2)) * norm_std ** 2
            res_mse_netens.append(mse_ensmean_netens)
            res_ensvar_netens.append(ensvar_mean_netens)
            mse_ensmean_member = np.array([compute_mse(y_pred_netens[:,i], truth) for i in range(n_ens)])
            res_mse_netens_permember.append(mse_ensmean_member)

            # compute crps (per forecast and per gridpoint). the crps_ensemble function
            # takes in arbitrary dimensions, and treats each point as a single forecast.
            # so what we get if we feed in our data is a crps score per time per gridpoint
            crps_netens = properscoring.crps_ensemble(truth, y_pred_netens, axis=1)
            # compute area mean crps
            crps_netens = np.mean(crps_netens, axis=(1,2))
            res_crps_netens.append(crps_netens)
            # make next forecast step for all members
            for i in range(n_ens):
                y_pred_netens[:,i] = network_ensemble_all[i].predict(y_pred_netens[:,i])

        res_mse_netens = np.array(res_mse_netens)
        res_ensvar_netens = np.array(res_ensvar_netens)
        res_mse_netens_permember = np.array(res_mse_netens_permember)
        res_crps_netens = np.array(res_crps_netens)

        out = {'leadtime': leadtimes,
               'mse_ensmean_netens': res_mse_netens,
               'spread_netens': res_ensvar_netens,
               'crps_netens': res_crps_netens,
               'n_ens': n_ens,
               'pert_scale': pert_scale,
               'mse_ensmean_netens_permember':res_mse_netens_permember,
               }
        pickle.dump(out, open(
            f'{outdir}/era5_fc_eval_results_netens_{param_string}_{test_startyear}-{test_endyear}_nresample{n_resample}_n_ens{n_ens}.pkl',
            'wb'))

    # svd ensemble
    for pert_scale in [0.001,0.003,0.01, 0.03,0.1,0.3,1, 3]:

        svd_params = f'n_svs{n_svs}_n_ens{n_ens}_pertscale{pert_scale}_nresample{n_resample}_svdleadtime{svd_leadtime}'
        # initialize, we have to discard the last max_leadtime initial fields, because we dont
        # have the targets for them. since we decreased the timeresolution, we have
        # to account for the decreased timeresolution as well
        x_init_ctrl = data_reduced[:-int(max_forecast_steps//tres_factor)]
        svecs = svecs_all[:-int(max_forecast_steps//tres_factor)]
        assert(len(x_init_ctrl) == len(svecs))
        # crate_perturbed_states_batch does not work with large batches (memory problems),
        # therefore for now we loop over all state and use the function for single states
        for n_svs_reduced in [10,20,40,60,80,100]:
            x_init_pert_svd = np.array([create_pertubed_states_svd(x_init_ctrl[i], svecs_all[i], n_svs_reduced) for i in tqdm(range(len(x_init_ctrl)))])
            n_samples = len(x_init_ctrl)
            x_init_pers = x_init_ctrl.copy()
            y_pred_ctrl = x_init_ctrl

            # x_init_pert has shape (ntime,nens,Nlat,Nlon,Nchannel_in)
            x_init_pert_svd_flat = x_init_pert_svd.reshape((n_samples*n_ens,Nlat,Nlon,n_channels_in))
            y_pred_ens_svd_flat = x_init_pert_svd_flat

            res_mse_ctrl = []
            res_mse_pers = []
            res_mse_ensmean_svd = []
            leadtimes = []
            res_mse_pers = []
            res_ensvar_svd = []
            res_crps_svd = []

            for ilead,fc_step in enumerate(range(0,max_forecast_steps)):
                print(ilead)
                leadtime = fc_step * (lead_time*time_resolution_hours)
                leadtimes.append(leadtime)

                # we start with evaluation at leadtime 0 (so without prediction yet)
                y_pred_ens = y_pred_ens_svd_flat.reshape(x_init_pert_svd.shape)
                y_pred_ensmean = np.mean(y_pred_ens, axis=1)  # member dimension is 2nd dime
                y_pred_ensvar_2d = np.var(y_pred_ens, axis=1)

                # get right target data for this leadtime
                if fc_step < max_forecast_steps:
                    truth = data[fc_step*lead_time:-(max_forecast_steps-fc_step*lead_time):tres_factor]
                else:
                    truth = data[fc_step*lead_time::tres_factor]

                assert(truth.shape == y_pred_ctrl.shape)
                mse_ctrl = compute_mse(y_pred_ctrl, truth)
                mse_ensmean_svd = compute_mse(y_pred_ensmean, truth)
                res_mse_ctrl.append(mse_ctrl)
                res_mse_ensmean_svd.append(mse_ensmean_svd)
                mse_pers = compute_mse(x_init_pers, truth)
                res_mse_pers.append(mse_pers)
                ensvar_mean = np.mean(y_pred_ensvar_2d,axis=(1,2)) * norm_std **2
                res_ensvar_svd.append(ensvar_mean)

                crps_svd = np.mean(properscoring.crps_ensemble(truth, y_pred_ens, axis=1),axis=(1,2))
                res_crps_svd.append(crps_svd)
                # predict next timestep
                y_pred_ctrl = model.predict(y_pred_ctrl, verbose=0)
                y_pred_ens_svd_flat = model.predict(y_pred_ens_svd_flat, verbose=0)

            res_mse_ensmean_svd = np.array(res_mse_ensmean_svd)
            res_ensvar_svd = np.array(res_ensvar_svd)
            res_crps_svd = np.array(res_crps_svd)
            
            out = {'leadtime': leadtimes,
                          'mse_ensmean_svd': res_mse_ensmean_svd,
                          'spread_svd': res_ensvar_svd,
                          'crps_svd': res_crps_svd,
                          'n_ens': n_ens,
                          'pert_scale': pert_scale,
                          'n_svs_reduced':n_svs_reduced,
                          }
            pickle.dump(out, open(
                f'{outdir}/era5_fc_eval_results_svd_{param_string}_{test_startyear}-{test_endyear}_{svd_params}_n_svs_reduced{n_svs_reduced}.pkl',
                'wb'))

    # rand ensemble
    for pert_scale in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]:
        x_init_ctrl = data_reduced[:-int(max_forecast_steps // tres_factor)]
        x_init_pert_rand = np.array(
            [create_pertubed_states_rand(x_init_ctrl[i]) for i in tqdm(range(len(x_init_ctrl)))])
        n_samples = len(x_init_ctrl)

        # x_init_pert has shape (ntime,nens,Nlat,Nlon,Nchannel_in)
        x_init_pert_rand_flat = x_init_pert_rand.reshape((n_samples * n_ens, Nlat, Nlon, n_channels_in))
        y_pred_ens_rand_flat = x_init_pert_rand_flat

        res_mse_ensmean_rand = []
        leadtimes = []
        res_ensvar_rand = []
        res_crps_rand = []

        for ilead, fc_step in enumerate(range(0, max_forecast_steps)):
            print(ilead)
            leadtime = fc_step * (lead_time * time_resolution_hours)
            leadtimes.append(leadtime)

            # we start with evaluation at leadtime 0 (so without prediction yet)
            y_pred_ens_rand = y_pred_ens_rand_flat.reshape(x_init_pert_rand.shape)
            y_pred_ensmean_rand = np.mean(y_pred_ens_rand, axis=1)  # member dimension is 2nd dime
            y_pred_ensvar_rand_2d = np.var(y_pred_ens_rand, axis=1)

            # get right target data for this leadtime
            if fc_step < max_forecast_steps:
                truth = data[fc_step * lead_time:-(max_forecast_steps - fc_step * lead_time):tres_factor]
            else:
                truth = data[fc_step * lead_time::tres_factor]

            mse_ensmean_rand = compute_mse(y_pred_ensmean_rand, truth)
            res_mse_ensmean_rand.append(mse_ensmean_rand)
            ensvar_mean_rand = np.mean(y_pred_ensvar_rand_2d, axis=(1, 2)) * norm_std ** 2
            res_ensvar_rand.append(ensvar_mean_rand)

            crps_rand = np.mean(properscoring.crps_ensemble(truth, y_pred_ens_rand, axis=1), axis=(1, 2))
            res_crps_rand.append(crps_rand)
            # predict next timestep
            y_pred_ens_rand_flat = model.predict(y_pred_ens_rand_flat, verbose=0)

        res_mse_ensmean_rand = np.array(res_mse_ensmean_rand)
        res_ensvar_rand = np.array(res_ensvar_rand)
        res_crps_rand = np.array(res_crps_rand)

        out = {'leadtime': leadtimes,
                   'mse_ensmean_rand': res_mse_ensmean_rand,
                   'spread_rand': res_ensvar_rand,
                   'crps_rand': res_crps_rand,
                   'n_ens': n_ens,
                   'pert_scale': pert_scale,
                   }
        pickle.dump(out, open(
            f'{outdir}/era5_fc_eval_results_rand_{param_string}_{test_startyear}-{test_endyear}_n_ens{n_ens}_pertscale{pert_scale}_nresample{n_resample}.pkl',
            'wb'))

