
"""

needs tensorflow 2.0

status 29.10.2019: only works with tf-nightly (because of problems with iterables/generators and threadsafeness
https://github.com/tensorflow/tensorflow/issues/31546




"""


import os
import pickle
import json
import sys
import threading

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras.layers import Convolution2D,Dropout
from tensorflow.keras import layers
import tensorflow.keras as keras
import dask

from multiprocessing.pool import ThreadPool
from dask.diagnostics import ProgressBar

ProgressBar().register()
# set a fixed threadpool for dask. I am not sure whether this is a good idea.
# the keras fit_generator spawns multile threads that load data via the generator,
# but the generator uses dask. on tetralith, net setting a fixed threadpool for dask
# lead to a very huge number of threads after some time....

# dask.config.set(scheduler="synchronous")

member = int(sys.argv[1])
datadir=f'data_mem{member}'
os.system(f'mkdir -p {datadir}')

# basepath for input data
if 'NSC_RESOURCE_NAME' in os.environ and os.environ['NSC_RESOURCE_NAME']=='tetralith':
    basepath = '/proj/bolinc/users/x_sebsc/weather-benchmark/2.5deg/'
    norm_weights_folder = '/proj/bolinc/users/x_sebsc/nn_/benchmark/normalization_weights/'
else:
    basepath = '/home/s/sebsc/pfs/weather_benchmark/2.5deg/'
    norm_weights_folder = '/home/s/sebsc/pfs/nn_ensemble/era5/normalization_weights/'


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# K.set_session(tf.Session(config=config))


lead_time = 1   # in steps
N_gpu=0
load_data_lazily = True
modelname='era5_2.5deg_weynetal'
train_startyear=1979
# train_startyear=2015
train_endyear=2016
time_resolution_hours = 6
variables = ['geopotential_500']
invariants = ['lsm', 'z']
valid_split = 2 / (train_endyear-train_startyear+1)

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
ds_whole = read_e5_data(train_startyear, train_endyear, variables=variables)
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



param_string = f'{modelname}_{train_startyear}-{train_endyear}_mem{member}'


time_indices_all = np.arange(0,n_data-time_resolution_hours*lead_time,time_resolution_hours)
data_train = np.array(ds_whole[0][time_indices_all], dtype='float32')
x_train = data_train[:-lead_time]
y_train = data_train[lead_time:]

x_train = (x_train - norm_mean) / norm_std
y_train = (y_train - norm_mean) / norm_std

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

n_rec_time=2

def build_model_weynetal2019():
    model = keras.Sequential([
        # the padding needs to be kernel_size//2 *dilation
        PeriodicPadding(axis=2, padding=2, input_shape=(Nlat,Nlon,n_channels_in)),
        layers.ZeroPadding2D(padding=(2, 0)),
        Convolution2D(32, kernel_size=3, dilation_rate=2, activation='tanh'),
        keras.layers.MaxPooling2D(pool_size=2),

        PeriodicPadding(axis=2, padding=1),
        layers.ZeroPadding2D(padding=(1, 0)),
        Convolution2D(64, kernel_size=3, dilation_rate=1, activation='tanh'),
        keras.layers.MaxPooling2D(pool_size=2),

        PeriodicPadding(axis=2, padding=1),
        layers.ZeroPadding2D(padding=(1, 0)),
        Convolution2D(128, kernel_size=3, dilation_rate=1, activation='tanh'),
        tf.keras.layers.UpSampling2D(size=2),

        PeriodicPadding(axis=2, padding=1),
        layers.ZeroPadding2D(padding=(1, 0)),
        Convolution2D(64, kernel_size=3, dilation_rate=1, activation='tanh'),
        tf.keras.layers.UpSampling2D(size=2),

        PeriodicPadding(axis=2, padding=2),
        layers.ZeroPadding2D(padding=(2, 0)),
        Convolution2D(32, kernel_size=3, dilation_rate=2, activation='tanh'),
        PeriodicPadding(axis=2, padding=2),
        layers.ZeroPadding2D(padding=(2, 0)),
        Convolution2D(n_channels_out, kernel_size=5, dilation_rate=1),

    ])
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


model = build_model_weynetal2019()


print(model.summary())

weight_file = f'{datadir}/weights_' + param_string + 'leadtime' + str(lead_time) + '{epoch:03d}-{val_loss:.5f}.h5'

callbacks = []
callbacks.append(keras.callbacks.ModelCheckpoint(weight_file, monitor='val_loss',
                                                 verbose=1,
                                                 save_weights_only=False, mode='auto', period=1))

print('start training')

hist = model.fit(x_train, y_train,
                     verbose=1,
                     epochs = 200,
                     validation_split=valid_split,
                 batch_size=batch_size,
                 callbacks=callbacks )



callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                        min_delta=0,
                        patience=50, # just to make sure we use a lot of patience before stopping
                        verbose=1, mode='auto',
                                           restore_best_weights=True))


# train more until early stopping
hist2 = model.fit(x_train, y_train,
                     verbose=1,
                  initial_epoch=201,
                     epochs = 400,
                     validation_split=valid_split,
                  callbacks=callbacks,
    batch_size=batch_size,
         )

print('finished training')
# save modelarchihtecture
json.dump(model.to_json(), open(f'{datadir}/modellayout_'+param_string+'.json','w'))
model.save(f'{datadir}/trained_model_'+param_string+'.h5')
model.save_weights(f'{datadir}/trained_model_weights_'+param_string+'.h5')


# reformat history

hist1 =  hist.history
hist2 =  hist2.history

pickle.dump((hist1,hist2),open(f'{datadir}/train_history_params_'+param_string+'.pkl','wb'))


