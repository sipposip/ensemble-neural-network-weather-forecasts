

import os
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from pylab import plt
from tqdm import tqdm
import xarray as xr

plotdir='plots/'
os.system(f'mkdir -p {plotdir}')

def savefig(figname):
    plt.savefig(figname+'.svg')
    plt.savefig(figname+'.pdf')

norm_weights_folder = 'normalization_weights/'
# in the make and eval script, we forgot to scale CRPS to real units (it was computed on the
# normalized data). therefore we need the norm_scale here
norm_std = xr.open_dataarray(norm_weights_folder+'/normalization_weights_era5_2.5deg_2015-2016_tres6_geopotential_500_std.nc').values

# # create a list of dictionaries, each ditc containing the reults for one n_ens - pert_scale combination
data_svd = []
for n_ens in [2,4,10,20, 100]:
    for pert_scale in [0.001,0.003,0.01, 0.03,0.1,0.3,1, 3]:
        for n_svs in [10, 20, 40, 60, 80, 100]:
            for svd_leadtime in [1,2,4,8]:
                ifile = f'output/era5_fc_eval_results_svd_era5_2.5deg_weynetal_1979-2016_2017-2018_n_svs100_n_ens{n_ens}_pertscale{pert_scale}_nresample1_svdleadtime{svd_leadtime}_n_svs_reduced{n_svs}.pkl'
                res = pickle.load(open(ifile,'rb'))
                # the data is on float32. convert all float data to float 64
                res['svd_leadtime'] = svd_leadtime
                for key in res.keys():
                    if type(res[key]) == np.ndarray and res[key].dtype=='float32':
                        res[key] = res[key].astype('float64')
                data_svd.append(res)

data_netens = []
for n_ens in [2,4,10,20]:
    ifile = f'output/era5_fc_eval_results_netens_era5_2.5deg_weynetal_1979-2016_2017-2018_nresample1_n_ens{n_ens}.pkl'
    res = pickle.load(open(ifile,'rb'))
    # the data is on float32. convert all float data to float 64
    for key in res.keys():
        if type(res[key]) == np.ndarray and res[key].dtype=='float32':
            res[key] = res[key].astype('float64')
    data_netens.append(res)


# read rand ensemble:
data_rand = []
for n_ens in [2,4,10,20,100]:
    for pert_scale in [0.001,0.003,0.01, 0.03,0.1,0.3,1, 3]:
        ifile = f'output/era5_fc_eval_results_rand_era5_2.5deg_weynetal_1979-2016_2017-2018_n_ens{n_ens}_pertscale{pert_scale}_nresample1.pkl'
        res = pickle.load(open(ifile,'rb'))
        # the data is on float32. convert all float data to float 64
        for key in res.keys():
            if type(res[key]) == np.ndarray and res[key].dtype=='float32':
                res[key] = res[key].astype('float64')
        data_rand.append(res)


# read drop ensemble
data_drop = []
for n_ens in [2,4,10,20,100]:
    for drop_rate in [0.001,0.003,0.01,0.03,0.1,0.2,0.4]:
        ifile = f'output/era5_fc_eval_results_drop_era5_2.5deg_weynetal_1979-2016_2017-2018_nresample1_n_ens{n_ens}_drop_rate{drop_rate}.pkl'
        res = pickle.load(open(ifile,'rb'))
        # the data is on float32. convert all float data to float 64
        for key in res.keys():
            if type(res[key]) == np.ndarray and res[key].dtype=='float32':
                res[key] = res[key].astype('float64')
        data_drop.append(res)




# loop over all configurations and compute rmse, and correlation
# since the different ensemble types have different possible configurations, we do this
# individually for each ensemble type
df_drop = []
for sub in tqdm(data_drop):
    # for each leadtime, compute mean error, spread, and spread error correlaation, the latter
    # including uncertainty estimates
    for ltime in range(len(sub['leadtime'])):

        if ltime > 0:  # for leadtime 0, correlation does not make sense
            # we want the correlation nbetween rmse and spread, so we have to take the sqrt since we have
            # mse and variance
            corr = np.corrcoef(np.sqrt(sub['mse_ensmean_drop'][ltime].squeeze()),
                                            np.sqrt(sub['spread_drop'][ltime].squeeze()))[0,1]

        else:
            corr = [0]

        _df = pd.DataFrame({'leadtime':sub['leadtime'][ltime],
                            'corr_drop':corr,
                            #compute RMSE and mean stddev
                            'rmse_ensmean_drop': np.sqrt(np.mean(sub['mse_ensmean_drop'][ltime])),
                            'spread_drop': np.sqrt(np.mean(sub['spread_drop'][ltime])),
                            # scale crps to real units
                            'crps_drop': np.mean(sub['crps_drop'][ltime])*norm_std,
                            'n_ens': sub['n_ens'],
                            'drop_rate':sub['drop_rate'],
                            }, index=[0]
                           )
        df_drop.append(_df)
df_drop = pd.concat(df_drop, sort=False)


df_svd = []
for sub in tqdm(data_svd):
    # for each leadtime, compute mean error, spread, and spread error correlaation, the latter
    # including uncertainty estimates
    for ltime in range(len(sub['leadtime'])):

        if ltime > 0:  # for leadtime 0, correlation does not make sense
            # we want the correlation nbetween rmse and spread, so we have to take the sqrt since we have
            # mse and variance
            corr = np.corrcoef(np.sqrt(sub['mse_ensmean_svd'][ltime].squeeze()),
                                            np.sqrt(sub['spread_svd'][ltime].squeeze()))[0,1]

        else:
            corr = [0]

        _df = pd.DataFrame({'leadtime':sub['leadtime'][ltime],
                            'corr_svd':corr,
                            #compute RMSE and mean stddev
                            'rmse_ensmean_svd': np.sqrt(np.mean(sub['mse_ensmean_svd'][ltime])),
                            'spread_svd': np.sqrt(np.mean(sub['spread_svd'][ltime])),
                            'crps_svd': np.mean(sub['crps_svd'][ltime])*norm_std,
                            'n_ens': sub['n_ens'],
                            'n_svs':sub['n_svs_reduced'],
                            'pert_scale':sub['pert_scale'],
                            'svd_leadtime':sub['svd_leadtime'],
                            }, index=[0]
                           )
        df_svd.append(_df)
df_svd = pd.concat(df_svd, sort=False)

df_netens = []
for sub in tqdm(data_netens):
    # for each leadtime, compute mean error, spread, and spread error correlaation, the latter
    # including uncertainty estimates
    for ltime in range(len(sub['leadtime'])):

        if ltime > 0:  # for leadtime 0, correlation does not make sense
            # we want the correlation nbetween rmse and spread, so we have to take the sqrt since we have
            # mse and variance
            corr = np.corrcoef(np.sqrt(sub['mse_ensmean_netens'][ltime].squeeze()),
                                            np.sqrt(sub['spread_netens'][ltime].squeeze()))[0,1]

        else:
            corr = [0]

        _df = pd.DataFrame({'leadtime':sub['leadtime'][ltime],
                            'corr_netens':corr,
                            #compute RMSE and mean stddev
                            'rmse_ensmean_netens': np.sqrt(np.mean(sub['mse_ensmean_netens'][ltime])),
                            # as ctrl we take the mean error of all members (not error ensemble mean, but
                            # mean of error of individual members). the member dimension is just another dimension,
                            # so we can take the mean of everything (time and member)
                            'rmse_ctrl':np.sqrt(np.mean(sub['mse_ensmean_netens_permember'][ltime])),
                            'spread_netens': np.sqrt(np.mean(sub['spread_netens'][ltime])),
                            'crps_netens':np.mean(sub['crps_netens'][ltime])*norm_std,
                            'n_ens': sub['n_ens'],
                            }, index=[0]
                           )
        df_netens.append(_df)
df_netens = pd.concat(df_netens, sort=False)

df_rand = []
for sub in tqdm(data_rand):
    # for each leadtime, compute mean error, spread, and spread error correlaation, the latter
    # including uncertainty estimates
    for ltime in range(len(sub['leadtime'])):

        if ltime > 0:  # for leadtime 0, correlation does not make sense
            # we want the correlation nbetween rmse and spread, so we have to take the sqrt since we have
            # mse and variance
            corr = np.corrcoef(np.sqrt(sub['mse_ensmean_rand'][ltime].squeeze()),
                                            np.sqrt(sub['spread_rand'][ltime].squeeze()))[0,1]

        else:
            corr = [0]

        _df = pd.DataFrame({'leadtime':sub['leadtime'][ltime],
                            'corr_rand':corr,
                            #compute RMSE and mean stddev
                            'rmse_ensmean_rand': np.sqrt(np.mean(sub['mse_ensmean_rand'][ltime])),
                            'spread_rand': np.sqrt(np.mean(sub['spread_rand'][ltime])),
                            'crps_rand': np.mean(sub['crps_rand'][ltime])*norm_std,
                            'n_ens': sub['n_ens'],
                            'pert_scale':sub['pert_scale'],
                            }, index=[0]
                           )
        df_rand.append(_df)
df_rand = pd.concat(df_rand, sort=False)


#
# ## plotting
sns.set_context("paper", font_scale=1.5, rc={'lines.linewidth': 2.5,
                                             'legend.fontsize':'small'})
plt.rcParams['savefig.bbox']='tight'
plt.rcParams['legend.frameon']=True
plt.rcParams['legend.framealpha']=0.4
plt.rcParams['legend.labelspacing']=0.1
figsize=(7.5,3)



## selection strategies:
## best RMSE, best CRPS, besr spread-error correltation.
# everything for different leadtimes


colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']

selection_leadtime=60
n_ens = 100
table = []
plt.figure(figsize=(20,11))
for iplot,optimization_var in enumerate(['rmse_ensmean','corr', 'crps']):

    if optimization_var == 'corr':
        opt_func = np.argmax
    else:
        opt_func = np.argmin

    # rand
    sub = df_rand.query('leadtime==@selection_leadtime & n_ens==n_ens')
    best_rand_sub = sub.iloc[opt_func(sub[optimization_var+'_rand'].values)]
    best_rand_pert_scale = best_rand_sub['pert_scale']
    best_rand = df_rand.query('pert_scale==@best_rand_pert_scale')
    # svd
    sub = df_svd.query('leadtime==@selection_leadtime & n_ens==n_ens')
    best_svd_sub = sub.iloc[opt_func(sub[optimization_var+'_svd'].values)]
    best_svd_pert_scale = best_svd_sub['pert_scale']
    best_svd_n_svs = best_svd_sub['n_svs']
    best_svd_svd_leadtime = best_svd_sub['svd_leadtime']
    best_svd = df_svd.query('pert_scale==@best_svd_pert_scale & n_svs==@best_svd_n_svs & svd_leadtime==@best_svd_svd_leadtime')

    # drop
    sub = df_drop.query('leadtime==@selection_leadtime & n_ens==n_ens')
    best_drop_sub = sub.iloc[opt_func(sub[optimization_var+'_drop'].values)]
    best_drop_rate = best_drop_sub['drop_rate']
    best_drop = df_drop.query('drop_rate==@best_drop_rate')

    sub_svd = best_svd.query('n_ens==@n_ens')
    sub_rand = best_rand.query('n_ens==@n_ens')
    sub_drop = best_drop.query('n_ens==@n_ens')

    sub_netens = df_netens.query('n_ens==20')
    #plt.figure(figsize=figsize)
    plt.subplot(3,3,1+iplot)
    plt.plot(sub_svd['leadtime'], sub_svd['rmse_ensmean_svd'], label='svd', color=colors[0])
    plt.plot(sub_rand['leadtime'], sub_rand['rmse_ensmean_rand'], label='rand', color=colors[1], zorder=1)
    plt.plot(sub_drop['leadtime'], sub_drop['rmse_ensmean_drop'], label='drop', color=colors[2])
    plt.plot(sub_netens['leadtime'], sub_netens['rmse_ensmean_netens'], label='multitrain', color=colors[3])
    plt.plot(sub_netens['leadtime'], sub_netens['rmse_ctrl'], label='unperturbed', color='grey')
    plt.plot(sub_svd['leadtime'], sub_svd['spread_svd'],  color=colors[0], linestyle='--', zorder=1)
    plt.plot(sub_rand['leadtime'], sub_rand['spread_rand'], color=colors[1], linestyle='--', zorder=0)
    plt.plot(sub_drop['leadtime'], sub_drop['spread_drop'], color=colors[2], linestyle='--')
    plt.plot(sub_netens['leadtime'], sub_netens['spread_netens'], color=colors[3], linestyle='--')
    if iplot==0:
        plt.legend(loc='upper left', fontsize=14)
        plt.ylabel('rmse (solid) \n spread (dashed) [$m^2/s^2$]')

    sns.despine()
    plt.ylim(ymax=2000)
    plt.xlim((0, 120))
    plt.title(f'selected on {optimization_var}')

    plt.subplot(3, 3, 4+iplot)
    plt.plot(sub_svd['leadtime'], sub_svd['crps_svd'], label='svd', color=colors[0])
    plt.plot(sub_rand['leadtime'], sub_rand['crps_rand'], label='rand', color=colors[1], zorder=1)
    plt.plot(sub_drop['leadtime'], sub_drop['crps_drop'], label='drop', color=colors[2], zorder=1)
    plt.plot(sub_netens['leadtime'], sub_netens['crps_netens'], label='retrain', color=colors[3])
    if iplot==0:
        plt.ylabel('crps [$m^2/s^2$]')
    sns.despine()
    plt.ylim(ymax=1200)
    plt.xlim((0, 120))

    plt.subplot(3, 3, 7+iplot)
    plt.plot(sub_svd['leadtime'][1:], sub_svd['corr_svd'][1:], label='svd', color=colors[0])
    #plt.fill_between(sub_df['leadtime'][1:], sub_df['corr_svd_lower'][1:],sub_df['corr_svd_upper'][1:],color='#1b9e77', alpha=0.5)
    plt.plot(sub_rand['leadtime'][1:], sub_rand['corr_rand'][1:], label='rand', color=colors[1])
    #plt.fill_between(sub_df['leadtime'][1:], sub_df['corr_rand_lower'][1:],sub_df['corr_rand_upper'][1:],color='#7570b3', alpha=0.5)
    plt.plot(sub_drop['leadtime'][1:], sub_drop['corr_drop'][1:], label='drop', color=colors[2])
    plt.plot(sub_netens['leadtime'][1:], sub_netens['corr_netens'][1:], label='retrain', color=colors[3])
    #plt.fill_between(sub_df['leadtime'][1:], sub_df['corr_netens_lower'][1:],sub_df['corr_netens_upper'][1:],color='#d95f02', alpha=0.5)

    plt.xlabel('leadtime [h]')
    if iplot == 0:
        plt.ylabel('spread-error correlation')
    sns.despine()
    plt.ylim((0,0.9))
    plt.xlim((0,120))

    table.append(pd.DataFrame({
        'selected on': optimization_var,
        '$\sigma_{rand}$': best_rand_pert_scale,
        '$\sigma_{svd}$': best_svd_pert_scale,
        '$n_{svs}$': best_svd_n_svs,
        # svd_leadtime is in steps, convert it to hours
        '$T_{svd}$': best_svd_svd_leadtime * 6,
        '$p_{drop}$': best_drop_rate,


    },index=[0]))

savefig(f'{plotdir}/era5_best_combis_all')


table = pd.concat(table)

# table with optimal parameters`

# save as latex table
with open('result_table.tex', 'w') as f:
    f.write(table.to_latex(escape=False, index=False))





# -------------------- plots rand ensemble

plt.figure(figsize=(20,11))

selvar='crps'
pert_scale = float(table[table['selected on']==selvar]['$\sigma_{rand}$'].values)
sub_df = df_rand.query('pert_scale==@pert_scale')

sub_df = sub_df[sub_df['leadtime']!=0]
plt.subplot(321)
sns.lineplot('n_ens','rmse_ensmean_rand', data=sub_df, hue='leadtime')
sns.lineplot('n_ens','spread_rand', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)],
                  legend=False)
plt.legend()
sns.despine()
plt.xlabel('')
plt.title('$\sigma_{rand}$='+str(pert_scale))
plt.ylabel('spread (dashed) \n error (solid)')
plt.subplot(323)
sns.lineplot('n_ens','corr_rand', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('spread error correlation')
plt.xlabel('')
plt.subplot(325)
sns.lineplot('n_ens','crps_rand', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('crps [$m^2/s^2$]')
plt.xlabel('$n_{ens}$')
#savefig(f'{plotdir}/era5_rand_n_ens_vs_all_pert_scale{pert_scale}')


## pert_scale vs rmse / corr/ crps with fixed n_ens
n_ens=100
sub_df = df_rand.query('n_ens==@n_ens')
# here we omit leadtime 0 (because we dont need and it makes the plots confusing, especially for corr)
sub_df = sub_df[sub_df['leadtime']!=0]
#plt.figure(figsize=(10,11))
plt.subplot(322)
sns.lineplot('pert_scale','rmse_ensmean_rand', data=sub_df, hue='leadtime')
sns.lineplot('pert_scale','spread_rand', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)],
                  legend=False)
plt.legend()
sns.despine()
plt.ylim(ymax=3000)
plt.title(f'n_ens={n_ens}')
plt.ylabel('spread (dashed) \n error (solid)')
plt.xlabel('')
plt.subplot(324)
sns.lineplot('pert_scale','corr_rand', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('spread error correlation')
plt.xlabel('')
plt.subplot(326)
sns.lineplot('pert_scale','crps_rand', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('crps [$m^2/s^2$]')
plt.xlabel('$\sigma_{rand}$')
savefig(f'{plotdir}/era5_rand_sensitivity')



# -------------------- plots drop ensemble

plt.figure(figsize=(20,11))

## n_ens vs rmse / corr/ crps with fixed drop_rate
selvar='crps'
drop_rate = float(table[table['selected on']==selvar]['$p_{drop}$'].values)
sub_df = df_drop.query('drop_rate==@drop_rate')

sub_df = sub_df[sub_df['leadtime']!=0]
plt.subplot(321)
sns.lineplot('n_ens','rmse_ensmean_drop', data=sub_df, hue='leadtime')
sns.lineplot('n_ens','spread_drop', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)],
                  legend=False)
plt.legend()
sns.despine()
plt.xlabel('')
plt.title('$p_{drop}$='+str(drop_rate))
plt.ylabel('spread (dashed) \n error (solid)')
plt.subplot(323)
sns.lineplot('n_ens','corr_drop', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('spread error correlation')
plt.xlabel('')
plt.subplot(325)
sns.lineplot('n_ens','crps_drop', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('crps [$m^2/s^2$]')
plt.xlabel('$n_{ens}$')


## pert_scale vs rmse / corr/ crps with fixed n_ens
n_ens=100
sub_df = df_drop.query('n_ens==@n_ens')
# here we omit leadtime 0 (because we dont need and it makes the plots confusing, especially for corr)
sub_df = sub_df[sub_df['leadtime']!=0]
plt.subplot(322)
sns.lineplot('drop_rate','rmse_ensmean_drop', data=sub_df, hue='leadtime')
sns.lineplot('drop_rate','spread_drop', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)],
                  legend=False)
plt.legend()
sns.despine()
plt.title(f'n_ens={n_ens}')
plt.ylabel('spread (dashed) \n error (solid)')
plt.xlabel('')
plt.subplot(324)
sns.lineplot('drop_rate','corr_drop', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('spread error correlation')
plt.xlabel('')
plt.subplot(326)
sns.lineplot('drop_rate','crps_drop', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('crps [$m^2/s^2$]')
plt.xlabel('$p_{drop}$')
savefig(f'{plotdir}/era5_drop_sensitiviy')


# -------------------- netens plots

## n_ens vs rmse / corr/ crps
plt.figure(figsize=(10,11))
sub_df = df_netens
# here we omit leadtime 0 (because we dont need and it makes the plots confusing, especially for corr)
sub_df = sub_df[sub_df['leadtime']!=0]
plt.subplot(3,1,1)
sns.lineplot('n_ens','rmse_ensmean_netens', data=sub_df, hue='leadtime')
sns.lineplot('n_ens','spread_netens', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)], legend=False)
plt.legend()
plt.xlabel('')
plt.ylabel('rmse (solid) \n spread (dashed) [$m^2/s^2$]')
plt.xticks(np.arange(2,21,2))
sns.despine()
plt.subplot(3,1,2)
sns.lineplot('n_ens','corr_netens', data=sub_df, hue='leadtime')
plt.legend()
plt.xlabel('')
plt.ylabel('spread error correlation')
plt.xticks(np.arange(2,21,2))
sns.despine()
plt.subplot(3,1,3)
sns.lineplot('n_ens','crps_netens', data=sub_df, hue='leadtime')
plt.legend()
plt.xlabel('$n_{ens}$')
plt.ylabel('crps [$m^2/s^2$]')
plt.xticks(np.arange(2,21,2))
sns.despine()
savefig(f'{plotdir}/era5_netens_n_ens_vs_all')







# -------------------- plots svd ensemble

plt.figure(figsize=(20,11))

selvar='crps'
pert_scale = float(table[table['selected on']==selvar]['$\sigma_{svd}$'].values)
n_svs = int(table[table['selected on']==selvar]['$n_{svs}$'].values)
svd_leadtime = int(table[table['selected on']==selvar]['$T_{svd}$'].values/6)


## n_ens vs rmse / corr/ crps with fixed pert_scale, svd_leadtime and n_svs
sub_df = df_svd.query('pert_scale==@pert_scale & svd_leadtime==@svd_leadtime & n_svs==@n_svs')

sub_df = sub_df[sub_df['leadtime']!=0]
plt.subplot(321)
sns.lineplot('n_ens','rmse_ensmean_svd', data=sub_df, hue='leadtime')
sns.lineplot('n_ens','spread_svd', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)],
                  legend=False)
plt.legend()
sns.despine()
plt.xlabel('')
plt.ylabel('spread (dashed) \n error (solid)')
plt.subplot(323)
sns.lineplot('n_ens','corr_svd', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('spread error correlation')
plt.xlabel('')
plt.subplot(325)
sns.lineplot('n_ens','crps_svd', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('crps [$m^2/s^2$]')
plt.xlabel('$n_{ens}$')

## pert_scale vs rmse / corr/ crps with fixed n_ens, svd_leadtime and n_svs
n_ens=100
sub_df = df_svd.query('n_ens==@n_ens & svd_leadtime==@svd_leadtime & n_svs==@n_svs')
# here we omit leadtime 0 (because we dont need and it makes the plots confusing, especially for corr)
sub_df = sub_df[sub_df['leadtime']!=0]
#plt.figure(figsize=(10,11))
plt.subplot(322)
sns.lineplot('pert_scale','rmse_ensmean_svd', data=sub_df, hue='leadtime')
sns.lineplot('pert_scale','spread_svd', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)],
                  legend=False)
plt.legend()
sns.despine()
plt.ylim(ymax=3000)
plt.ylabel('spread (dashed) \n error (solid)')
plt.xlabel('')
plt.subplot(324)
sns.lineplot('pert_scale','corr_svd', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('spread error correlation')
plt.xlabel('')
plt.subplot(326)
sns.lineplot('pert_scale','crps_svd', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('crps [$m^2/s^2$]')
plt.xlabel('$\sigma_{svd}$')
savefig(f'{plotdir}/era5_svd_sensitivity_1')



plt.figure(figsize=(20,11))
## n_svs vs rmse / corr/ crps with fixed n_ens, svd_leadtime and pert_scale

sub_df = df_svd.query('pert_scale==@pert_scale & svd_leadtime==@svd_leadtime & n_ens==@n_ens')

sub_df = sub_df[sub_df['leadtime']!=0]
plt.subplot(321)
sns.lineplot('n_svs','rmse_ensmean_svd', data=sub_df, hue='leadtime')
sns.lineplot('n_svs','spread_svd', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)],
                  legend=False)
plt.legend()
sns.despine()
plt.xlabel('')
plt.ylabel('spread (dashed) \n error (solid)')
plt.subplot(323)
sns.lineplot('n_svs','corr_svd', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('spread error correlation')
plt.xlabel('')
plt.subplot(325)
sns.lineplot('n_svs','crps_svd', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('crps [$m^2/s^2$]')
plt.xlabel('$n_{svs}$')

## svd_leadtime vs rmse / corr/ crps with fixed rest
n_ens=100
sub_df = df_svd.query('n_ens==@n_ens & pert_scale==@pert_scale & n_svs==@n_svs')
# here we omit leadtime 0 (because we dont need and it makes the plots confusing, especially for corr)
sub_df = sub_df[sub_df['leadtime']!=0]
# convert svd_leadtime from steps to hours
sub_df['svd_leadtime'] = sub_df['svd_leadtime'] * 6
plt.subplot(322)
sns.lineplot('svd_leadtime','rmse_ensmean_svd', data=sub_df, hue='leadtime')
sns.lineplot('svd_leadtime','spread_svd', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)],
                  legend=False)
plt.legend()
sns.despine()
plt.ylim(ymax=3000)
plt.ylabel('spread (dashed) \n error (solid)')
plt.xlabel('')
plt.subplot(324)
sns.lineplot('svd_leadtime','corr_svd', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('spread error correlation')
plt.xlabel('')
plt.subplot(326)
sns.lineplot('svd_leadtime','crps_svd', data=sub_df, hue='leadtime')
plt.legend()
sns.despine()
plt.ylabel('crps [$m^2/s^2$]')
plt.xlabel('$T_{svd} [h]$')
savefig(f'{plotdir}/era5_svd_sensitivity_2')