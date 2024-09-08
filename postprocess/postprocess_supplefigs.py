# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:08:59 2023

@author: leyan
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import wsimod.core.constants as constants
import pickle

figs_dir = r'D:\Huawei\PDRA-DAFNI\paper'
# %% arc results
results_folder = os.path.join(os.path.abspath(""), os.pardir)

flows = pd.read_csv(os.path.join(results_folder, 'flows.csv'))

flows['time'] = pd.to_datetime(flows['time'].astype(str))
flows = flows[(flows['time'] > '2014-12-31') & (flows['time'] <= '2024-01-01')]

flows = flows.groupby('arc')
# examine all arc results
if False:
    
    for arc in flows.groups.keys():
        flow = flows.get_group(arc).set_index('time')
        flow.index = pd.to_datetime(flow.index)
        fig, ax = plt.subplots()
        ax.plot(flow.index, flow['flow'])
        ax.set_title(arc)
# %%
# validation
metrics_ = {}
# river flow
if True:
    
    rainfall = pd.read_csv(os.path.join(results_folder, '1823-land-inputs.csv.gz')).groupby('variable').get_group('precipitation').set_index('time')
    rainfall.index = pd.to_datetime(rainfall.index)
    rainfall = rainfall['value'] / constants.MM_TO_M
    
    val_flows = {k: pd.read_csv(k).set_index('time') for k in ['38013_gdf.csv', '38033_gdf.csv']}
    for v in val_flows.values():
        v.index = pd.to_datetime(v.index)
    
    val_arc = '1823-river-to-1823-outlet'#'1823-storm-to-1823-storm-cso'#'1823-outlet-to-1823-downstream'#'luton_stw-wwtw-to-1823-outlet'#
    t_start = pd.to_datetime(flows.get_group(val_arc)['time'].unique()[0])
    t_end = pd.to_datetime(flows.get_group(val_arc)['time'].unique()[-1])
    
    import math as math
    from sklearn import metrics
    import numpy as np
    
    fig_width = 36
    fig_height = 7.5
    figs, axs = plt.subplots(3, 1, figsize=(fig_width,fig_height*3))
    metric_df = pd.DataFrame(index = list(val_flows.keys()), columns = ['nse', 'rmse', 'r', 'ksg', 'mape', 'pb'])
    
    nrfa_id = '38033_gdf.csv'
    obs_flow = val_flows[nrfa_id]
    
    ax1 = axs[0]
    # ax1.set_title(nrfa_id, fontsize = 30)
    
    sim_flow = flows.get_group(val_arc).set_index('time').flow / constants.D_TO_S # [m3/s]
    sim_flow.index = pd.to_datetime(sim_flow.index)
    ax1.plot(sim_flow.index, sim_flow, color = 'r', alpha = 1, label = 'sim', linewidth = 4, zorder = 2)
    # calculate metrics
    sim = pd.DataFrame(sim_flow, columns = ['flow'])
    sim = pd.DataFrame(sim.loc[pd.to_datetime('2000-01-01'):, 'flow'])
    obs = pd.DataFrame(pd.to_numeric(obs_flow['value']))
    obs.columns = ['flow']
    obs = obs.dropna()
    ind = sim.index.intersection(obs.index)
    if len(ind) > 0:
        obs = obs.loc[ind]
        sim = sim.loc[ind]
        
        cc = np.corrcoef(obs.values.ravel(), sim.values.ravel())
        r = cc[0,1]
        
        metric_df.loc[nrfa_id, 'nse'] = (1 - ((obs.sub(sim))**2).mean()/((obs.sub(obs.mean()))**2).mean())[0]
        metric_df.loc[nrfa_id, 'rmse'] = np.sqrt(metrics.mean_squared_error(obs, sim))
        metric_df.loc[nrfa_id, 'r'] = r
        metric_df.loc[nrfa_id, 'mape'] = (np.mean(np.abs((sim - obs) / obs), axis=0) * 100)[0]
        metric_df.loc[nrfa_id, 'ksg'] = 1 - math.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2 + (sim.mean() / obs.mean() - 1) ** 2)
        metric_df.loc[nrfa_id, 'pb'] = (sim.sub(obs).sum() / obs.sum())[0]

    ax1.scatter(obs_flow.index, obs_flow, s = 25, c = 'blue', label = 'obs', zorder=1)
    ax1.legend()
    ax1.set_ylabel('Flow ($m^3/s$)', size = 30)
    ax1.yaxis.set_label_coords(-0.04, 0.5)
    ymax = 5#max(max(obs.loc[t_start:t_end, 'flow']) * 1.5, 1)
    ax1.set_ylim([0,ymax])
    
    y1_ticks = np.linspace(0, ymax, 6, dtype = int)
    y1_ticklabels = [str(i) for i in y1_ticks]
    ax1.set_yticks(y1_ticks)
    ax1.tick_params(labelsize=30)
    ax1.tick_params(axis = 'x', pad = 10)
    
    # Create second axes, in order to get the bars from the top you can multiply by -1
    ax11 = ax1.twinx()
    ax11.plot(rainfall.index, -rainfall, c = 'black')
    # Now need to fix the axis labels
    max_pre = math.ceil(max(rainfall) * 4)
    y2_ticks = np.linspace(0, 200, 5, dtype=int)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax11.set_yticks(-1 * y2_ticks)
    y2_ticks = np.linspace(0, 50, 2, dtype=int)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax11.set_yticks(-1 * y2_ticks)
    ax11.set_yticklabels(y2_ticklabels, size = 30)
    ax11.set_ylabel('Rainfall (mm)', size = 30)
    ax1.legend(loc = 'upper left', prop={'size':30}, frameon=False, ncol=3, borderpad = 0
                , bbox_to_anchor = (0, 0.90)
                )
    
    ax1.set_xlim([pd.to_datetime('2015-01-01'), pd.to_datetime('2023-10-01')])
    # ax1.set_xlim([obs.index[0], obs.index[-1]])
    ax11.set_frame_on(False)
    
    ax1.set_frame_on(False)
    ax1.yaxis.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.2)
    ax1.yaxis.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.2)
    
    metrics_['flow'] = metric_df.copy()
# %% wq
# extract wq validation data
if False:
    wims_path = os.path.join('D:\\','Huawei','PhD','Year4','GW-WSIMOD','River Lee_Wangdong1','additional_data_LL','wims')
    fids = os.listdir(wims_path)
    fids = list(filter(lambda f: f.endswith('.csv'), fids))
    # extract the data of these stations
    # target variables - if not included, remove the corresponding stations
    target_wq_variables = ['Nitrogen, Total Oxidised as N',
                            'Orthophosphate, reactive as P',
                            'Ammoniacal Nitrogen as N'
                            ]
    station = 'TH-PLER0061'
    val_wq = {k:[] for k in target_wq_variables}
    for fid in fids:
        # read csv and convert it to geodataframe
        wims = pd.read_csv(os.path.join(wims_path, fid))
        wims = wims.groupby('sample.samplingPoint.notation').get_group(station)
        wims = wims[wims['determinand.definition'].isin(target_wq_variables)]
        for k,v in val_wq.items():
            v.append(wims.groupby('determinand.definition').get_group(k)[['sample.sampleDateTime', 'result']])
    
    for k,v in val_wq.items():
        v = pd.concat(v, axis = 0)
        v.columns = ['time', 'value']
        v['time'] = pd.to_datetime(v['time']).dt.date
        v = v.set_index('time')
        val_wq[k] = v
    # save to pickle
    with open('val_wq.pkl', 'wb') as handle:
         pickle.dump(val_wq, handle, protocol=pickle.HIGHEST_PROTOCOL)

# read and validate
if True:
    with open('val_wq.pkl', 'rb') as handle:
        val_wq = pickle.load(handle)
    for k,v in val_wq.items():
        v.index = pd.to_datetime(v.index)
    
    val_arc = '1823-outlet-to-1823-downstream'
    t_start = pd.to_datetime(flows.get_group(val_arc)['time'].unique()[0])
    t_end = pd.to_datetime(flows.get_group(val_arc)['time'].unique()[-1])
    
    import math as math
    from sklearn import metrics
    import numpy as np
    
    ## validate against observed water quality
    val_wq_lookup_england = {#'Ammonia' : 'Ammoniacal Nitrogen as N',
                               'DIN' : 'Nitrogen, Total Oxidised as N',
                              'SRP' : 'Orthophosphate, reactive as P'
                              }
    
    metric_df_wq = pd.DataFrame(index = val_wq_lookup_england.keys(), columns = ['nse', 'rmse', 'r', 'ksg', 'mape', 'pb', 'no_obs'])
    for var in val_wq_lookup_england.keys():
        idx = list(val_wq_lookup_england.keys()).index(var)
        ax1 = axs[idx+1]
    
        # calculate sim_wq
        if var == 'DIN':
            sim_wq = (flows.get_group(val_arc).set_index('time').ammonia + \
                      flows.get_group(val_arc).set_index('time').nitrate + \
                      flows.get_group(val_arc).set_index('time').nitrite) / \
                      flows.get_group(val_arc).set_index('time').flow * constants.KG_M3_TO_MG_L # [mg/l]
        elif var == 'SRP':
            sim_wq = flows.get_group(val_arc).set_index('time').phosphate / \
                      flows.get_group(val_arc).set_index('time').flow * constants.KG_M3_TO_MG_L # [mg/l]
        elif var == 'Ammonia':
            sim_wq = flows.get_group(val_arc).set_index('time').ammonia / \
                      flows.get_group(val_arc).set_index('time').flow * constants.KG_M3_TO_MG_L # [mg/l]
        # calculate obs_wq
        obs_wq = val_wq[val_wq_lookup_england[var]]
        obs_wq.index = pd.to_datetime(obs_wq.index)
        # plot
        x = pd.to_datetime(sim_wq.index)
        y = sim_wq.values
        ax1.plot(x, y, color='r', label = 'sim', linewidth = 4, zorder=1)
        ax1.scatter(obs_wq.index, obs_wq.values, s = 150, c = 'blue', label = 'obs', zorder=2)
        ax1.set_ylabel(var + ' (mg/l)', size = 30)
        ax1.yaxis.set_label_coords(-0.04, 0.5)
        ymax = y.max()
        if obs_wq.any().any():
            #ymax = round(np.nanmax(obs_wq.loc[t_start:t_end]) * 1.5)
            ymax = round(np.nanmax(obs_wq) * 1.5)
            ymax = max(1, ymax)
            if var in ['SRP', 'Ammonia']:
                ymax = 2
            if ymax <= 1:
                y1_ticks = np.linspace(0, ymax, 5, dtype = float)
            else:                
                y1_ticks = np.linspace(0, ymax, 5, dtype = int)
            y1_ticklabels = [str(i) for i in y1_ticks]
            ax1.set_yticks(y1_ticks)
        # ax1.set_xlim([t_start, t_end])
        ax1.set_xlim([pd.to_datetime('2015-01-01'), pd.to_datetime('2023-10-01')])
        ax1.set_ylim([0,ymax])
        ax1.tick_params(labelsize=30)
        ax1.tick_params(axis = 'x', pad = 10)
        ax1.legend(loc = 'upper left', prop={'size':30}, frameon=False, ncol=3, borderpad = 0
                    , bbox_to_anchor = (0, 0.90)
                    )
        ax1.set_frame_on(False)
        ax1.yaxis.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.2)
        ax1.yaxis.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.2)
        # ax1.set_title(var, fontsize = 30)
        
        ####################
        # metric
        ####################
        node = var
        
        sim = pd.DataFrame(sim_wq, columns = ['result'])
        sim.index = pd.to_datetime(sim.index)
        sim = pd.DataFrame(sim.loc[t_start:t_end, 'result'])
        obs = obs_wq.dropna()
        obs.columns = ['result']
        obs.index = pd.to_datetime(obs.index)
        obs = obs.groupby(obs.index).mean()
        ind = sim.index.intersection(obs.index)
        if len(ind) > 0:
            obs = obs.loc[ind]
            sim = sim.loc[ind]
            
            cc = np.corrcoef(obs.values.ravel(), sim.values.ravel())
            r = cc[0,1]
            
            metric_df_wq.loc[node, 'nse'] = (1 - ((obs.sub(sim))**2).mean()/((obs.sub(obs.mean()))**2).mean())[0]
            metric_df_wq.loc[node, 'rmse'] = np.sqrt(metrics.mean_squared_error(obs, sim))
            metric_df_wq.loc[node, 'r'] = r
            metric_df_wq.loc[node, 'mape'] = (np.mean(np.abs((sim - obs) / obs), axis=0) * 100)[0]
            metric_df_wq.loc[node, 'ksg'] = 1 - math.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2 + (sim.mean() / obs.mean() - 1) ** 2)
            metric_df_wq.loc[node, 'pb'] = (sim.sub(obs).sum() / obs.sum())[0]
            metric_df_wq.loc[node, 'no_obs'] = len(obs)
    
    metrics_['wq'] = metric_df_wq
    
    plt.savefig(os.path.join(figs_dir, "Fig_S2.svg"), format="svg", dpi = 300, bbox_inches='tight',)
# %% gw head
if True:
    tanks = pd.read_csv(os.path.join(results_folder, 'tanks.csv'))
    
    tanks['time'] = pd.to_datetime(tanks['time'].astype(str))
    tanks = tanks[(tanks['time'] > '2014-12-31') & (tanks['time'] < '2024-01-01')]

    tanks = tanks.groupby('node')
    
    gw_storage = tanks.get_group('1823-gw')
    # read gw parameters
    import yaml
    yml_dir = os.path.join(results_folder, "settings_separatesewers_stormleakage_generalhead.yaml")
    fs = open(yml_dir, encoding='utf-8')
    yml = yaml.load(fs, Loader=yaml.FullLoader)
    
    s = yml['nodes']['1823-gw']['s']
    datum = yml['nodes']['1823-gw']['datum']
    area = yml['nodes']['1823-gw']['area']
    
    head = gw_storage['storage'] / area / s + datum # head [m]
    head = pd.concat([gw_storage[['node', 'time', 'storage']], head], axis = 1).set_index('time')
    head.columns = ['node', 'storage', 'head']
    
    for var in ['storage', 
                'head']:
        head.index = pd.to_datetime(head.index)
        fig, ax = plt.subplots()
        ax.plot(head.index, head[var])
        ax.set_title(var)
        if var == 'head':
            ax.axhline(yml['nodes']['1823-river']['datum'], linestyle = '--')
    
    # compare with Lea model and observed gw level #############
    file="1823.csv"
    wfdid = file.replace('.csv', '')
    
    # absolute value
    node = wfdid+'-gw'
    f, axs = plt.subplots()
    x = head.groupby('node').get_group(node).index
    y = head.groupby('node').get_group(node)['head']
    axs.plot(x, y, label = 'Simulation')
    axs.set_ylabel(node+' head [m asl]')
    obs = pd.read_csv(file).set_index('date')
    obs.index = pd.to_datetime(obs.index)
    for col in obs.columns:
        axs.scatter(obs.index, obs[col])
    axs.set_xlim([x[0], x[-1]])
    axs.legend()
    
    # average the groundwater level
    # record the total number of boreholes & extract between t_start:t_end
    t_start = pd.to_datetime(head.index[0])
    t_end = pd.to_datetime(head.index[-1])
    
    obs_gw = obs.loc[t_start:t_end]
    # interpolate the groundwater levels
    obs_gw = obs_gw.interpolate()
    # drop nan rows
    obs_gw = obs_gw.dropna()
    # average for all boreholes
    obs_gw_mean = obs_gw.mean(axis = 1)
    obs_gw_mean.name = 'mean'
    
    # generate sgi
    from numpy import linspace
    from scipy.stats import norm
    
    def sgi(series):
        """Method to compute the Standardized Groundwater Index
        :cite:t:`bloomfield_analysis_2013`.
    
        Parameters
        ----------
        series: pandas.Series
    
        Returns
        -------
        sgi_series: pandas.Series
            Pandas time series of the groundwater levels. Time series index should be a
            pandas DatetimeIndex.
        """
        series = series.copy()  # Create a copy to ensure series is untouched.
    
        # Loop over the months
        for month in range(1, 13):
            data = series[series.index.month == month]
            n = data.size  # Number of observations
            pmin = 1 / (2 * n)
            pmax = 1 - pmin
            sgi_values = norm.ppf(linspace(pmin, pmax, n))
            series.loc[data.sort_values().index] = sgi_values
        return series
    
    # generate individual plot - monthly-average
    # time head series plot for specific gw
    node = wfdid+'-gw'
    # fig_width = 36
    # fig_height = 7.5
    # f, axs = plt.subplots(1, 1, figsize=(fig_width,fig_height))
    f, axs = plt.subplots()
    label_sim = 'Simulation'
    label_obs = 'Observation'
    # observation plot - all use the observation index
    obs = obs_gw_mean
    obs = sgi(obs)
    x = obs.index
    axs.plot(x, obs, label = label_obs, linestyle = '--')
    
    y = head.groupby('node').get_group(node)['head'][t_start:t_end]
    y = y.groupby(pd.PeriodIndex(y.index, freq="M")).mean() # average for each month if there are multiple values in a month
    y.index = pd.to_datetime(y.index.astype(str))
    y = y.loc[x]
    y = sgi(y)
    axs.plot(x, y, label = label_sim, c = 'r')
    
    import numpy as np
    r = {}
    cc = np.corrcoef(obs.values.ravel(), y.values.ravel())
    r['luton'] = cc[0,1]
    # Lea gw head ##########################################################
    if False:
        import geopandas as gpd
        #Directories
        data_dir = os.path.join('D:\\','Huawei','PhD','Year4','GW-WSIMOD','River Lee_Wangdong1','v0')
        model_dir = os.path.join(data_dir, "model")
        results_dir = os.path.join(model_dir, "results1", "_gwh_reservoir_timevaryingdemand")
        # read tank results
        with open(os.path.join(results_dir, 'tanks_FullList_check.pkl'), 'rb') as handle:
            tanks = pickle.load(handle)
        tanks.time = pd.to_datetime(tanks.time.astype(str))
        # read input parameters
        flow_para = pd.read_csv(os.path.join(data_dir, os.pardir, 'flow_parameters.csv')).set_index('wfdid')
        flow_para.index = flow_para.index.astype(str)
        # read elevation data
        ele_dir = os.path.join(data_dir, "elevation.csv")
        elevation = pd.read_csv(ele_dir).set_index('wfdid')
        elevation.index = elevation.index.astype(str)
        # read yml
        yml_dir = os.path.join(model_dir, "config_ModifyRecharge_NW_noneABS_gwh_reservoir_timevaryingdemand.yml")
        fs = open(yml_dir, encoding='utf-8')
        yml = yaml.load(fs, Loader=yaml.FullLoader)
        # calculate simulated head
        # read catchment geojson
        cb = gpd.read_file(os.path.join(data_dir, "subcatchments.geojson"))
        cb = cb.set_crs(27700, allow_override=True)
        gw_names = ['1823-gw']
        area = {}
        l = []
        for gw in gw_names:
            # get gw area
            wfdid = gw.split('-')[0]
            area[gw] = float(cb.loc[cb['wfdid'] == wfdid, 'area_m2']) # gw area [m2]
            s = flow_para.loc[wfdid, 's'] # specific yield
            # calculate gw head
            storage = tanks.groupby('node').get_group(gw)
            head = storage['storage'] / area[gw] / s + yml['nodes'][gw]['datum'] # head [m]
            head = pd.concat([storage[['node', 'time']], head], axis = 1)
            head = head.rename({'storage': 'head'}, axis=1) # replace 'storage' with 'head' in the column names
            l.append(head)
        head = pd.concat(l).set_index('time')
        
        x = obs.index[obs.index < head.index[-1]]
        y = head.groupby('node').get_group(node)['head'][t_start:]
        y = y.groupby(pd.PeriodIndex(y.index, freq="M")).mean() # average for each month if there are multiple values in a month
        y.index = pd.to_datetime(y.index.astype(str))
        y = y.loc[x]
        y = sgi(y)
        axs.plot(y.index, y, label = 'Simulation_Lea')
    axs.set_xlim([t_start, t_end])
    axs.set_ylim([-3, 3])
    axs.set_title('Standardised groundwater level index (SGI)')
    axs.legend(frameon = False, ncol = 2, loc = 'upper right')
    # axs.set_ylabel('SGI')
    
    cc = np.corrcoef(obs.loc[x].values.ravel(), y.values.ravel())
    r['lea'] = cc[0,1]
    
    metrics_['gw'] = r['luton']
    
    plt.savefig(os.path.join(figs_dir, "Fig_S3.svg"), format="svg", dpi = 300, bbox_inches='tight',)
# %%
# validate cso
if True:
    
    rainfall = pd.read_csv(os.path.join(results_folder, '1823-land-inputs.csv.gz')).groupby('variable').get_group('precipitation').set_index('time')
    rainfall.index = pd.to_datetime(rainfall.index)
    rainfall = rainfall['value'] / constants.MM_TO_M

    cso = pd.read_csv('cso.csv')
    # basic information on Permit - from https://environment.data.gov.uk/public-register/view/search-water-discharge-consents
    cso_types = {'WwTW/Sewage Treatment Works (water company)': {'val_arc': 'luton_stw-1823-combined-to-1823-outlet',
                                                                'permitnumbers': ['TEMP.2735', 'CANM.0473']},
                'Storm Tank/CSO on Sewerage Network (water company)': {'val_arc': '1823-stormcso-cso-to-1823-river',#'1823-storm-cso-to-1823-river',
                                                                       'permitnumbers': ['TEMP.2984', 'CANM.0550', 'CANM.0549']}}
    
    metric = {k: {} for k in cso_types.keys()}
    # fill_between start-end/offlinestart-offlineend
    ll = []
    for cso_type, info in cso_types.items():
        permitnumbers = info['permitnumbers']
        for pn in permitnumbers:
            df = cso.groupby('PermitNumber').get_group(pn)
            df = df.iloc[::-1].reset_index(drop=True) # reverse order
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            group = 1+permitnumbers.index(pn)/len(permitnumbers)
            df['Group'] = group
            # 
            l = []
            for idx in df.index:
                if not idx % 2: # start
                    if idx+1 in df.index: # end
                        start = df.loc[idx, 'DateTime']
                        end = df.loc[idx+1, 'DateTime']
                        date_insert = pd.date_range(start, end, freq = '15min')
                        df_insert = pd.concat([pd.DataFrame(df.loc[idx]).T]*len(date_insert), ignore_index=True)
                        df_insert['DateTime'] = date_insert
                        df_insert['Group'] = group
                        df_insert['Event'] = idx / 2
                        if 'Offline' in df.loc[idx, 'AlertType']:
                            df_insert['AlertType'] = 'Offline Middle'
                        else:
                            df_insert['AlertType'] = 'Middle'
                        # change the first and last row into 0
                        df_insert.iloc[0, df.columns.get_loc('AlertType')] = df.loc[idx, 'AlertType']
                        df_insert.iloc[-1, df.columns.get_loc('AlertType')] = df.loc[idx+1, 'AlertType']
                        l.append(df_insert)
                    else:
                        print('End is not detected for the spill starting from ', df.loc[idx, 'DateTime'])
            df = pd.concat(l, axis = 0).reset_index(drop=True)
            df = df.sort_values(by=['DateTime'])
            ll.append(df)
    cso_extended = pd.concat(ll, axis = 0).reset_index(drop=True)
    
    t_start = pd.to_datetime('2022-01-01') #
    t_end = pd.to_datetime(flows.get_group(val_arc)['time'].unique()[-1])
    
    # t_start = pd.to_datetime('2023-09-01')
    # t_end = pd.to_datetime(flows.get_group(val_arc)['time'].unique()[-1])
    # t_start = pd.to_datetime('2022-10-01')
    # t_end = pd.to_datetime('2022-12-01')
    # t_start = pd.to_datetime('2023-03-01')
    # t_end = pd.to_datetime('2023-05-01')
    
    import math as math
    from sklearn import metrics
    import numpy as np
    
    fig_width = 36
    fig_height = 7.5
    figs, axs = plt.subplots(2, 1, figsize=(fig_width,fig_height*2))
    titles = {'WwTW/Sewage Treatment Works (water company)': 'Wastewater CSO spill',
                'Storm Tank/CSO on Sewerage Network (water company)': 'Stormwater CSO spill'}
    # metric_df = pd.DataFrame(index = list(val_flows.keys()), columns = ['nse', 'rmse', 'r', 'ksg', 'mape', 'pb'])
        
    for cso_type, info in cso_types.items():
        idx = list(cso_types.keys()).index(cso_type)
        ax1 = axs[idx]
        
        ax1.set_title(titles[cso_type], fontsize = 30)
        
        val_arc = info['val_arc']
        sim_flow = flows.get_group(val_arc).set_index('time').flow / constants.D_TO_S # [m3/s]
        sim_flow.index = pd.to_datetime(sim_flow.index)
        ax1.plot(sim_flow.index, sim_flow, color = 'r', alpha = 1, label = 'sim', linewidth = 4, zorder = 2)
        # plot obs
        permitnumbers = info['permitnumbers']
        # option 1 - vspan
        if False: 
            for pn in permitnumbers:
                obs = cso.groupby('PermitNumber').get_group(pn)
                obs = obs.iloc[::-1].reset_index(drop=True) # reverse order
                obs['DateTime'] = pd.to_datetime(obs['DateTime']).dt.date.values # normalise to date
                for idx in obs.index:
                    if not idx % 2: # start
                        if idx+1 in obs.index: # end
                            start = pd.to_datetime(obs.loc[idx, 'DateTime'])
                            end = pd.to_datetime(obs.loc[idx+1, 'DateTime'])
                            alerttype = obs.loc[idx, 'AlertType']
                            if 'Offline' not in alerttype: # real observed cso
                                ax1.axvspan(start, end, 
                                            color="crimson", alpha=0.3)
                            else:
                                ax1.axvspan(start, end, 
                                            color="blue", alpha=0.1)
            # generate fake legend
            ax1.axvspan(None,None, 
                        label="obs", 
                        color="crimson", alpha=0.3)               
            ax1.axvspan(None,None, 
                        label="unknown", 
                        color="blue", alpha=0.1)
        # option 2 - scatter/vline
        if True:
            for pn in permitnumbers:
                obs = cso_extended.groupby('PermitNumber').get_group(pn)
                obs.loc[obs.index, 'DateTime'] = list(pd.to_datetime(obs['DateTime']).dt.date.values) # normalise to date
                # obs.loc[obs.index, 'DateTime'] = list(pd.to_datetime(obs['DateTime']).values) # normalise to date
                for event_no in obs['Event'].unique():
                    event = obs.groupby('Event').get_group(event_no)
                    alerttype = event['AlertType'].iloc[0]
                    if 'Offline' not in alerttype: # real observed cso
                        # ax1.scatter(event['DateTime'], event['Group'], s = 250, c = 'blue', marker = '*')
                        ax1.vlines(list(event['DateTime'].unique()), ymin = -1, ymax = 10, colors = 'blue', linestyle = '--')
                    else:
                        # ax1.plot(event['DateTime'], event['Group'], linewidth = 10, c = 'blue', alpha = 0.3)
                        ax1.axvspan(event['DateTime'].iloc[0], event['DateTime'].iloc[-1], 
                                    color="blue", alpha=0.1)
            # generate fake legend
            ax1.vlines([], ymin = -1, ymax = 10, colors = 'blue', linestyle = '--', label = 'obs')
            ax1.axvspan(None, None, label = 'unknown',
                        color="blue", alpha=0.1)
        
        ax1.legend()
        ax1.set_ylabel('Flow ($m^3/s$)', size = 30)
        ax1.yaxis.set_label_coords(-0.04, 0.5)
        ymax = 5#max(max(obs.loc[t_start:t_end, 'flow']) * 1.5, 1)
        ax1.set_ylim([0,ymax])
        
        y1_ticks = np.linspace(0, ymax, 6, dtype = int)
        y1_ticklabels = [str(i) for i in y1_ticks]
        ax1.set_yticks(y1_ticks)
        ax1.tick_params(labelsize=30)
        ax1.tick_params(axis = 'x', pad = 10)
        
        # Create second axes, in order to get the bars from the top you can multiply by -1
        ax11 = ax1.twinx()
        ax11.plot(rainfall.index, -rainfall, c = 'black')
        # Now need to fix the axis labels
        max_pre = math.ceil(max(rainfall) * 4)
        y2_ticks = np.linspace(0, 200, 5, dtype=int)
        y2_ticklabels = [str(i) for i in y2_ticks]
        ax11.set_yticks(-1 * y2_ticks)
        y2_ticks = np.linspace(0, 50, 2, dtype=int)
        y2_ticklabels = [str(i) for i in y2_ticks]
        ax11.set_yticks(-1 * y2_ticks)
        ax11.set_yticklabels(y2_ticklabels, size = 30)
        ax11.set_ylabel('Rainfall (mm)', size = 30)
        ax1.legend(loc = 'upper left', prop={'size':30}, frameon=False, ncol=3, borderpad = 0
                    , bbox_to_anchor = (0, 0.90)
                    )
        
        ax1.set_xlim([t_start, t_end])
        ax11.set_frame_on(False)
        
        ax1.set_frame_on(False)
        ax1.yaxis.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.2)
        ax1.yaxis.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.2)
        
        # calculate metrics = percentage of cso spill accurately predicted
        for pn in permitnumbers:
            spill_dates = []
            cso_extended_pn = cso_extended.groupby('PermitNumber').get_group(pn)
            for event_no in cso_extended_pn['Event']:
                event = cso_extended_pn.groupby('Event').get_group(event_no)
                alerttype = event['AlertType'].iloc[0]
                if 'Offline' not in alerttype: # real observed cso
                    spill_dates += list(set(list(pd.to_datetime(event['DateTime']).dt.date.astype(str).values)))
            spill_dates = list(set(spill_dates))
            metric[cso_type][pn] = sim_flow[[date for date in spill_dates if date in sim_flow.index]]
    
    metric_sum = {cso_type: {pn: (v > 0).sum() / len(v) for pn, v in vv.items()} for cso_type, vv in metric.items()}
    plt.savefig(os.path.join(figs_dir, "Fig_S4.svg"), format="svg", dpi = 300, bbox_inches='tight',)
