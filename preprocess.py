# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 09:25:52 2023

@author: leyan
"""
import yaml
import pandas as pd
# %%
# set dates
if True:
    yml_dir = "settings_separatesewers_2cso_generalhead.yaml"
    fs = open(yml_dir, encoding='utf-8')
    yml = yaml.load(fs, Loader=yaml.FullLoader)
    
    yml['dates'] = pd.date_range(start='2000-01-01', end='2023-10-31').astype(str).to_list()
    
    # save as new yml
    yml_dir = "settings_separatesewers_2cso_generalhead.yaml"
    with open(yml_dir, 'w') as f:
        yaml.dump(yml, 
                  f,
                  default_flow_style = False,
                  sort_keys = False,
                  Dumper=yaml.SafeDumper)
# %%
# settings_separatesewers_stormleakage_generalhead.yaml
if False:
    yml_dir = "settings_separatesewers_stormleakage.yaml.txt"
    fs = open(yml_dir, encoding='utf-8')
    yml = yaml.load(fs, Loader=yaml.FullLoader)
    
    generalheads = ['6aead97c-0040-4e31-889d-01f628faf990',
                'fb3a7adf-ae40-4a9f-ad3f-55b3e4d5c6b7',
                '7e0cc125-fe7a-445b-af6b-bf55ac4065f9',
                'e07ddbc6-7158-4a47-b987-eb2b934dd257',
                'e4b324b5-60f9-48c2-9d64-d89d22a5305e',
                '88c7e69b-e4b3-4483-a438-0d6f9046cdee',
                'a057761f-e18e-4cad-84d4-9458edc182ef',
                '2b5397b7-a129-40a6-873d-cb2a0dd7d5b8'
                ]
    
    yml['nodes']['1823-gw']['c_aquifer'] = {}
    for generalhead in generalheads:
        T = 123.6 # [m2/day]
        h_initial = float(pd.read_csv(generalhead+'.csv')['value'].iloc[0])
        yml['nodes'][generalhead] = {'infiltration_threshold': 0.0,
                                    'decays': None,
                                    'name': generalhead,
                                    'datum': -70,
                                    'infiltration_pct': 0.0,
                                    'area': 1000000000000.0,
                                    'type_': 'Groundwater',
                                    'node_type_override': 'Groundwater_h',
                                    'h_initial': h_initial,
                                    'z_surface': 200,
                                    's': 0.01,
                                    'c_riverbed': 1.0e-05,
                                    'c_aquifer':
                                      {'1823-gw': T},
                                    'filename': generalhead+'.csv'}
        yml['nodes']['1823-gw']['c_aquifer'][generalhead] = T
        yml['arcs'][generalhead+'-to-1823-gw'] = {'capacity': 1000000000000000.0,
                                                    'name': generalhead+'-to-1823-gw',
                                                    'preference': 1,
                                                    'type_': 'Arc',
                                                    'in_port': generalhead,
                                                    'out_port': '1823-gw'
                                                    }
        yml['arcs']['1823-gw-to-'+generalhead] = {'capacity': 1000000000000000.0,
                                                    'name': '1823-gw-to-'+generalhead,
                                                    'preference': 1,
                                                    'type_': 'Arc',
                                                    'in_port': '1823-gw',
                                                    'out_port': generalhead
                                                    }
    # save as new yml
    yml_dir = "settings_separatesewers_stormleakage_generalhead.yaml"
    with open(yml_dir, 'w') as f:
        yaml.dump(yml, 
                  f,
                  default_flow_style = False,
                  sort_keys = False,
                  Dumper=yaml.SafeDumper)
# %%
# settings_combinedsewers_stormleakage_generalhead.yaml
if False:
    yml_dir = "settings_separatesewers_stormleakage.yaml"
    fs = open(yml_dir, encoding='utf-8')
    yml = yaml.load(fs, Loader=yaml.FullLoader)
    
    yml['nodes']['1823-cso'] = {
                                'capacity': 1000000000000000.0,
                                'name': '1823-cso',
                                'pipe_time': 0,
                                'chamber_area': 1,
                                'pipe_timearea': '*id001',
                                'chamber_floor': 10,
                                'type_': 'Sewer',
                                'node_type_override': 'Sewer'}
    del yml['arcs']['1823-storm-to-1823-river']
    del yml['arcs']['luton_stw-1823-foul-to-luton_stw-wwtw']
    yml['arcs']['luton_stw-1823-foul-to-1823-cso'] = {'capacity': 1000000000000000.0,
                                                      'name': 'luton_stw-1823-foul-to-1823-cso',
                                                      'preference': 1,
                                                      'type_': 'Arc',
                                                      'in_port': 'luton_stw-wwtw-1823-foul',
                                                      'out_port': '1823-cso'
                                                      }
    yml['arcs']['1823-storm-to-1823-cso'] = {'capacity': 1000000000000000.0,
                                             'name': '1823-storm-to-1823-cso',
                                             'preference': 1,
                                             'type_': 'Arc',
                                             'in_port': '1823-storm',
                                             'out_port': '1823-cso'
                                             }

# %%
# extend climate data [before 2020-12-31 using HadUK; after using ERA5]
if False:
    # read ERA5 data
    import os
    import gzip
    import csv
    ext_prec = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"preprocess", "cso", "precipitation.csv"))[['time', 'tp']] # [m]
    ext_prec['time'] = pd.to_datetime(ext_prec['time'])
    ext_evap = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"preprocess", "cso", "et0.csv"))[['time', 'pev']] # [m]
    ext_evap['time'] = pd.to_datetime(ext_evap['time'])
    ext_temp = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"preprocess", "cso", "temperature.csv"))[['time', 't2m']] # [m]
    ext_temp['time'] = pd.to_datetime(ext_temp['time'])
    # transform from hourly to daily
    ext_prec = pd.DataFrame(ext_prec.groupby([ext_prec['time'].dt.date])['tp'].sum()).reset_index()
    ext_prec = ext_prec[ext_prec['time'] >= pd.Timestamp('2021-01-01').date()]
    ext_prec = ext_prec[ext_prec['time'] < pd.Timestamp('2023-11-01').date()]
    ext_prec.columns = ['time', 'value']
    ext_prec.loc[ext_prec['value'] < 0, 'value'] = 0 # check negative values
    ext_evap = pd.DataFrame(ext_evap.groupby([ext_evap['time'].dt.date])['pev'].sum()).reset_index()
    ext_evap = ext_evap[ext_evap['time'] >= pd.Timestamp('2021-01-01').date()]
    ext_evap = ext_evap[ext_evap['time'] < pd.Timestamp('2023-11-01').date()]
    ext_evap.columns = ['time', 'value']
    ext_temp = pd.DataFrame(ext_temp.groupby([ext_temp['time'].dt.date])['t2m'].mean()).reset_index()
    ext_temp = ext_temp[ext_temp['time'] >= pd.Timestamp('2021-01-01').date()]
    ext_temp = ext_temp[ext_temp['time'] < pd.Timestamp('2023-11-01').date()]
    ext_temp.columns = ['time', 'value']
    ext_temp['value'] -= 273.15 # [degree C] 
    
    ext_ = {'precipitation': ext_prec,
            'et0': ext_evap,
            'temperature': ext_temp}
    # revise the climate input.csv.gz [They are the same!]
    filenames = ['1823-land-inputs.csv.gz', '1823-river-inputs.csv.gz',
                 'chalton_stw-1823-demand-inputs.csv.gz', 'luton_stw-1823-demand-inputs.csv.gz']
    for filename in filenames:
        df = pd.read_csv(filename)
        df_example = df.iloc[-len(ext_prec):]
        l = []
        for var in df['variable'].unique():
            df_var = df.groupby('variable').get_group(var)
            df_extended = pd.DataFrame(df_example)
            df_extended['variable'] = var
            df_extended['time'] = pd.to_datetime(ext_[var]['time']).dt.strftime("%Y-%m-%d %H:%M:%S").values
            if var == 'precipitation':
                df_extended['value'] = ext_[var]['value'].values
            else:
                _ = pd.DataFrame(df_var).set_index('time')
                df_extended['value'] = _.loc['2017-01-01':'2019-11-01', 'value'].values # use 2017-2019 data as a proxy - because weird values from ERA5
            df_var = pd.concat([df_var, df_extended])
            l.append(df_var)
        df = pd.concat(l).reset_index(drop=True)
        # check and save
        open_func = gzip.open
        mode = "wt"
        with open_func(filename, mode, newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(df.columns)
            for key, value in df.iterrows():
                writer.writerow(list(value))
                
    # revise the gw-demand input.csv.gz [use average]
    filename = '1823-gw-demand-inputs.csv.gz'
    df = pd.read_csv(filename)
    df['time'] = pd.to_datetime(df['time'])
    df_example = df.iloc[-len(ext_prec):].copy()
    df_example.loc[df_example.index, 'time'] = pd.to_datetime(ext_prec['time']).dt.strftime("%Y-%m-%d %H:%M:%S").to_list()
    df_example.loc[df_example.index, 'value'] = df['value'].mean()
    df = pd.concat([df, df_example]).reset_index(drop=True)
    # check and save
    open_func = gzip.open
    mode = "wt"
    with open_func(filename, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(df.columns)
        for key, value in df.iterrows():
            writer.writerow(list(value))
    
    # revise the gw level input.csv.gz [use average]
    filenames = os.listdir()
    filenames = [filename for filename in filenames if filename.endswith(".csv") 
                  if '-' in filename]
    
    for filename in filenames:
        df = pd.read_csv(filename)
        df['time'] = pd.to_datetime(df['time'])
        df_example = df.iloc[-len(ext_prec):].copy()
        df_example.loc[df_example.index, 'time'] = pd.to_datetime(ext_prec['time']).dt.strftime("%Y-%m-%d %H:%M:%S").to_list()
        df_example.loc[df_example.index, 'value'] = df['value'].mean()
        df = pd.concat([df, df_example]).reset_index(drop=True)
        # check and save
        df.to_csv(filename, index=False)
        # open_func = gzip.open
        # mode = "wt"
        # with open_func(filename, mode, newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(df.columns)
        #     for key, value in df.iterrows():
        #         writer.writerow(list(value))
    
    # revise the atmospheric deposition input.csv.gz [extend from 2020-12-31 to 2024-01-01][use average]
    filenames = os.listdir()
    filenames = [filename for filename in filenames if filename.endswith(".csv.gz") 
                  if filename.split('-')[0] == '1823' and filename.split('-')[1] == 'land' and filename.split('-')[2] != 'inputs.csv.gz']
    
    for filename in filenames:
        df = pd.read_csv(filename)
        to_extend = df['variable'].unique()
        time_replace = list(pd.to_datetime(ext_prec['time']).dt.to_period('M').unique().astype(str))
        df_example = df.iloc[-len(time_replace):].copy()
        l = []
        for var in to_extend:
            df_original = df.groupby('variable').get_group(var)
            df_extended = pd.DataFrame(df_example)
            df_extended['variable'] = var
            df_extended['time'] = time_replace
            df_extended['value'] = df_original['value'].mean()
            # concat
            df_extended = pd.concat([df_original, df_extended], axis = 0).reset_index(drop=True)
            l.append(df_extended)
        df = pd.concat(l).reset_index(drop=True)
        df.index.name = 'index'
    
        # check and save
        open_func = gzip.open
        mode = "wt"
        with open_func(filename, mode, newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(df.columns)
            for key, value in df.iterrows():
                writer.writerow(list(value))
     
# %%
# extend atmospheric deposition data (to 2020-12-31)
if False: 
    from os import listdir
    import gzip
    import csv
    
    filenames = listdir()
    filenames = [filename for filename in filenames if filename.endswith(".csv.gz") 
                 if filename.split('-')[0] == '1823' and filename.split('-')[1] == 'land' and filename.split('-')[2] != 'inputs.csv.gz']
    
    to_extend = ['nhx-dry', 'nhx-wet', 'noy-dry', 'noy-wet']
    for filename in filenames:
        df = pd.read_csv(filename)
        df_example = df.groupby('variable').get_group('srp-wet')
        for var in to_extend:
            df_extended = pd.DataFrame(df_example)
            df_extended['variable'] = var
            df_extended['value'] = pd.NA
            # set 'time' as the index
            df_extended = df_extended.set_index('time')
            to_replace = df.groupby('variable').get_group(var).set_index('time')
            # keep original values and fillna using average values
            df_extended.loc[to_replace.index, 'value'] = to_replace['value']
            df_extended['value'] = df_extended['value'].fillna(df_extended['value'].mean())
            df_extended = df_extended.reset_index()
            # drop the orginal group
            df = df[df['variable'] != var]
            # concat
            df = pd.concat([df, df_extended], axis = 0).reset_index(drop=True)
        df.index.name = 'index'
    
        # check and save
        open_func = gzip.open
        mode = "wt"
        with open_func(filename, mode, newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(df.columns)
            for key, value in df.iterrows():
                writer.writerow(list(value))
