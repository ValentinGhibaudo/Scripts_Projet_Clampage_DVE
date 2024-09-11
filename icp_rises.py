import numpy as np
import xarray as xr
import pandas as pd
from pycns import CnsStream, CnsReader
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from tools import *
from configuration import *
from params_icp_rises import *
import jobtools
from tqdm import tqdm
from matplotlib.lines import Line2D

def load_table_nory(sub = None):
    nory_file = base_folder / 'documents_valentin' / 'liste_monio_multi_Nory_pour_Valentin.xlsx'
    df_nory = pd.read_excel(nory_file) # load event file
    if not sub is None:
        df_nory = df_nory[df_nory['ID_pseudo'] == sub].reset_index(drop = True) # mask on patient dataframe
    return df_nory

def format_window_dates(df_sub, i , window, win_duration):
    if window == 'baseline':
        date = df_sub.loc[i,'baseline_start_date']
        if isinstance(date, str): # check if date is not a datetime = a str, if True , transtype to datetime
            date = pd.to_datetime(date, dayfirst=True)
        day_time = df_sub.loc[i,'baseline_start_heure']
        h,m,s = str(day_time).split(':')
        start = date + pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        start = pd.to_datetime(start).tz_localize('Europe/Paris').tz_convert('GMT')
        stop = start + pd.Timedelta(win_duration, 'm')
    elif window == 'pre':
        date = df_sub.loc[i,'date_debut_ev']
        if isinstance(date, str): # check if date is not a datetime = a str, if True , transtype to datetime
            date = pd.to_datetime(date, dayfirst=True)
        day_time = df_sub.loc[i,'heure_debut_ev']
        h,m,s = str(day_time).split(':')
        stop = date + pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        stop = pd.to_datetime(stop).tz_localize('Europe/Paris').tz_convert('GMT')
        start = stop - pd.Timedelta(win_duration, 'm')
    elif window == 'per':
        date_start = df_sub.loc[i,'date_debut_ev']
        if isinstance(date_start, str): # check if date is not a datetime = a str, if True , transtype to datetime
            date_start = pd.to_datetime(date_start, dayfirst=True)
        day_time_start = df_sub.loc[i,'heure_debut_ev']
        h,m,s = str(day_time_start).split(':')
        start_local = date_start + pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        start = pd.to_datetime(start_local).tz_localize('Europe/Paris').tz_convert('GMT')

        duration = df_sub.loc[i,'duree_ev']
        h,m,s = str(duration).split(':')
        stop_local = start_local + pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        stop = pd.to_datetime(stop_local).tz_localize('Europe/Paris').tz_convert('GMT')

    return start, stop


##### SPECTRUM JOB
def spectrum(sub, **p):
    df_nory = load_table_nory() # load event file
    df_sub = df_nory[df_nory['ID_pseudo'] == sub].reset_index(drop = True) # mask on patient dataframe
    df_sub = df_sub.dropna(subset = ['baseline_start_heure']).reset_index(drop = True) # remove events where don't have baseline start hour
    n_events = df_sub.shape[0] # count number of events of the patient

    side_lesion = df_sub['lateralite_lesion'].unique()[0] # get lesion side
    translate_side = {'droit':'right','gauche':'left','bilatéral':'right'}
    side_lesion = translate_side[side_lesion] # translate lesion side in english
    
    icp_windows = ['baseline','pre','per'] # list of possible windows
    sides = ['injured','healthy'] # list of possible side types
    side_injuring = {'injured':side_lesion, 'healthy':'left' if side_lesion == 'right' else 'right'} # define injured/healthy sides

    cns_reader = CnsReader(base_folder / data_path / sub) # initialize reader of patient data
    srate = cns_reader.streams['EEG'].sample_rate # load sample rate
    chan_names = cns_reader.streams['EEG'].channel_names # load chan names
    
    win_size_points = int(p['win_size_secs'] * srate) # define welch window size for spectrum computing

    da_spectra = None

    for i in range(n_events): # loop over events = over icp rises
        for icp_window in icp_windows:
            start, stop = format_window_dates(df_sub=df_sub, i=i, window=icp_window, win_duration=p['baseline_pre_win_size_mins']) # get start and stop dates of the window
                    
            for side in sides:
                sigs = cns_reader.streams['EEG'].get_data(sel = slice(start,stop), apply_gain = True) # load sigs with gain applied

                chan1, chan2 = p['derivations'][side_injuring[side]].split('-') # get chan names
                chan1_ind, chan2_ind = chan_names.index(chan1), chan_names.index(chan2) # find indices of chan names
                sig = sigs[:,chan1_ind] - sigs[:,chan2_ind] # bipolarization
                
                f, psd = scipy.signal.welch(sig, srate, nperseg = win_size_points, scaling=p['power_scaling'])
                if da_spectra is None:
                    da_spectra = init_da({'n_event':range(n_events), 'side':sides, 'icp_window':icp_windows, 'freq':f}) # initialize empty dataarray at first iteration
                da_spectra.loc[i, side, icp_window,:] = psd # store power spectrum at the right localization

    ds = xr.Dataset() # initialize xarray dataset
    ds['spectrum'] = da_spectra # store datarray with spectra in dataset
    return ds # return dataset

def test_spectrum(sub):
    print(sub)
    ds = spectrum(sub, **spectrum_params)
    print(ds['spectrum'])

spectrum_job = jobtools.Job(precomputedir, 'spectrum', spectrum_params, spectrum)
jobtools.register_job(spectrum_job)

##### SPECTRAL FEATURES JOB
def spectral_features(sub, **p):
    da_spectra = spectrum_job.get(sub)['spectrum'] # load spectrum
    freqs = da_spectra['freq'].values # get frequency vector

    rows = []

    for i in da_spectra['n_event'].values: # loop over events = over icp rises
        for side in da_spectra['side'].values: # loop over sides
            for icp_window in da_spectra['icp_window'].values: # loop over icp windows
                power_spectra = da_spectra.loc[i, side, icp_window,:].values # load power spectra               
                slope = compute_spectrum_log_slope(spectrum = power_spectra.copy(), # compute slope of spectrum
                                                   freqs = freqs,
                                                   freq_range=p['freq_range_slope']
                                                   )
                spectral_entropy = compute_spectral_entropy(power_spectra.copy(), # compute spectral entropy
                                                            normalized = p['normalize_entropy']
                                                            ) 
                mask_delta = (freqs >= p['freq_range_delta'][0]) & (freqs < p['freq_range_delta'][1]) # create delta freq mask
                mask_alpha= (freqs >= p['freq_range_alpha'][0]) & (freqs < p['freq_range_alpha'][1]) # create alpha freq mask
                total_power = np.sum(power_spectra)
                delta_power = np.trapz(power_spectra[mask_delta]) # trapezoïdal rule to compute delta power
                alpha_power = np.trapz(power_spectra[mask_alpha]) # trapezoïdal rule to compute alpha power
                adr_power = alpha_power / delta_power # alpha / delta ratio
                delta_power_normalized = delta_power / total_power # trapezoïdal rule to compute delta power
                alpha_power_normalized = alpha_power / total_power# trapezoïdal rule to compute alpha power
                adr_power_normalized = alpha_power_normalized / delta_power_normalized # alpha / delta ratio
                row = [sub, i, side, icp_window, slope, delta_power, alpha_power, adr_power, spectral_entropy, delta_power_normalized, alpha_power_normalized , adr_power_normalized] # store elements in a list
                rows.append(row)
    df_spectral_features = pd.DataFrame(rows, columns = ['patient','n_event','side','icp_window','slope','delta_power','alpha_power','ADRatio','spectral_entropy','delta_power_normalized', 'alpha_power_normalized' , 'ADRatio_normalized']) # store in dataframe
    ds = xr.Dataset(df_spectral_features) # store dataframe in dataset
    return ds # return dataset

def test_spectral_features(sub):
    print(sub)
    ds = spectral_features(sub, **spectral_features_params)
    print(ds.to_dataframe())
    # print(ds.to_dataframe()[['delta_power','delta_power_normalized']])
    # print(ds.to_dataframe()[['alpha_power','alpha_power_normalized']])

spectral_features_job = jobtools.Job(precomputedir, 'spectral_features', spectral_features_params, spectral_features)
jobtools.register_job(spectral_features_job)

##### SUPPRRESSION
def suppression(sub, **p):
    df_nory = load_table_nory() # load event file
    df_sub = df_nory[df_nory['ID_pseudo'] == sub].reset_index(drop = True) # mask on patient dataframe
    df_sub = df_sub.dropna(subset = ['baseline_start_heure']).reset_index(drop = True) # remove events where don't have baseline start hour
    n_events = df_sub.shape[0] # count number of events of the patient

    side_lesion = df_sub['lateralite_lesion'].unique()[0] # get lesion side
    translate_side = {'droit':'right','gauche':'left','bilatéral':'right'}
    side_lesion = translate_side[side_lesion] # translate lesion side in english
    
    icp_windows = ['baseline','pre','per'] # list of possible windows
    sides = ['injured','healthy'] # list of possible side types
    side_injuring = {'injured':side_lesion, 'healthy':'left' if side_lesion == 'right' else 'right'} # define injured/healthy sides

    cns_reader = CnsReader(base_folder / data_path / sub) # initialize reader of patient data
    srate = cns_reader.streams['EEG'].sample_rate # load sample rate
    chan_names = cns_reader.streams['EEG'].channel_names # load chan names
    
    rows = []

    for i in range(n_events): # loop over events = over icp rises
        for icp_window in icp_windows:
            start, stop = format_window_dates(df_sub=df_sub, i=i, window=icp_window, win_duration=p['baseline_pre_win_size_mins']) # get start and stop dates of the window
                    
            for side in sides:
                sigs = cns_reader.streams['EEG'].get_data(sel = slice(start,stop), apply_gain = True) # load sigs with gain applied

                chan1, chan2 = p['derivations'][side_injuring[side]].split('-') # get chan names
                chan1_ind, chan2_ind = chan_names.index(chan1), chan_names.index(chan2) # find indices of chan names
                sig = sigs[:,chan1_ind] - sigs[:,chan2_ind] # bipolarization
                sig = sig - np.median(sig) # centering sig by removing median
                suppression_ratio_homemade = compute_suppression_ratio_homemade(sig,
                                                                                srate,
                                                                                threshold_μV=p['threshold_µV'],
                                                                                lowcut = p['bandpass_cutoff'][0], 
                                                                                highcut = p['bandpass_cutoff'][1]
                                                                                )

                sig_filtered = iirfilt(sig, srate, lowcut = p['bandpass_cutoff'][0], highcut = p['bandpass_cutoff'][1]) # IIR butterworh 4th order filter
                sig_filtered_centered = sig_filtered - np.median(sig_filtered) # centering by median subtraction
                suppression_ratio, _ = compute_suppression_ratio(sig_filtered_centered, # compute dynamic of suppression ratio
                                          srate, 
                                          threshold_µV = p['threshold_µV'], 
                                          win_size_sec_epoch = p['win_size_sec_epoch'], 
                                          win_size_sec_moving = p['win_size_sec_moving']
                                          )
                
                mean_suppression_ratio = np.mean(suppression_ratio) # compute mean suppresion ratio

                row = [sub, i, side, icp_window, mean_suppression_ratio, suppression_ratio_homemade] # store elements in a list
                rows.append(row)

    df_suppression = pd.DataFrame(rows, columns = ['patient','n_event','side','icp_window','suppression_ratio', 'suppression_ratio_homemade']) # store in dataframe
    ds = xr.Dataset(df_suppression) # store dataframe in dataset
    return ds # return dataset

def test_suppression(sub):
    print(sub)
    ds = suppression(sub, **suppression_params)
    print(ds.to_dataframe())

suppression_job = jobtools.Job(precomputedir, 'suppression', suppression_params, suppression)
jobtools.register_job(suppression_job)


##### PRx
def PRx(sub, **p):
    df_nory = load_table_nory() # load event file
    df_sub = df_nory[df_nory['ID_pseudo'] == sub].reset_index(drop = True) # mask on patient dataframe
    df_sub = df_sub.dropna(subset = ['baseline_start_heure']).reset_index(drop = True) # remove events where don't have baseline start hour
    n_events = df_sub.shape[0] # count number of events of the patient

    cns_reader = CnsReader(base_folder / data_path / sub) # initialize reader of patient data

    prx_r, prx_pval, dates = compute_prx(cns_reader, 
                                         wsize_mean_secs = p['wsize_mean_secs'], 
                                         wsize_corr_mins = p['wsize_corr_mins'], 
                                         overlap_corr_prop =  p['overlap_corr_prop']
    )
    icp_windows = ['baseline','pre','per'] # list of possible windows

    rows = []

    for i in range(n_events): # loop over events = over icp rises
        for icp_window in icp_windows:
            start, stop = format_window_dates(df_sub=df_sub, i=i, window=icp_window, win_duration=p['baseline_pre_win_size_mins']) # get start and stop dates of the window
            
            mask_dates = (dates > np.datetime64(start)) & (dates < np.datetime64(stop))
            try:
                local_prx = prx_r[mask_dates] # mask prx computed between start and stop of the window
                median_prx = np.median(local_prx) # compute median of prx of the window
                start_prx, stop_prx = local_prx[0], local_prx[-1]
                delta_prx = stop_prx - start_prx
            except:
                median_prx = None
                start_prx, stop_prx = None, None
                delta_prx = None
            row = [sub, i, icp_window, start_prx, median_prx, stop_prx, delta_prx] # store elements in a list
            rows.append(row)

    df_prx = pd.DataFrame(rows, columns = ['patient','n_event','icp_window', 'start_prx', 'median_prx', 'stop_prx', 'delta_prx']) # store in dataframe
    ds = xr.Dataset(df_prx) # store dataframe in dataset
    return ds # return dataset

def test_PRx(sub):
    print(sub)
    ds = PRx(sub, **PRx_params)
    print(ds.to_dataframe())

PRx_job = jobtools.Job(precomputedir, 'PRx', PRx_params, PRx)
jobtools.register_job(PRx_job)


##### ICP_PPC
def ICP_PPC(sub, **p):
    df_nory = load_table_nory() # load event file
    df_sub = df_nory[df_nory['ID_pseudo'] == sub].reset_index(drop = True) # mask on patient dataframe
    df_sub = df_sub.dropna(subset = ['baseline_start_heure']).reset_index(drop = True) # remove events where don't have baseline start hour
    n_events = df_sub.shape[0] # count number of events of the patient

    cns_reader = CnsReader(base_folder / data_path / sub) # initialize reader of patient data
    all_streams = cns_reader.streams.keys()
    if 'ABP_Mean' in all_streams:
        abp_stream_name = 'ABP_Mean'
    elif 'ART_Mean' in all_streams:
        abp_stream_name = 'ART_Mean'
    icp_stream_name = 'ICP_Mean'
    icp_windows = ['baseline','pre','per'] # list of possible windows

    stream_names = [icp_stream_name,abp_stream_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    dates = ds['times'].values
    abp = ds[abp_stream_name].values
    icp = ds[icp_stream_name].values

    rows = []

    for i in range(n_events): # loop over events = over icp rises
        for icp_window in icp_windows:
            start, stop = format_window_dates(df_sub=df_sub, i=i, window=icp_window, win_duration=p['baseline_pre_win_size_mins']) # get start and stop dates of the window
            mask_dates = (dates > np.datetime64(start)) & (dates < np.datetime64(stop))
            local_icp = icp[mask_dates]
            local_abp = abp[mask_dates]
            if np.sum(np.isnan(local_icp)) == 0:
                median_icp = np.median(local_icp)
                start_icp = local_icp[0]
                stop_icp = local_icp[-1]
                delta_icp = stop_icp - start_icp
            else:
                median_icp = None
                start_icp = None
                stop_icp = None
                delta_icp = None
            if np.sum(np.isnan(local_abp)) == 0:
                median_abp = np.median(local_abp)
                start_abp = local_abp[0]
                stop_abp = local_abp[-1]
                delta_abp = stop_abp - start_abp
            else:
                median_abp = None
                start_abp = None
                stop_abp = None
                delta_abp = None
            if np.sum(np.isnan(local_abp)) == 0 and np.sum(np.isnan(local_icp)) == 0:
                local_ppc = local_abp - local_icp
                median_ppc = np.median(local_ppc)
                start_ppc = local_ppc[0]
                stop_ppc = local_ppc[-1]
                delta_ppc = stop_ppc - start_ppc
            else:
                median_ppc = None
                start_ppc = None
                stop_ppc = None
                delta_ppc = None
            row = [sub, i, icp_window, start_icp, median_icp,  stop_icp, delta_icp,  start_abp, median_abp, stop_abp, delta_abp, start_ppc, median_ppc, stop_ppc, delta_ppc] # store elements in a list
            rows.append(row)

    df_icp_abp_ppc = pd.DataFrame(rows, columns = ['patient','n_event','icp_window', 'start_icp', 'median_icp',  'stop_icp', 'delta_icp',  'start_abp', 'median_abp', 'stop_abp', 'delta_abp', 'start_ppc', 'median_ppc', 'stop_ppc', 'delta_ppc']) # store in dataframe
    ds = xr.Dataset(df_icp_abp_ppc) # store dataframe in dataset
    return ds # return dataset

def test_ICP_PPC(sub):
    print(sub)
    ds = ICP_PPC(sub, **ICP_PPC_params)
    print(ds.to_dataframe())

ICP_PPC_job = jobtools.Job(precomputedir, 'ICP_PPC', ICP_PPC_params, ICP_PPC)
jobtools.register_job(ICP_PPC_job)


# CONCAT RESULTS
def concat_res_icp_rises(global_key, **p):
    df_nory = load_table_nory() # load event file
    subs = list(df_nory['ID_pseudo'].unique())
    concat = []
    for sub in subs:
        # print(sub)
        spectral_features = spectral_features_job.get(sub).to_dataframe()
        suppression = suppression_job.get(sub).to_dataframe().drop(columns = ['patient','n_event','side','icp_window'])
        prx = PRx_job.get(sub).to_dataframe()
        icp_abp_ppc = ICP_PPC_job.get(sub).to_dataframe()
        concat_sub = pd.concat([spectral_features,suppression], axis = 1)
        concat_sub_set_ind = concat_sub.set_index(['patient','n_event','icp_window'])
        for metric_type in ['prx','icp','abp','ppc']:
            if metric_type == 'prx':
                df_search = prx.copy()
            else:
                df_search = icp_abp_ppc.copy()
            for win_metric in ['start','median','stop','delta']:
                metric_name = f'{win_metric}_{metric_type}'
                concat_sub_set_ind[metric_name] = None
                for i, row in df_search.iterrows():
                    n_event = row['n_event']
                    icp_window = row['icp_window']
                    concat_sub_set_ind.loc[(sub, n_event, icp_window),metric_name] = row[metric_name]
        concat.append(concat_sub_set_ind.reset_index())
    df_concat = pd.concat(concat)
    df_concat.to_excel(base_folder / 'figures' / 'icp_rises3' / 'res_nory.xlsx')
    ds = xr.Dataset(df_concat) # store dataframe in dataset

    return ds # return dataset

def test_concat_res_icp_rises():
    ds = concat_res_icp_rises('all', **concat_res_icp_rises_params)
    print(ds.to_dataframe())

concat_res_icp_rises_job = jobtools.Job(precomputedir, 'concat_res_icp_rises', concat_res_icp_rises_params, concat_res_icp_rises)
jobtools.register_job(concat_res_icp_rises_job)

##### RESULTS
def by_subject_results():

    lowcut = 0.5
    highcut = 40
    ylim_ratio = 10

    save_folder = base_folder / 'figures' / 'icp_rises'

    plots = ['spectra_healthy','spectra_injured','slope',
             'delta_power','alpha_power','ADRatio',
             'spectral_entropy','suppression_ratio_homemade','suppression_ratio',
             'prx'
             ]
    nrows = 4
    ncols = 3
    subplot_pos = attribute_subplots(plots, nrows, ncols)

    for sub in get_run_keys():
    # for sub in sub_keys:
        print(sub)

        gby = ['side','icp_window']
        spectra = spectrum_job.get(sub)
        spectral_features = spectral_features_job.get(sub)
        suppression = suppression_job.get(sub)
        prx = PRx_job.get(sub)
        if spectra is None or spectral_features is None or suppression is None or prx is None:
            print('BUG')
            continue
        spectra = spectra['spectrum']
        spectra = spectra.mean('n_event')
        min_spectra = spectra.min()
        max_spectra = spectra.max()
        freqs = spectra['freq'].values

        mask_freqs = (freqs >= lowcut) & (freqs < highcut)
        spectral_features = spectral_features.to_dataframe()
        n_events = spectral_features['n_event'].unique().size
        suppression = suppression.to_dataframe()
        prx = prx.to_dataframe()

        sides = spectra['side'].values
        icp_windows = spectra['icp_window'].values
        

        fig, axs = plt.subplots(nrows, ncols, figsize = (10, 10), constrained_layout = True)
        suptitle = f'{sub}\nN events : {n_events}'
        fig.suptitle(suptitle, fontsize = 12, y = 1.05)

        for metric, pos in subplot_pos.items():
            ax = axs[pos[0], pos[1]]

            if metric in ['spectra_healthy','spectra_injured']:
                side = metric.split('_')[-1]
                for icp_window in icp_windows:
                    m = spectra.loc[side, icp_window,:].values
                    ax.semilogy(freqs[mask_freqs], m[mask_freqs], lw = 0.8, label = icp_window)
                ax.legend()
                ax.set_title(f'{side} side')
                ax.set_ylabel('Power (µV²)')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylim(min_spectra - min_spectra / ylim_ratio,  max_spectra + max_spectra / ylim_ratio)
            
            if metric in ['slope','delta_power','alpha_power','ADRatio','spectral_entropy','suppression_ratio_homemade','suppression_ratio']:
                if metric in ['slope','delta_power','alpha_power','ADRatio','spectral_entropy']:
                    df = spectral_features.copy()
                elif metric in ['suppression_ratio_homemade','suppression_ratio']:
                    df = suppression.copy()  

                sns.pointplot(data = df, 
                            x = 'icp_window',
                            y = metric,
                            hue = 'side',
                            order = icp_windows,
                            ax=ax
                            )

            if metric == 'prx':
                sns.pointplot(data = prx, 
                            x = 'icp_window',
                            y = 'median_PRx',
                            order = icp_windows,
                            ax=ax
                            )
                
        for ax in [axs[3,1], axs[3,2]]:
            ax.remove()

        fig.savefig(save_folder / f'{sub}.png', bbox_inches = 'tight', dpi = 500)
        plt.close('all')

def by_subject_results2():
    lowcut = 0.5
    highcut = 40
    ylim_ratio = 10

    save_folder = base_folder / 'figures' / 'icp_rises3'

    plots = ['spectra_healthy','spectra_injured','slope',
             'delta_power','alpha_power','ADRatio',
             'delta_power_normalized','alpha_power_normalized','ADRatio_normalized',
             'spectral_entropy','suppression_ratio_homemade','suppression_ratio',
             'icp','abp','ppc',
             'prx'
             ]
    nrows = 4
    ncols = 4
    subplot_pos = attribute_subplots(plots, nrows, ncols)

    colors_side = {'healthy':'tab:blue','injured':'darkorange'}
    linestyles = list(Line2D.lineStyles.keys())

    for sub in get_run_keys():
    # for sub in sub_keys[:1]:
    # for sub in sub_keys:
        print(sub)

        gby = ['side','icp_window']
        spectra = spectrum_job.get(sub)
        spectral_features = spectral_features_job.get(sub)
        suppression = suppression_job.get(sub)
        prx = PRx_job.get(sub)
        icp_abp_ppc = ICP_PPC_job.get(sub)

        if spectra is None or spectral_features is None or suppression is None or prx is None or icp_abp_ppc is None:
            print('BUG')
            continue
        spectra = spectra['spectrum']
        spectra = spectra.mean('n_event')
        min_spectra = spectra.min()
        max_spectra = spectra.max()
        freqs = spectra['freq'].values

        mask_freqs = (freqs >= lowcut) & (freqs < highcut)
        spectral_features = spectral_features.to_dataframe()
        event_labels = spectral_features['n_event'].unique()
        n_events = event_labels.size
        suppression = suppression.to_dataframe()
        prx = prx.to_dataframe()
        icp_abp_ppc = icp_abp_ppc.to_dataframe()
        sides = spectra['side'].values
        icp_windows = spectra['icp_window'].values
        

        fig, axs = plt.subplots(nrows, ncols, figsize = (11, 12), constrained_layout = True)
        suptitle = f'{sub}\nN events : {n_events}'
        fig.suptitle(suptitle, fontsize = 12, y = 1.05)

        for metric, pos in subplot_pos.items():
            ax = axs[pos[0], pos[1]]

            if metric in ['spectra_healthy','spectra_injured']:
                side = metric.split('_')[-1]
                for icp_window in icp_windows:
                    m = spectra.loc[side, icp_window,:].values
                    ax.semilogy(freqs[mask_freqs], m[mask_freqs], lw = 0.8, label = icp_window)
                ax.legend()
                ax.set_title(f'{side} side')
                ax.set_ylabel('Power (µV²)')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylim(min_spectra - min_spectra / ylim_ratio,  max_spectra + max_spectra / ylim_ratio)
            
            if metric in ['slope','delta_power','alpha_power','ADRatio','spectral_entropy','suppression_ratio_homemade','suppression_ratio''slope','delta_power_normalized','alpha_power_normalized','ADRatio_normalized']:
                if metric in ['slope','delta_power','alpha_power','ADRatio','spectral_entropy','delta_power_normalized','alpha_power_normalized','ADRatio_normalized']:
                    df = spectral_features.copy()
                elif metric in ['suppression_ratio_homemade','suppression_ratio']:
                    df = suppression.copy()  


                for i, event_label in enumerate(event_labels):
                    for side in sides:
                        df_plot = df[(df['n_event'] == event_label) & (df['side'] == side)]
                        y = df_plot[metric].values
                        x = range(y.size)
                        ax.plot(x, y, color = colors_side[side], label = f'{side} - N:{i}', lw = 2, ls = linestyles[i])
                        ax.scatter(x, y, color = colors_side[side])
                        ax.set_xticks(x, labels = df_plot['icp_window'].unique())
                        ax.set_ylabel(metric)
                ax.legend(fontsize = 5, loc = 1)

            if metric in ['prx','icp','abp','ppc']:
                if metric in ['prx']:
                    df_plot = prx.copy()
                else:
                    df_plot = icp_abp_ppc.copy()
                for i, event_label in enumerate(event_labels):
                    df_plot_sel = df_plot[(df_plot['n_event'] == event_label)]
                    y = df_plot_sel['median_PRx'].values
                    x = range(y.size)
                    ax.plot(x, y, lw = 2, ls = linestyles[i], color = 'k', label = f'N:{i}')
                    ax.scatter(x, y, color = 'k')
                    ax.set_xticks(x, labels = df_plot_sel['icp_window'].unique())
                    ax.set_ylabel('PRx')
                    ax.legend(loc = 1, fontsize = 6)
                
        # for ax in [axs[3,1], axs[3,2]]:
        #     ax.remove()

        fig.savefig(save_folder / f'{sub}.png', bbox_inches = 'tight', dpi = 500)
        plt.close('all')

def pooled_subject_results():
    import ghibtools as gh

    save_folder = base_folder / 'figures' / 'icp_rises3'

    concat_spectral_features = []
    concat_suppression = []
    concat_prx = []
    concat_icp_abp_ppc = []
    for sub in get_run_keys():
    # for sub in sub_keys:
        gby = ['patient','side','icp_window']
        spectral_features = spectral_features_job.get(sub)
        suppression = suppression_job.get(sub)
        prx = PRx_job.get(sub)
        icp_abp_ppc = ICP_PPC_job.get(sub)
        if spectral_features is None or suppression is None or prx is None or icp_abp_ppc is None:
            print('BUG')
            continue

        spectral_features = spectral_features.to_dataframe()
        spectral_features = spectral_features.groupby(gby).mean(True).reset_index().drop(columns = ['n_event'])
        suppression = suppression.to_dataframe()
        suppression = suppression.groupby(gby).mean(True).reset_index().drop(columns = ['n_event'])
        prx = prx.to_dataframe()
        prx = prx.groupby(['patient','icp_window']).mean(True).reset_index()
        icp_abp_ppc = icp_abp_ppc.to_dataframe()
        icp_abp_ppc = icp_abp_ppc.groupby(['patient','icp_window']).mean(True).reset_index()

        concat_spectral_features.append(spectral_features)
        concat_suppression.append(suppression)
        concat_prx.append(prx)
        concat_icp_abp_ppc.append(icp_abp_ppc)
    
    order = ['baseline','pre','per']

    concat_spectral_features = pd.concat(concat_spectral_features)
    concat_suppression = pd.concat(concat_suppression)
    concat_prx = pd.concat(concat_prx)
    concat_icp_abp_ppc = pd.concat(concat_icp_abp_ppc)

    sides = list(concat_spectral_features['side'].unique())
    icp_windows = list(concat_spectral_features['icp_window'].unique())

    nrows = 14
    fig, axs = plt.subplots(nrows = nrows, ncols = len(sides), figsize = (9, nrows * 3), constrained_layout = True)

    for c, side in enumerate(sides):        
        for r, feature in zip([0,1,2,3,4,5,6,7,8,9],['slope','delta_power','alpha_power','ADRatio','spectral_entropy','suppression_ratio','suppression_ratio_homemade','delta_power_normalized','alpha_power_normalized','ADRatio_normalized']):
                        ax = axs[r,c]
                        if 'suppression' in feature:
                            df = concat_suppression.copy()
                        else:
                            df = concat_spectral_features.copy()
                        
                        gh.auto_stats(df = df[df['side'] == side],
                                            predictor = 'icp_window',
                                            outcome = feature,
                                            design = 'within',
                                            subject = 'patient',
                                            ax=ax,
                                            order = order,
                                            xtick_info=False
                        )
                        ax.set_ylabel(f'{ax.get_ylabel()}\n{side}')

    for r, metric in zip([10,11,12,13],['median_PRx','median_icp','median_abp','median_ppc']):
        ax = axs[r,0]
        if r == 10:
            df_plot = concat_prx.copy()
        else:
            df_plot = concat_icp_abp_ppc.copy()
        gh.auto_stats(df = df_plot,
                    predictor = 'icp_window',
                    outcome = metric,
                    design = 'within',
                    subject = 'patient',
                    ax=ax,
                    order = order,
                    xtick_info=False
                        )

        ax.set_ylabel(f'{ax.get_ylabel()}\n{side}')
    
    fig.savefig(save_folder / f'pooled.png', bbox_inches = 'tight', dpi = 500)
    plt.close('all')
    
def get_srate_from_dates(dates):
    return 1 / np.median(np.diff(dates) / np.timedelta64(1, 's'))

def resample_traces(new_dates, old_dates, trace_to_resample):
    sample_rate_new = get_srate_from_dates(new_dates)
    start = new_dates[0]
    stop = new_dates[-1]
    period_ns = np.int64(1/sample_rate_new * 1e9)
    common_times = np.arange(start.astype('datetime64[ns]').astype('int64'),
                            stop.astype('datetime64[ns]').astype('int64'),
                            period_ns).astype('datetime64[ns]')

    old_dates = old_dates.astype('datetime64[ns]')
    f = scipy.interpolate.interp1d(old_dates.astype('int64'), trace_to_resample, kind='linear', axis=0,
                                copy=True, bounds_error=False,
                                fill_value=np.nan, assume_sorted=True)
    trace_resampled = f(common_times.astype('int64'))
    if new_dates.size - trace_resampled.size != 0:
        size_points_to_add = new_dates.size - trace_resampled.size
        trace_resampled = np.append(trace_resampled, np.full(size_points_to_add, np.median(trace_resampled)))
    return trace_resampled


def erp_like_fig_sub(sub, **p):

    savefolder = base_folder / 'figures' / 'pour_nory'

    N_rolling = p['N_rolling']
    win_duration_mins = p['win_duration_mins']
    freq_bands = p['freq_bands']
    derivations = p['derivations']
    sides = ['injured','healthy']# list of possible side types

    df_nory = load_table_nory()
    df_sub = df_nory[df_nory['ID_pseudo'] == sub].reset_index(drop = True) # mask on patient dataframe
    df_sub = df_sub.dropna(subset = ['baseline_start_heure']).reset_index(drop = True) # remove events where don't have baseline start hour
    n_events = df_sub.shape[0] # count number of events of the patient

    side_lesion = df_sub['lateralite_lesion'].unique()[0] # get lesion side
    translate_side = {'droit':'right','gauche':'left','bilatéral':'right'}
    side_lesion = translate_side[side_lesion] # translate lesion side in english
    
    icp_windows = ['baseline','pre','per'] # list of possible windows
    sides = ['injured','healthy'] # list of possible side types
    side_injuring = {'injured':side_lesion, 'healthy':'left' if side_lesion == 'right' else 'right'} # define injured/healthy sides

    cns_reader = CnsReader(base_folder / data_path / sub) # initialize reader of patient data
    eeg_srate = cns_reader.streams['EEG'].sample_rate
    eeg_chan_names = cns_reader.streams['EEG'].channel_names # load chan names

    prx_trace, _, dates_prx = compute_prx(cns_reader)

    all_streams = cns_reader.streams.keys()
    if 'ABP_Mean' in all_streams:
        abp_stream_name = 'ABP_Mean'
    elif 'ART_Mean' in all_streams:
        abp_stream_name = 'ART_Mean'
    icp_stream_name = 'ICP_Mean'

    stream_names = [icp_stream_name,abp_stream_name]
    # srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    srate = 1
    win_size_points = int(win_duration_mins * 60 * srate)
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    
    dates = ds['times'].values
    abp = ds[abp_stream_name].values
    icp = ds[icp_stream_name].values

    prx_resampled = resample_traces(new_dates = dates, 
                                    old_dates = dates_prx, 
                                    trace_to_resample = prx_trace
                                    )

    rows = []

    da_non_eeg = None
    da_eeg = None

    for i in range(n_events): # loop over events = over icp rises
        start_pre, _ = format_window_dates(df_sub=df_sub, i=i, window='pre', win_duration=baseline_pre_win_size_mins) # get start and stop dates of the window
        _, stop_per = format_window_dates(df_sub=df_sub, i=i, window='per', win_duration=baseline_pre_win_size_mins) # get start and stop dates of the window
        # start_pre = start_pre - np.timedelta64(60, 'm')
        stop_per_mins = (stop_per - start_pre).seconds / 60
        stop_per_mins = stop_per_mins if stop_per_mins < win_duration_mins else win_duration_mins
        stop_win = start_pre + np.timedelta64(win_duration_mins, 'm')
        mask_dates = (dates > np.datetime64(start_pre)) & (dates < np.datetime64(stop_win))
        local_dates = dates[mask_dates]
        local_icp = icp[mask_dates]
        local_abp = abp[mask_dates]
        local_ppc = local_abp - local_icp
        local_prx = prx_resampled[mask_dates]

        local_times_mins = (np.arange(local_abp.size) / srate) / 60


        local_eeg_sigs, local_dates_eeg = cns_reader.streams['EEG'].get_data(sel = slice(start_pre,stop_win), apply_gain = True, with_times = True) # load sigs with gain applied
        local_eeg_sig_sides = {'delta':{'healthy':None,'injured':None}, 'theta':{'healthy':None,'injured':None}, 'alpha':{'healthy':None,'injured':None}}
        for side in sides:
            chan1, chan2 = derivations[side_injuring[side]].split('-') # get chan nameschan1, chan2 = p['derivations'][side_injuring[side]].split('-') # get chan names
            chan1_ind, chan2_ind = eeg_chan_names.index(chan1), eeg_chan_names.index(chan2) # find indices of chan names
            local_eeg_sig = local_eeg_sigs[:,chan1_ind] - local_eeg_sigs[:,chan2_ind] # bipolarization
            for band, cuts in freq_bands.items():
                local_amp = get_amp(iirfilt(local_eeg_sig, eeg_srate, lowcut=cuts[0], highcut = cuts[1]))
                local_amp_resampled = resample_traces(new_dates = local_dates, old_dates=local_dates_eeg, trace_to_resample=local_amp)
                local_eeg_sig_sides[band][side] = local_amp_resampled
        if da_non_eeg is None:
            da_non_eeg = init_da({'dtype':['icp','abp','ppc','prx'], 'event':range(n_events), 'time':local_times_mins})
            da_non_eeg.attrs['srate'] = srate
        if da_eeg is None:
            da_eeg = init_da({'band':['delta','theta','alpha'], 'side':sides, 'event':range(n_events), 'time':local_times_mins})
            da_eeg.attrs['srate'] = srate

        fig, axs = plt.subplots(nrows = 7, sharex=True, figsize = (9, 6))
        fig.subplots_adjust(hspace = 0)
        fig.suptitle(f'{sub}\nEvent n°{i+1}', fontsize = 15)
        for r, trace, label in zip([0,1,2,3,4,5,6],[local_icp, local_abp, local_ppc, local_prx, None , None, None],['icp','abp','ppc','prx','delta','theta','alpha']):
            ax = axs[r]
            if label in ['icp','abp','ppc','prx']:
                trace = pd.Series(trace).rolling(N_rolling).median().values
                ax.plot(local_times_mins, trace, color = 'k', lw = 2)
                da_non_eeg.loc[label,i,:] = trace
            else:
                for side in sides:
                    trace = local_eeg_sig_sides[label][side]
                    trace = np.log(trace)
                    trace = pd.Series(trace).rolling(N_rolling).median().values
                    ax.plot(local_times_mins, trace, lw = 1, label = side)
                    da_eeg.loc[label,side,i,:] = trace

            ax.axvspan(0, baseline_pre_win_size_mins, label = 'Pre ICP Rise', alpha = 0.1, color = 'g')
            ax.axvspan(baseline_pre_win_size_mins, stop_per_mins, label = 'Per ICP Rise', alpha = 0.1, color = 'r')
            if label == 'prx':
                ylabel = label
            elif label in ['icp','abp','ppc','prx']:
                ylabel = f'{label}\n(mmHg)'
            elif label in ['delta','theta','alpha']:
                ylabel =  f'{label}\nln(µV)'
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, loc = 1)
            ax.set_xlim(0, win_duration_mins)
        ax.set_xlabel('Time (min)')
        fig.savefig(savefolder / 'erp_like_figs_sub_event' / f'{sub}_event{i+1}.png', bbox_inches = 'tight', dpi = 500)
        plt.close(fig)
    ds = xr.Dataset()
    ds['non_eeg'] = da_non_eeg
    ds['eeg'] = da_eeg
    return ds

def test_erp_like_fig_sub(sub):
    print(sub)
    ds = erp_like_fig_sub(sub, **erp_like_fig_sub_params)
    print(ds['non_eeg'])
    print(ds['eeg'])


erp_like_fig_sub_job = jobtools.Job(precomputedir, 'erp_like_fig_sub', erp_like_fig_sub_params, erp_like_fig_sub)
jobtools.register_job(erp_like_fig_sub_job)

##### RUN

def get_run_keys():
    df_nory = load_table_nory() # load event file
    # return list(df_nory['ID_pseudo'].unique())
    return ['P11']
    # return ['P67','P69']
    # return [(sub,) for sub in df_nory['ID_pseudo'].unique()]
    # return [(sub,) for sub in df_nory['ID_pseudo'].unique() if sub in ['FC13','WJ14','LA19','P50','P62','P64','P66','P68','']]
    
def compute_all():
    run_keys = get_run_keys()
    # jobtools.compute_job_list(spectrum_job, run_keys, force_recompute=True, engine='loop')
    # jobtools.compute_job_list(spectral_features_job, run_keys, force_recompute=True, engine='loop')
    # jobtools.compute_job_list(suppression_job, run_keys, force_recompute=True, engine='loop')
    # jobtools.compute_job_list(PRx_job, run_keys, force_recompute=True, engine='loop')
    # jobtools.compute_job_list(ICP_PPC_job, run_keys, force_recompute=True, engine='loop')
    # jobtools.compute_job_list(concat_res_icp_rises_job, [('all',)], force_recompute=True, engine='loop')
    jobtools.compute_job_list(erp_like_fig_sub_job, run_keys, force_recompute=True, engine='loop')

if __name__ == "__main__":
    # test_spectrum('P17')
    # test_spectral_features('P17')
    # test_suppression('P17')
    # test_PRx('P67') # P67, P69
    # test_ICP_PPC('P69') # P67, P69
    # test_concat_res_icp_rises()
    # test_erp_like_fig_sub('P3') # 

    compute_all()

    # by_subject_results()
    # by_subject_results2()
    # pooled_subject_results()

    


# FIG montée de PIC typique de Pre jusqu'à 2h après
# Une ligne par évènement par malade en supprimant ceux où la slope est pas bonne et les events où ratio suppression > 30 % 
# Une ligne pour la PIC puis pour la PPC puis pour le PRx puis double colonne puis pour les autres métriques en continuum temporel