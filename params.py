subs = ['Patient_2024_May_16__9_33_08_427295',
#  'Patient_2024_May_8__9_51_19_328502',
 'Patient_2024_May_25__16_48_44_716049',
 'Patient_2024_May_25__16_09_03_881877',
 'Patient_2024_Jun_24__12_33_38_160174',
 'PatientData_1723890119868125'
 ]

icp_chan_name = {'Patient_2024_May_16__9_33_08_427295':'P1',
#  'Patient_2024_May_8__9_51_19_328502':'P1',
 'Patient_2024_May_25__16_48_44_716049':'P',
 'Patient_2024_May_25__16_09_03_881877':'P1',
 'Patient_2024_Jun_24__12_33_38_160174':'ICP', # P1 pas assez de temps
 'PatientData_1723890119868125':'ICP', # ICP waveforme devient bizarre vers 2024-08-18T11:55:33.924284
 }

resp_abp_spectro_params = {
    'srate':5,
    'fmin':0.05,
    'fmax':0.5,
    'window_duration':100, # in secs
    'overlap_factor':2,
    'nfft_factor':2
}

slopes_distributions_fig_params = {
    'win_size_compute':30, # seconds
    'start_bin':-4,
    'stop_bin':1,
    'step_bin':0.1,
}

power_distributions_fig_params = {
    'win_size_compute':30,
    'n_bins':100
}

slope_power_artifacting_params = {
    'win_size_compute':30,
    'slope_threshold_artifacting':-1, # signal where slopes are higher than this threshold is considered artifacted
    'power_log_threshold_artifacting':25, # signal where total power higher than this threshold is considered artifacted
}

zscore_muscles_params = {
    'ftype':'butter',
    'order':4,
    'highcut_zscore':1,
    'estimators':'classic', # robust (median,mad) or classic (mean,std)
    'lowcut':30,
    'compute_estimators_on_good_signal':True # if True, zscore is computed based on estimators taken just on good quality masked signal, if False on all signal
}

muscle_artifacts_params = {
    'zscore_muscles_params':zscore_muscles_params,
    'win_load_duration_hours':5,
}


mask_artifacting_params = {
    'win_size_compute_secs':30, # seconds
    'slope_threshold_artifacting':-1, # signal where slopes are higher than this threshold is considered artifacted
    'power_log_threshold_artifacting':25, # signal where total power higher than this threshold is considered artifacted
    'threshold_n_chans_good':1 # proportion of channels (among ECoG and among Scalp) that have to not be artifacted at the same time to consider a good data quality
}

local_muscle_artifacts_params = {
    'win_load_duration_mins':10,
    'ftype':'butter',
    'order':4,
    'highcut_zscore':1,
    'estimators':'classic', # robust (median,mad) or classic (mean,std)
    'lowcut':30,
    'down_sample_results':True,
    'down_srate_nyquist_factor':4,
}

mask_muscle_params = {
    'local_muscle_artifacts_params':local_muscle_artifacts_params,
    'muscle_treshold':5,
    'consider_artifacts':True,
    'mask_artifacting_params':mask_artifacting_params
}

mask_6Hz_params = {
    'n_cycles_wavelet':20,
    'target_frequency':6,
    'log_threshold':11,
    'decimate_factor':10,
    'threshold_n_chans_bad':1
}

hep_fig_params = {
    'duration':20, # duration of analysis in mins
    'lowcut_eeg':1,
    'lowcut_bio':0.5,
    'highcut_eeg':None,
    'highcut_bio':None,
    'srate_resample':250,
    'size_before_ms':300,
    'size_after_ms':600,
}

depolarization_detection_params = {
    'lowcut_dc':0.001, 
    'highcut_dc':0.01,
    'order':4, 
    'threshold_depolarization_µV':2000
}

depolarization_cleaning_params = {
    'depolarization_detection_params':depolarization_detection_params,
    'threshold_max_amplitude_µV':20000,
    'max_time_lag_secs':900,
    'min_time_lag_1st_2nd':60
}



