global_key = 'all'

load_one_eeg_params = {
    'hours_load':0.1, # duration in hours of eeg stream windows that are loaded
    'apply_gain':True
}

detect_resp_params = {}

detect_ecg_params = {}

rsa_params = {
    'detect_resp_params':detect_resp_params,
    'detect_ecg_params':detect_ecg_params
}

detect_abp_params = {
    'lowcut':0.08,
    'highcut':10,
    'order':4,
    'ftype':'butter',
}

detect_icp_params = {
    'lowcut':0.1,
    'highcut':10,
    'order':4,
    'ftype':'butter'
}

prx_params = {
    'wsize_mean_secs':10, 
    'wsize_corr_mins':5, 
    'overlap_corr_prop':0.8,
}

psi_params = {}

crps_params = {
    'detect_resp_params':detect_resp_params,
    'detect_ecg_params':detect_ecg_params
}

