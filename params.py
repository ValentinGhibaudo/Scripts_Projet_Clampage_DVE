subs = ['Patient_2024_May_16__9_33_08_427295',
 'Patient_2024_May_13__9_36_06_745131',
 'Patient_2024_May_8__9_51_19_328502',
 'Patient_2024_May_25__16_09_03_881877',
 'Patient_2024_Sep_10__11_32_35_613533',
 'Patient_2024_May_25__16_48_44_716049',
 'Patient_2024_Jun_24__12_33_38_160174',
 'PatientData_1723890119868125'
 ]

detect_icp_params = {
    'lowcut':0.1,
    'highcut':10,
    'order':4,
    'ftype':'butter',
    'exclude_sweep_ms':200, 
}

psi_params = {
}

heart_resp_spectral_peaks_params = {
    'spectrogram_win_size_secs':60,
    'resp_fband':(0.18,0.6),
    'heart_fband':(0.8,2.5),
    'rolling_N_time_spectrogram':5,
    'savefig':True
 } 

ratio_P1P2_params = {
    'N_pulse_sliding_window_fig':20,
}

metrics_params = {
    'detect_icp_params':detect_icp_params,
    'psi_params':psi_params,
    'ratio_P1P2_params':ratio_P1P2_params,
    'heart_resp_spectral_peaks_params':heart_resp_spectral_peaks_params,
    'analyzing_window_start_hours_after_clamp':[0,12,24],
    'analyzing_window_duration_hours':2,
}

concat_metrics_params = {
    'metrics_params':metrics_params
}