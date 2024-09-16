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

detect_icp_params = {
    'lowcut':0.1,
    'highcut':10,
    'order':4,
    'ftype':'butter',
    'exclude_sweep_ms':200, 
    'icp_chan_name':icp_chan_name
}

psi_params = {
    'icp_chan_name':icp_chan_name,
}

heart_resp_spectral_peaks_params = {
    'icp_chan_name':icp_chan_name,
    'spectrogram_win_size_secs':60,
    'resp_fband':(0.18,0.6),
    'heart_fband':(0.8,2.5),
    'rolling_N_time_spectrogram':5,
    'savefig':True
 } 

ratio_P1P2_params = {
    'icp_chan_name':icp_chan_name,
}