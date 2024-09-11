import numpy as np
import xarray as xr
import pandas as pd
from pycns import CnsStream, CnsReader
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import jobtools
from tools import *
from configuration import *
from params_icp_to_resp import *
from icp_rises import load_table_nory, format_window_dates
from multi_projects_jobs import *
from stats_tools import stats_quantitative, auto_stats, auto_stats_summary, save_auto_stats_summary

def interpolate_samples(data, data_times, time_vector, kind = 'linear'):
    f = scipy.interpolate.interp1d(data_times, data, fill_value="extrapolate", kind = kind)
    xnew = time_vector
    ynew = f(xnew)
    return ynew

def plot_icp_resp(sub, **p):
    start_record, stop_record = get_patient_dates(sub)
    # start = start_record + np.timedelta64(1, 'D')
    start = start_record + np.timedelta64(p['n_hours_post_start_record'], 'h')
    stop = start + np.timedelta64(p['duration_analysis_hours'], 'h')
    
    raw_folder = data_path / sub
    cns_reader = CnsReader(raw_folder)

    stream_names = ['ICP','ABP','CO2','ECG_II']
    srate = 200.
    ds = cns_reader.export_to_xarray(stream_names, start=start, stop=stop, resample=True, sample_rate=srate)
    
    dates = ds['times'].values
    t = np.arange(0, dates.size / srate, 1 / srate)

    raw_icp = ds['ICP'].values
    raw_abp = ds['ABP'].values
    raw_co2 = ds['CO2'].values
    raw_ecg = ds['ECG_II'].values
    
    fig, axs = plt.subplots(nrows = 3, ncols = 4, figsize = p['figsize'], constrained_layout = True)
    fig.suptitle(sub, fontsize = 20, y = 1.05)
    
    # DETECT ICP PEAKS AND PLOT AVERAGE WAVEFORM
    icp_detection = compute_icp(raw_icp, srate)
    icp_peak_inds = icp_detection['peak_ind'].values
    
    half_win_size = p['half_win_size'] # sec
    half_win_size_inds = int(half_win_size * srate)

    mask_inds = ((icp_peak_inds - half_win_size_inds) > 0) & ((icp_peak_inds + half_win_size_inds) < raw_icp.size)
    icp_peak_inds_loop = icp_peak_inds[mask_inds] 

    icp_cycles = None

    for i, icp_peak_ind in enumerate(icp_peak_inds_loop):
        start_win = icp_peak_ind - half_win_size_inds
        stop_win = icp_peak_ind + half_win_size_inds
        icp_sel = raw_icp[start_win:stop_win]

        if icp_cycles is None:
            icp_cycles = np.zeros((icp_peak_inds_loop.size, icp_sel.size))

        icp_cycles[i,:] = icp_sel
        
    icp_cycles_centered = icp_cycles.T - np.mean(icp_cycles, axis = 1)
    icp_cycles_centered = icp_cycles_centered.T

    m, s = np.mean(icp_cycles_centered, axis = 0), np.std(icp_cycles_centered, axis = 0)
    x = np.arange(- half_win_size, half_win_size, 1 / srate)

    ax = axs[0,0]
    ax.plot(x, m, color = 'r', lw = 2, label = 'mean')
    ax.fill_between(x, m-s, m+s, color = 'k', alpha = 0.2, label = '+/- sd')
    ax.legend()
    ax.set_title('Average ICP waveform')
    ax.set_xlabel('Time (sec)')
    
    # DETECT ABP PEAKS AND PLOT AVERAGE WAVEFORM
    abp_detection = compute_abp(raw_abp, srate)
    abp_peak_inds = abp_detection['peak_ind'].values
    
    half_win_size = p['half_win_size'] # sec
    half_win_size_inds = int(half_win_size * srate)

    mask_inds = ((abp_peak_inds - half_win_size_inds) > 0) & ((abp_peak_inds + half_win_size_inds) < raw_abp.size)
    abp_peak_inds_loop = abp_peak_inds[mask_inds] 

    abp_cycles = None

    for i, abp_peak_ind in enumerate(abp_peak_inds_loop):
        start_win = abp_peak_ind - half_win_size_inds
        stop_win = abp_peak_ind + half_win_size_inds
        abp_sel = raw_abp[start_win:stop_win]

        if abp_cycles is None:
            abp_cycles = np.zeros((abp_peak_inds_loop.size, abp_sel.size))

        abp_cycles[i,:] = abp_sel
        
    abp_cycles_centered = abp_cycles.T - np.mean(abp_cycles, axis = 1)
    abp_cycles_centered = abp_cycles_centered.T

    m, s = np.mean(abp_cycles_centered, axis = 0), np.std(abp_cycles_centered, axis = 0)
    x = np.arange(- half_win_size, half_win_size, 1 / srate)

    ax = axs[1,0]
    ax.plot(x, m, color = 'r', lw = 2, label = 'mean')
    ax.fill_between(x, m-s, m+s, color = 'k', alpha = 0.2, label = '+/- sd')
    ax.legend()
    ax.set_title('Average ABP waveform')
    ax.set_xlabel('Time (sec)')

    # DETECT RESP 
    _, resp_cycles = physio.compute_respiration(raw_co2, srate, parameter_preset = 'human_co2')
    
    # AND PLOT CYCLICAL DEFORMATION OF ICP BY RESP
    cycle_times = resp_cycles[['inspi_time','expi_time','next_inspi_time']].values
    segment_ratios = 0.4
    nbins = p['nbins']
    icp_by_resp_cycle = physio.deform_traces_to_cycle_template(data = raw_icp, 
                                                               times = t,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = nbins
                                                              )
    phase_resp = np.linspace(0, 1, nbins)

    icp_by_resp_cycle_centered = (icp_by_resp_cycle.T - np.mean(icp_by_resp_cycle, axis = 1)).T 
    icp_by_resp_cycle_centered = icp_by_resp_cycle_centered + np.abs(np.min(icp_by_resp_cycle_centered))

    m = np.mean(icp_by_resp_cycle_centered, axis = 0)

    relative_variation = (np.ptp(m) / np.mean(m)) * 100

    ax = axs[0,1]
    ax.plot(phase_resp, m)
    ax.axvline(segment_ratios, color = 'r', label = 'Inspi>Expi')
    ax.set_title(f'ICP deformed by resp\nvar : {round(relative_variation, 2)}%')
    ax.legend()
    
    # AND PLOT CYCLICAL DEFORMATION OF ABP BY RESP
    abp_by_resp_cycle = physio.deform_traces_to_cycle_template(data = raw_abp, 
                                                               times = t,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = nbins
                                                              )
    phase_resp = np.linspace(0, 1, nbins)

    abp_by_resp_cycle_centered = (abp_by_resp_cycle.T - np.mean(abp_by_resp_cycle, axis = 1)).T 
    abp_by_resp_cycle_centered = abp_by_resp_cycle_centered + np.abs(np.min(abp_by_resp_cycle_centered))

    m = np.mean(abp_by_resp_cycle_centered, axis = 0)

    relative_variation = (np.ptp(m) / np.mean(m)) * 100

    ax = axs[1,1]
    ax.plot(phase_resp, m)
    ax.axvline(segment_ratios, color = 'r', label = 'Inspi>Expi')
    ax.set_title(f'ABP deformed by resp\nvar : {round(relative_variation, 2)}%')
    ax.legend()
    
    # COMPUTE AND PLOT INTERPOLATED ICP RISE VECTOR
    ICP_rises_amps_interpolated = interpolate_samples(icp_detection['rise_amplitude'], 
                                                       icp_detection['peak_time'],
                                                      t)

    icp_rise_amplitude_by_resp = physio.deform_traces_to_cycle_template(data = ICP_rises_amps_interpolated, 
                                                               times = t,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = nbins
                                                              )
    
    ax = axs[0,2]
    m = np.mean(icp_rise_amplitude_by_resp, axis = 0)
    relative_variation = (np.ptp(m) / np.mean(m)) * 100

    ax.plot(phase_resp, m)
    ax.axvline(segment_ratios, color = 'r', label = 'Inspi>Expi')
    ax.legend()
    ax.set_title(f'ICP Rise Amplitude deformed by resp\nvar : {round(relative_variation, 2)}%')
    
    # COMPUTE AND PLOT INTERPOLATED ABP RISE VECTOR
    abp_rises_amps_interpolated = interpolate_samples(abp_detection['rise_amplitude'], 
                                                       abp_detection['peak_time'],
                                                      t)

    abp_rise_amplitude_by_resp = physio.deform_traces_to_cycle_template(data = abp_rises_amps_interpolated, 
                                                               times = t,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = nbins
                                                              )
    
    ax = axs[1,2]
    m = np.mean(abp_rise_amplitude_by_resp, axis = 0)
    relative_variation = (np.ptp(m) / np.mean(m)) * 100

    ax.plot(phase_resp, m)
    ax.axvline(segment_ratios, color = 'r', label = 'Inspi>Expi')
    ax.legend()
    ax.set_title(f'ABP Rise Amplitude deformed by resp\nvar : {round(relative_variation, 2)}%')
    
    # PLOT HEART RATE
    ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_preset='human_ecg')
    instantaneous_heart_rate = physio.compute_instantaneous_rate(
        ecg_peaks,
        t,
        limits=[30, 150],
        units='bpm',
        interpolation_kind='linear',
    )
    

    heart_rate_by_cycle = physio.deform_traces_to_cycle_template(data = instantaneous_heart_rate, 
                                                                 times = t,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = nbins
                                                              )
    m = np.mean(heart_rate_by_cycle, axis = 0)

    relative_variation = (np.ptp(m) / np.mean(m)) * 100

    ax = axs[2,1]
    ax.plot(phase_resp, m)
    ax.axvline(segment_ratios, color = 'r', label = 'Inspi>Expi')
    ax.set_title(f'Heart rate deformed by resp\nvar : {round(relative_variation, 2)}%')
    ax.legend()
    
    
    # COMPUTE AND PLOT PSDs
    f, Pxx_ICP = spectre(raw_icp, srate, lowest_freq = 0.1)
    f, Pxx_CO2 = spectre(raw_co2, srate, lowest_freq = 0.1)
    f, Pxx_ABP = spectre(raw_abp, srate, lowest_freq = 0.1)
    f, Pxx_IHR = spectre(instantaneous_heart_rate, srate, lowest_freq = 0.1)
    
    ax = axs[0,3]
    ax.plot(f, np.sqrt(Pxx_ICP), color = 'k', label = 'ICP')
    ax.set_xlim(0.05, 1.5)
    ax.set_title('ICP spectrum')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency (Hz)')
    ax.legend(loc = 2)
    ax2 = ax.twinx()
    ax2.plot(f, np.sqrt(Pxx_CO2), color = 'r', label = 'CO2')
    ax2.set_xlim(0.05, 1.5)
    ax2.legend(loc = 1)
    
    ax = axs[1,3]
    ax.plot(f, np.sqrt(Pxx_ABP), color = 'k', label = 'ABP')
    ax.set_xlim(0.05, 1.5)
    ax.set_title('ABP spectrum')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency (Hz)')
    ax.legend(loc = 2)
    ax2 = ax.twinx()
    ax2.plot(f, np.sqrt(Pxx_CO2), color = 'r', label = 'CO2')
    ax2.set_xlim(0.05, 1.5)
    ax2.legend(loc = 1)
    
    ax = axs[2,3]
    ax.plot(f, np.sqrt(Pxx_IHR), color = 'k', label = 'IHR')
    ax.set_xlim(0.05, 1.5)
    ax.set_title('Heart rate spectrum')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency (Hz)')
    ax.legend(loc = 2)
    ax2 = ax.twinx()
    ax2.plot(f, np.sqrt(Pxx_CO2), color = 'r', label = 'CO2')
    ax2.set_xlim(0.05, 1.5)
    ax2.legend(loc = 1)
    
    fig.savefig(base_folder / 'figures' / 'icp_co2' / f'{sub}.png', dpi = 500, bbox_inches = 'tight')
    plt.close('all')
    
    return xr.Dataset()

def test_plot_icp_resp(sub):
    print(sub)
    ds = plot_icp_resp(sub, **plot_icp_resp_params)
    
plot_icp_resp_job = jobtools.Job(precomputedir, 'plot_icp_resp', plot_icp_resp_params, plot_icp_resp)
jobtools.register_job(plot_icp_resp_job)

##### FIG BRAIN COMLPIANCE RESP HEART ICP

def compute_heart_resp_spectral_ratio_in_icp(abp, resp, icp, srate, wsize_secs = 50):
    
    nperseg = int(wsize_secs * srate)
    nfft = int(nperseg * 2)
    
    # Compute instantaneous resp rate with CO2
    search_freq_band = (0.05,0.7)    
    freqs, times_spectrum_s, Sxx_co2 = scipy.signal.spectrogram(resp, fs = srate, nperseg =  nperseg, nfft = nfft)
    mask_f = (freqs > search_freq_band[0]) & (freqs < search_freq_band[1])
    freqs = freqs[mask_f]
    Sxx_co2 = Sxx_co2[mask_f,:]
    inds_where_resp_is = np.argmax(Sxx_co2, axis = 0)
    instantaneous_resp_rate = np.apply_along_axis(lambda i:freqs[i], axis = 0, arr = inds_where_resp_is)
    
    # Compute instantaneous heart rate with ABP
    search_freq_band = (0.7,2.5)
    freqs, times_spectrum_s, Sxx_abp = scipy.signal.spectrogram(abp, fs = srate, nperseg =  nperseg, nfft = nfft)
    mask_f = (freqs > search_freq_band[0]) & (freqs < search_freq_band[1])
    freqs = freqs[mask_f]
    Sxx_abp = Sxx_abp[mask_f,:]
    inds_where_heart_is = np.argmax(Sxx_abp, axis = 0)
    instantaneous_heart_rate = np.apply_along_axis(lambda i:freqs[i], axis = 0, arr = inds_where_heart_is)
    
    # Compute ICP spectrogram
    search_freq_band = (0.05,2.5)
    freqs_icp, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(icp, fs = srate, nperseg =  nperseg, nfft = nfft)
    Sxx_icp = np.sqrt(Sxx_icp) # power to amplitude
    mask_f = (freqs_icp > search_freq_band[0]) & (freqs_icp < search_freq_band[1])
    freqs_icp = freqs_icp[mask_f]
    Sxx_icp = Sxx_icp[mask_f,:]
    
    heart_amplitude = np.zeros(instantaneous_resp_rate.size)
    resp_amplitude = np.zeros(instantaneous_resp_rate.size)
    ratio_heart_resp = np.zeros(instantaneous_resp_rate.size)
    for ti in range(times_spectrum_s.size):
        local_heart_rate = instantaneous_heart_rate[ti]
        local_resp_rate = instantaneous_resp_rate[ti]
        amplitude_at_heart_rate_in_icp = Sxx_icp[freqs_icp == local_heart_rate,ti][0]
        amplitude_at_resp_rate_in_icp = Sxx_icp[freqs_icp == local_resp_rate,ti][0]
        ratio = amplitude_at_heart_rate_in_icp / amplitude_at_resp_rate_in_icp
        heart_amplitude[ti] = amplitude_at_heart_rate_in_icp
        resp_amplitude[ti] = amplitude_at_resp_rate_in_icp
        ratio_heart_resp[ti] = ratio
    res = {'times_spectrum_s':times_spectrum_s,'heart_in_icp_spectrum':heart_amplitude, 'resp_in_icp_spectrum':resp_amplitude,'ratio_heart_resp_in_icp_spectrum':ratio_heart_resp}
    return res

def compute_heart_resp_spectral_ratio_in_icp2(icp, srate, wsize_secs = 50):
    
    nperseg = int(wsize_secs * srate)
    nfft = int(nperseg * 2)

    # Compute spectro ICP
    freqs, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(icp, fs = srate, nperseg =  nperseg, nfft = nfft)

    # Compute instantaneous resp rate with ICP
    search_freq_band = (0.05,0.6)    
    mask_f_co2 = (freqs > search_freq_band[0]) & (freqs < search_freq_band[1])
    freqs_co2 = freqs[mask_f_co2]
    Sxx_co2 = Sxx_icp[mask_f_co2,:]
    inds_where_resp_is = np.argmax(Sxx_co2, axis = 0)
    instantaneous_resp_rate = np.apply_along_axis(lambda i:freqs_co2[i], axis = 0, arr = inds_where_resp_is)
    
    # Compute instantaneous heart rate with ICP
    search_freq_band = (0.7,2.5)
    mask_f_abp = (freqs > search_freq_band[0]) & (freqs < search_freq_band[1])
    freqs_abp = freqs[mask_f_abp]
    Sxx_abp = Sxx_icp[mask_f_abp,:]
    inds_where_heart_is = np.argmax(Sxx_abp, axis = 0)
    instantaneous_heart_rate = np.apply_along_axis(lambda i:freqs_abp[i], axis = 0, arr = inds_where_heart_is)
    
    # Compute ICP spectrogram
    search_freq_band = (0.05,2.5)
    Sxx_icp = np.sqrt(Sxx_icp) # power to amplitude
    mask_f_icp = (freqs > search_freq_band[0]) & (freqs < search_freq_band[1])
    freqs_icp = freqs[mask_f_icp]
    Sxx_icp = Sxx_icp[mask_f_icp,:]
    
    heart_amplitude = np.zeros(instantaneous_resp_rate.size)
    resp_amplitude = np.zeros(instantaneous_resp_rate.size)
    ratio_heart_resp = np.zeros(instantaneous_resp_rate.size)
    for ti in range(times_spectrum_s.size):
        local_heart_rate = instantaneous_heart_rate[ti]
        local_resp_rate = instantaneous_resp_rate[ti]
        amplitude_at_heart_rate_in_icp = Sxx_icp[freqs_icp == local_heart_rate,ti][0]
        amplitude_at_resp_rate_in_icp = Sxx_icp[freqs_icp == local_resp_rate,ti][0]
        ratio = amplitude_at_heart_rate_in_icp / amplitude_at_resp_rate_in_icp
        heart_amplitude[ti] = amplitude_at_heart_rate_in_icp
        resp_amplitude[ti] = amplitude_at_resp_rate_in_icp
        ratio_heart_resp[ti] = ratio
    res = {'times_spectrum_s':times_spectrum_s,'heart_in_icp_spectrum':heart_amplitude, 'resp_in_icp_spectrum':resp_amplitude,'ratio_heart_resp_in_icp_spectrum':ratio_heart_resp}
    return res

def compute_heart_resp_spectral_ratio_in_icp3(icp, srate, wsize_secs = 50, resp_fband = (0.12,0.6), heart_fband = (0.8,2.5), rolling_N_time = 5):
    
    nperseg = int(wsize_secs * srate)
    nfft = int(nperseg)

    # Compute spectro ICP
    freqs, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(icp, fs = srate, nperseg =  nperseg, nfft = nfft)
    Sxx_icp = np.sqrt(Sxx_icp)
    da = xr.DataArray(data = Sxx_icp, dims = ['freq','time'], coords = {'freq':freqs, 'time':times_spectrum_s})
    resp_amplitude = da.loc[resp_fband[0]:resp_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    heart_amplitude = da.loc[heart_fband[0]:heart_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    ratio_heart_resp = heart_amplitude / resp_amplitude
    res = {'times_spectrum_s':times_spectrum_s,'heart_in_icp_spectrum':heart_amplitude.values, 'resp_in_icp_spectrum':resp_amplitude.values,'ratio_heart_resp_in_icp_spectrum':ratio_heart_resp.values}
    return res

def compute_relative_variation_of_icp_pulse_modulation_by_resp(co2, icp, srate):
    times = np.arange(0, co2.size / srate, 1 / srate)
    _,resp_cycles = physio.compute_respiration(co2, srate, parameter_preset = 'human_co2')
    icp_detection = compute_icp(icp, srate)
    ICP_rises_amps_interpolated = interpolate_samples(icp_detection['rise_amplitude'], 
                                                   icp_detection['peak_time'],
                                                      times)
    cycle_times = resp_cycles[['inspi_time','expi_time','next_inspi_time']].values
    segment_ratios = 0.4
    nbins = 100
    icp_rise_amplitude_by_resp = physio.deform_traces_to_cycle_template(data = ICP_rises_amps_interpolated, 
                                                               times = times,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = nbins
                                                              )
    m = np.mean(icp_rise_amplitude_by_resp, axis = 0)
    relative_variation = np.abs((np.ptp(m) / np.mean(m))) * 100
    return relative_variation

def compute_relative_variation_of_abp_pulse_modulation_by_resp(co2, abp, srate):
    times = np.arange(0, co2.size / srate, 1 / srate)
    _,resp_cycles = physio.compute_respiration(co2, srate, parameter_preset = 'human_co2')
    abp_detection = compute_abp(abp, srate)
    ABP_rises_amps_interpolated = interpolate_samples(abp_detection['rise_amplitude'], 
                                                   abp_detection['peak_time'],
                                                      times)
    cycle_times = resp_cycles[['inspi_time','expi_time','next_inspi_time']].values
    segment_ratios = 0.4
    nbins = 100
    abp_rise_amplitude_by_resp = physio.deform_traces_to_cycle_template(data = ABP_rises_amps_interpolated, 
                                                               times = times,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = nbins
                                                              )
    m = np.mean(abp_rise_amplitude_by_resp, axis = 0)
    relative_variation = np.abs((np.ptp(m) / np.mean(m))) * 100
    return relative_variation

def plot_compliance_metrics_correlations(sub, **p):

    raw_folder = data_path / sub
    cns_reader = CnsReader(raw_folder)
    
    prx, prx_pval, prx_dates = compute_prx(cns_reader)
    
    srate = p['load_srate']
    wsize_mins = p['wsize_load_mins']

    stream_names = ['ICP','ABP','CO2']

    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    df = pd.DataFrame()
    df['dates'] = ds['times'].values
    df['times'] = np.arange(0, df['dates'].size / srate, 1 / srate)
    df['ICP'] = ds['ICP'].values
    df['ABP'] = ds['ABP'].values
    df['CO2'] = ds['CO2'].values
    df = df.dropna()
    
    start_dates = np.arange(df['dates'].values[0], df['dates'].values[-1], np.timedelta64(wsize_mins, 'm'))
    start_dates = start_dates[:-1]
    
    rows = []
    # loop = tqdm(range(start_dates.size))
    loop = range(start_dates.size)
    for i in loop:
        start_date = start_dates[i]
        stop_date = start_date + np.timedelta64(wsize_mins, 'm')
        mask_df = (df['dates'] >= start_date) & (df['dates'] < stop_date)
        df_sel = df[mask_df]
    
        ratio_heart_resp, _ = compute_heart_resp_spectral_ratio_in_icp(df_sel['ABP'].values, 
                                                                       df_sel['CO2'].values, 
                                                                       df_sel['ICP'].values, 
                                                                       srate, 
                                                                       wsize_secs = p['wsize_secs_spectro']
                                                                      )
        ratio_heart_resp = np.median(ratio_heart_resp)
        
        relative_variation_icp_pulse_by_resp = compute_relative_variation_of_icp_pulse_modulation_by_resp(df_sel['CO2'].values, 
                                                                                                          df_sel['ICP'].values, 
                                                                                                          srate
                                                                                                         )
        relative_variation_icp_pulse_by_resp = np.median(relative_variation_icp_pulse_by_resp)
        
        relative_variation_abp_pulse_by_resp = compute_relative_variation_of_abp_pulse_modulation_by_resp(df_sel['CO2'].values, 
                                                                                                          df_sel['ABP'].values, 
                                                                                                          srate
                                                                                                         )
        relative_variation_abp_pulse_by_resp = np.median(relative_variation_abp_pulse_by_resp)
        
        mask_local_dates_prx = (prx_dates >= start_date) & (prx_dates < stop_date)
        local_prx = prx[mask_local_dates_prx]
        local_prx = np.median(local_prx)
        
        rows.append([start_date, ratio_heart_resp, relative_variation_icp_pulse_by_resp , relative_variation_abp_pulse_by_resp, local_prx])
    
    df_res = pd.DataFrame(rows, columns = ['date','ratio_heart_resp_in_icp','relative_variation_icp_pulse_by_resp', 'relative_variation_abp_pulse_by_resp', 'median_prx'])
    
    res_corr_ratio_icp_pulse_resp = scipy.stats.pearsonr(df_res['ratio_heart_resp_in_icp'].values, df_res['relative_variation_icp_pulse_by_resp'].values)
    res_corr_ratio_icp_pulse_resp = res_corr_ratio_icp_pulse_resp.statistic
    res_corr_ratio_icp_pulse_resp = round(res_corr_ratio_icp_pulse_resp, 3)
    
    res_corr_abp_icp_pulse_resp = scipy.stats.pearsonr(df_res['relative_variation_icp_pulse_by_resp'].values, df_res['relative_variation_abp_pulse_by_resp'].values)
    res_corr_abp_icp_pulse_resp = res_corr_abp_icp_pulse_resp.statistic
    res_corr_abp_icp_pulse_resp = round(res_corr_abp_icp_pulse_resp, 3)
    
    fig, axs = plt.subplots(nrows = 2, figsize = (9, 8), constrained_layout = True)
    fontsize_legend = 8
    ax = axs[0]
    ax.plot(df_res['date'].values, df_res['ratio_heart_resp_in_icp'], label = 'Ratio Heart/Resp in ICP spectra')
    ax.plot(df_res['date'].values, df_res['relative_variation_icp_pulse_by_resp'], label = 'Prct Variation ICP Pulse Modulation by Resp')
    ax.set_title(f'Pearson R : {res_corr_ratio_icp_pulse_resp}')
    ax.set_ylabel('Ratio / % Variation')
    ax.set_xlabel('Datetime')
    ax.legend(fontsize = fontsize_legend, loc = 2)
    ax2 = ax.twinx()
    ax2.plot(df_res['date'].values, df_res['median_prx'].values, color = 'g', label = 'PRx')
    ax2.legend(fontsize = fontsize_legend, loc = 1)
    ax2.set_ylabel('PRx')

    
    ax = axs[1]
    ax.plot(df_res['date'].values, df_res['relative_variation_abp_pulse_by_resp'], label = 'Prct Variation ABP Pulse Modulation by Resp')
    ax.plot(df_res['date'].values, df_res['relative_variation_icp_pulse_by_resp'], label = 'Prct Variation ICP Pulse Modulation by Resp')
    ax.set_ylabel('% Variation')
    ax.set_title(f'Pulse Modulation by Resp of ICP vs ABP (corr : {res_corr_abp_icp_pulse_resp})') 
    ax.set_xlabel('Datetime')
    ax.legend(fontsize = fontsize_legend, loc = 2)
    ax2 = ax.twinx()
    ax2.plot(df_res['date'].values, df_res['median_prx'].values, color = 'g', label = 'PRx')
    ax2.legend(fontsize = fontsize_legend, loc = 1)
    ax2.set_ylabel('PRx')
    save_folder = base_folder / 'figures' / 'brain_compliance_icp_heart_resp' / f'{srate}_Hz'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    fig.savefig(save_folder / f'{sub}.png', dpi = 500, bbox_inches = 'tight')
    plt.close('all')

    row = [sub, 
           res_corr_ratio_icp_pulse_resp, 
           df_res['relative_variation_abp_pulse_by_resp'].min(), 
           df_res['relative_variation_abp_pulse_by_resp'].max(),
           df_res['median_prx'].min(),
           df_res['median_prx'].max()
           ]

    df_stats_return = pd.Series(row, index = ['patient','corr_RespPAC_RespSAR','Resp_dPP_min','Resp_dPP_max','PRx_min','PRx_max'])
    df_stats_return = df_stats_return.to_frame().T
    
    return xr.Dataset(df_stats_return)

def test_plot_compliance_metrics_correlations():
    sub = 'MF12'
    ds = plot_compliance_metrics_correlations(sub, **plot_compliance_metrics_correlations_params)
    print(ds.to_dataframe())
    
plot_compliance_metrics_correlations_job = jobtools.Job(precomputedir, 'plot_compliance_metrics_correlations', plot_compliance_metrics_correlations_params, plot_compliance_metrics_correlations)
jobtools.register_job(plot_compliance_metrics_correlations_job)

def format_stats(data, n_dec=2):
    m = np.mean(data)
    m = round(m, n_dec)
    q25 = np.quantile(data, 0.25)
    q25 = round(q25, n_dec)
    q75 = np.quantile(data, 0.75)
    q75 = round(q75, n_dec)
    return f'{m} (IQR: [{q25}, {q75}])'

def save_abstract_neurotrauma_results(n_dec = 2):
    concat = []
    for sub in ['MF12','P2','P4','P6','P12','P37','P39','P43','SP2']:
        ds = plot_compliance_metrics_correlations_job.get(sub)
        # print(ds)
        concat.append(ds.to_dataframe())
    
    df_stats = pd.concat(concat)
    N = df_stats.shape[0]
    corr_RespPAC_RespSAR = format_stats(df_stats['corr_RespPAC_RespSAR'].values, n_dec)
    resp_dPP_min = format_stats(df_stats['Resp_dPP_min'].values, n_dec)
    resp_dPP_max = format_stats(df_stats['Resp_dPP_max'].values, n_dec)
    PRx_min = format_stats(df_stats['PRx_min'].values, n_dec)
    PRx_max = format_stats(df_stats['PRx_max'].values, n_dec)

    row = [N,
           corr_RespPAC_RespSAR,
           resp_dPP_min,
           resp_dPP_max,
           PRx_min,
           PRx_max
    ]

    columns = ['N',
           'corr_RespPAC_RespSAR',
           'resp_dPP_min',
           'resp_dPP_max',
           'PRx_min',
           'PRx_max'
    ]
    
    df_save = pd.Series(row, index = columns).to_frame().T
    df_save.to_excel(base_folder / 'documents_valentin' / 'Congrès' / 'neurotrauma_2024' / 'stats_for_abstract.xlsx')


def figure_sfar(sub):
    p = {
        'n_hours_post_start_record':1,
        'duration_analysis_mins':30,
        'figsize':(13,10),
        'half_win_size':0.8, # sec
        'nbins':100,
        'wsize_secs_spectro':60
    }

    start_record, stop_record = get_patient_dates(sub)
    start = start_record + np.timedelta64(p['n_hours_post_start_record'], 'h')
    stop = start + np.timedelta64(p['duration_analysis_mins'], 'm')

    print(sub, start, stop)
    
    raw_folder = data_path / sub
    cns_reader = CnsReader(raw_folder)

    stream_names = ['ICP','ABP','CO2','ICP_Mean']
    srate = 200.
    ds = cns_reader.export_to_xarray(stream_names, start=start, stop=stop, resample=True, sample_rate=srate)
    dates_ds = ds['times'].values
    t_all = np.arange(ds['ICP'].size) / srate
    prx_all, prx_pval, dates_prx = compute_prx(cns_reader)
    ratio_heart_resp, times_spectrum_s = compute_heart_resp_spectral_ratio_in_icp(ds['ABP'].values, 
                                                                       ds['CO2'].values, 
                                                                       ds['ICP'].values, 
                                                                       srate, 
                                                                       wsize_secs = p['wsize_secs_spectro']
                                                                      )
    dates_ratio_heart_resp = dates_ds[np.searchsorted(t_all, times_spectrum_s)]

    fig, axs = plt.subplots(nrows = 3, figsize = (8,8), constrained_layout = True)

    ax = axs[0]
    duration = 10
    raw_co2 = ds['CO2'][:int(duration*srate)].values
    t = np.arange(raw_co2.size) / srate
    alpha = 0.8
    ax.plot(t, raw_co2, label = 'CO2', alpha = alpha)
    ax.plot(t, ds['ABP'][:int(duration*srate)].values, label = 'ABP', alpha = alpha)
    ax2 = ax.twinx()
    ax2.plot(t, ds['ICP'][:int(duration*srate)].values, label = 'ICP', alpha = alpha, color = 'g')
    ax2.set_ylabel('Amplitude (ICP : mmHg)')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (ABP : mmHg)')
    ax.set_title('Temporal domain')
    ax.legend(loc = 1)
    ax2.legend(loc = 2)

    ax = axs[1]
    wsize_secs = 60
    wsize = int(wsize_secs * srate)
    f, Pxx_icp = scipy.signal.welch(ds['ICP'].values, srate, nperseg = wsize, nfft = wsize * 2)
    mask_f_resp = (f > 0.15) & (f < 0.5)
    mask_f_heart = (f > 0.65) & (f < 2)
    resp_f = f[mask_f_resp][np.argmax(Pxx_icp[mask_f_resp])]
    heart_f = f[mask_f_heart][np.argmax(Pxx_icp[mask_f_heart])]
    f_mask = (f > 0.05) & (f < 1.5)
    ax.plot(f[f_mask], np.abs(Pxx_icp[f_mask]), color = 'k', label = 'ICP Spectra')
    ax.scatter(f[f == resp_f], Pxx_icp[f == resp_f], color = 'tab:blue', label = 'Respi peak')
    ax.scatter(f[f == heart_f], Pxx_icp[f == heart_f], color = 'r', label = 'Heart peak')
    ax.set_ylabel('Amplitude (AU)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Frequency domain')
    ax.legend()

    ax = axs[2]
    mask_local_dates_prx = (dates_prx >= start) & (dates_prx < stop)
    local_dates_prx = dates_prx[(dates_prx >= start) & (dates_prx < stop)]
    local_prx = prx_all[mask_local_dates_prx]
    ax.plot(local_dates_prx , local_prx, label = 'PRx')
    ax.plot(dates_ratio_heart_resp, ratio_heart_resp, label ='R-SAR')
    ax2 = ax.twinx()
    # ax2.plot(dates_ds, iirfilt(ds['ICP'].values, srate, highcut = 0.5), color = 'g', label = 'ICP')
    ax2.plot(dates_ds, ds['ICP_Mean'].values, color = 'g', label = 'ICP')
    ax2.legend()
    ax.legend(loc = 1)
    ax.set_ylabel('Ratio (R-SAR)\nPearson R (PRx)')
    ax2.legend(loc = 2)
    ax2.set_ylabel('mmHg (ICP)')
    ax.set_title('Spectral Ratio (R-SAR) - Pressure Reactivity Index (PRx) - Intracranial Pressure (ICP)')
    ax.set_xlabel('Date')
    
    fig.savefig(base_folder / 'figures' / 'icp_co2' / f'pour_baptiste_sfar_{sub}.png', dpi = 500, bbox_inches = 'tight')
    
    plt.close('all')

# ERP LIKE FIGS
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
        # to_add = np.nanmedian(trace_resampled)
        to_add = trace_resampled[-1]
        trace_resampled = np.append(trace_resampled, np.full(size_points_to_add, to_add))
    return trace_resampled

def compute_relative_variation_of_icp_pulse_modulation_by_resp2(sub, times_for_interpolation):
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    icp_detection = detect_icp_job.get(sub).to_dataframe()
    ICP_rises_amps_interpolated = interpolate_samples(icp_detection['rise_amplitude'], 
                                                   icp_detection['peak_time'],
                                                      times_for_interpolation)
    cycle_times = resp_cycles[['inspi_time','expi_time','next_inspi_time']].values
    segment_ratios = 0.4
    nbins = 100
    icp_rise_amplitude_by_resp = physio.deform_traces_to_cycle_template(data = ICP_rises_amps_interpolated, 
                                                               times = times_for_interpolation,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = nbins
                                                              )

    m_for_each_cycle = np.mean(icp_rise_amplitude_by_resp, axis = 1)
    ptp_for_each_cycle = np.ptp(icp_rise_amplitude_by_resp, axis = 1)
    relative_variation_for_each_cycle = np.abs((ptp_for_each_cycle / m_for_each_cycle)) * 100

    dates_res = resp_cycles['inspi_date'].values
    return dates_res, relative_variation_for_each_cycle

def compute_relative_variation_of_icp_modulation_by_resp(sub):
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    reader = CnsReader(data_path / sub)
    icp_stream = reader.streams['ICP']
    srate = icp_stream.sample_rate
    icp, icp_dates = icp_stream.get_data(with_times = True, apply_gain = True)
    icp = iirfilt(icp, srate, highcut = 0.7)
    times = np.arange(icp.size) / srate

    resp_cycles = resp_cycles[resp_cycles['next_inspi_time'] < times[-1]]
    cycle_times = resp_cycles[['inspi_time','expi_time','next_inspi_time']].values
    segment_ratios = 0.4
    nbins = 100
    icp_by_resp = physio.deform_traces_to_cycle_template(data = icp, 
                                                               times = times,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = nbins
                                                              )

    m_for_each_cycle = np.mean(icp_by_resp, axis = 1)
    ptp_for_each_cycle = np.ptp(icp_by_resp, axis = 1)
    relative_variation_for_each_cycle = np.abs((ptp_for_each_cycle / m_for_each_cycle)) * 100

    dates_res = resp_cycles['inspi_date'].values
    return dates_res, relative_variation_for_each_cycle

def erp_like_fig_sub_event(sub, **p):

    savefolder = base_folder / 'figures' / 'icp_co2'

    baseline_pre_win_size_mins = p['baseline_pre_win_size_mins']
    post_per_win_size_mins = p['post_per_win_size_mins']

    df_sub = load_table_nory(sub)
    df_sub = df_sub.dropna(subset = ['baseline_start_heure']).reset_index(drop = True) # remove events where don't have baseline start hour
    n_events = df_sub.shape[0] # count number of events of the patient

    side_lesion = df_sub['lateralite_lesion'].unique()[0] # get lesion side
    diagnostic = df_sub['diagnostic_initial'].unique()[0]

    cns_reader = CnsReader(base_folder / data_path / sub) # initialize reader of patient data

    da_prx = prx_job.get(sub)['prx']
    prx_trace = da_prx.values
    dates_prx = da_prx['date'].values

    da_psi = psi_job.get(sub)['psi']
    psi_trace = da_psi.values
    dates_psi = da_psi['date'].values

    all_streams = cns_reader.streams.keys()
    if 'ABP_Mean' in all_streams:
        abp_stream_name = 'ABP'
    elif 'ART_Mean' in all_streams:
        abp_stream_name = 'ART'
    icp_stream_name = 'ICP'
    co2_stream_name = 'CO2'

    stream_names = [abp_stream_name,co2_stream_name,icp_stream_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    
    dates = ds['times'].values
    abp = ds[abp_stream_name].values
    resp = ds[co2_stream_name].values
    icp = ds[icp_stream_name].values

    icp_detection = detect_icp_job.get(sub).to_dataframe()
    icp_rises = icp_detection['rise_amplitude'].values
    icp_rises_dates = icp_detection['peak_date'].values

    co2_stream = cns_reader.streams[co2_stream_name]
    co2_srate = co2_stream.sample_rate
    co2_trace = co2_stream.get_data()
    co2_times = np.arange(co2_trace.size) / co2_srate

    dates_relative_var_icp_pulse_resp, relative_var_icp_pulse_resp = compute_relative_variation_of_icp_pulse_modulation_by_resp2(sub, co2_times)
    dates_relative_var_icp_sig_resp, relative_var_icp_sig_resp = compute_relative_variation_of_icp_modulation_by_resp(sub)

    rows_stats = []
    
    for i in range(n_events): # loop over events = over icp rises
        start_per, stop_per = format_window_dates(df_sub=df_sub, i=i, window='per', win_duration=baseline_pre_win_size_mins) # get start and stop dates of the window
        start_win_date = start_per - np.timedelta64(baseline_pre_win_size_mins, 'm')
        stop_win_date = stop_per + np.timedelta64(post_per_win_size_mins, 'm')
        t0 = np.datetime64(start_win_date)
        t1 = np.datetime64(stop_win_date)
        mask_dates = (dates > t0) & (dates < t1)
        local_dates = dates[mask_dates]
        local_icp = icp[mask_dates]
        local_resp = resp[mask_dates]
        local_abp = abp[mask_dates]
        local_ppc = local_abp - local_icp

        mask_dates_prx = (dates_prx > t0) & (dates_prx < t1)
        local_dates_prx = dates_prx[mask_dates_prx]
        local_prx = prx_trace[mask_dates_prx]

        mask_dates_psi = (dates_psi > t0) & (dates_psi < t1)
        local_dates_psi = dates_psi[mask_dates_psi]
        local_psi = psi_trace[mask_dates_psi]

        mask_icp_rises_dates = (icp_rises_dates > t0) & (icp_rises_dates < t1)
        local_icp_rises = icp_rises[mask_icp_rises_dates]
        local_icp_rises_dates = icp_rises_dates[mask_icp_rises_dates]

        mask_relative_var_icp_pulse_resp_dates = (dates_relative_var_icp_pulse_resp > t0) & (dates_relative_var_icp_pulse_resp < t1)
        local_relative_var_icp_pulse_resp = relative_var_icp_pulse_resp[mask_relative_var_icp_pulse_resp_dates]
        local_relative_var_icp_pulse_resp_dates = dates_relative_var_icp_pulse_resp[mask_relative_var_icp_pulse_resp_dates]

        mask_relative_var_icp_sig_resp_dates = (dates_relative_var_icp_sig_resp > t0) & (dates_relative_var_icp_sig_resp < t1)
        local_relative_var_icp_sig_resp_dates = dates_relative_var_icp_sig_resp[mask_relative_var_icp_sig_resp_dates]
        local_relative_var_icp_sig_resp = relative_var_icp_sig_resp[mask_relative_var_icp_sig_resp_dates]

        # res = compute_heart_resp_spectral_ratio_in_icp(local_abp, local_resp, local_icp, srate, p['spectrogram_win_size_secs'])
        # dates_spectrum  = (res['times_spectrum_s'] * 1e6) * np.timedelta64(1, 'us') + local_dates[0]

        res = compute_heart_resp_spectral_ratio_in_icp3(icp = local_icp, srate= srate, wsize_secs=p['spectrogram_win_size_secs'], resp_fband=p['resp_fband'], heart_fband=p['heart_fband'], rolling_N_time=p['rolling_N_time_spectrogram'])
        dates_spectrum  = (res['times_spectrum_s'] * 1e6) * np.timedelta64(1, 'us') + local_dates[0]

        traces = [local_icp, local_abp, local_ppc, res['heart_in_icp_spectrum'], res['resp_in_icp_spectrum'], res['ratio_heart_resp_in_icp_spectrum'], local_icp_rises, local_prx, local_relative_var_icp_pulse_resp, local_relative_var_icp_sig_resp, local_psi]
        date_vectors = [local_dates, local_dates, local_dates, dates_spectrum,dates_spectrum,dates_spectrum, local_icp_rises_dates, local_dates_prx, local_relative_var_icp_pulse_resp_dates, local_relative_var_icp_sig_resp_dates, local_dates_psi]
        labels = ['icp','abp','ppc','heart_in_icp_spectrum','resp_in_icp_spectrum','ratio_heart_resp_in_icp_spectrum','heart_in_icp_tempo','prx','relative_var_icp_pulse_resp','relative_var_icp_sig_resp','psi']

        nrows = len(traces)
        row_inds = range(nrows)

        fig, axs = plt.subplots(nrows = nrows, ncols=3, sharex=False, figsize = (20, int(nrows * 3)), constrained_layout = True)
        fig.suptitle(f'{sub}\nEvent n°{i+1}\ndg : {diagnostic} - side lesion : {side_lesion}', fontsize = 12)
        for r, trace, date_vector, label in zip(row_inds,traces,date_vectors,labels):
            # print(label)
            ax = axs[r,0]
            if label in ['icp','abp','ppc']:
                trace = iirfilt(trace, srate, highcut = p['highcut_icp_abp_ppc_plot'])
            
            if label in ['heart_in_icp_tempo']:
                trace = pd.Series(trace).rolling(20).median()
                trace = trace.ffill()
                trace = trace.bfill()
                trace = trace.values

            if label in ['relative_var_icp_pulse_resp','relative_var_icp_sig_resp']:
                trace = pd.Series(trace).rolling(10).median()
                trace = trace.ffill()
                trace = trace.bfill()
                trace = trace.values
            
            # if label in ['heart_in_icp_spectrum','resp_in_icp_spectrum','ratio_heart_resp_in_icp_spectrum']:
            #     # ax.plot(date_vector, trace, color = 'k', lw = 2, alpha = 0.8, label = 'not only with ICP')
            #     # ax.plot(date_vector, res_only_with_icp[label], color = 'tab:blue', alpha = 0.8, label = 'only with ICP')
            # else:
            ax.plot(date_vector, trace, color = 'k', lw = 2)
            ax.axvspan(start_per, stop_per, label = 'Per ICP Rise', alpha = 0.1, color = 'r')

            if label in ['icp','abp','ppc']:
                ylabel = f'{label}\n(mmHg)'
            else:
                ylabel =  f'{label}'

            if label == 'prx':
                ax.set_ylim(-1,1)
            if label == 'psi':
                ax.set_ylim(-0.1,4.1)

            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, loc = 1)

            for target_sig, label_target, col_ind in zip([local_icp, local_psi],['icp','psi'],[1,2]):
                df_corr = pd.DataFrame()
                if label_target == 'icp':
                    df_corr[label_target] = iirfilt(target_sig, srate, highcut = p['highcut_icp_abp_ppc_plot'])[np.searchsorted(local_dates, date_vector)]
                    df_corr[label] = trace
                elif label_target == 'psi':
                    if target_sig.size > trace.size:
                        df_corr[label_target] = target_sig[np.searchsorted(local_dates_psi, date_vector) -1]
                        df_corr[label] = trace
                    else:
                        df_corr[label_target] = target_sig
                        df_corr[label] = trace[np.searchsorted(date_vector, local_dates_psi) - 1]
                df_corr = df_corr.dropna()
                if df_corr.shape[0] != 0:
                    try:
                        ax, res_corr = stats_quantitative(df = df_corr, xlabel=label_target,ylabel=label,ax=axs[r,col_ind], return_res=True)
                        row_stats = [sub, diagnostic, side_lesion, i+1, label_target, label, res_corr['r'], res_corr['p'], 1 if res_corr['p'] < 0.05 else 0]
                    except:
                        row_stats = [sub, diagnostic, side_lesion, i+1, label_target, label,  None, None, None]
                else:
                    row_stats = [sub, diagnostic, side_lesion, i+1, label_target, label,  None, None, None]
                rows_stats.append(row_stats)

        fig.savefig(savefolder / 'erp_like_figs_sub_event' / f'{sub}_event{i+1}.png', bbox_inches = 'tight', dpi = 500)
        plt.close(fig)
    df_stats = pd.DataFrame(rows_stats, columns = ['Patient','Diagnostic','Lesion Side','N_Event','Target_Metric','Metric_to_corr','R','p-value','Is_Significant'])
    ds = xr.Dataset(df_stats)
    return ds

def test_erp_like_fig_sub_event(sub):
    print(sub)
    ds = erp_like_fig_sub_event(sub, **erp_like_fig_sub_event_params)
    print(ds.to_dataframe())

erp_like_fig_sub_event_job = jobtools.Job(precomputedir, 'erp_like_fig_sub_event', erp_like_fig_sub_event_params, erp_like_fig_sub_event)
jobtools.register_job(erp_like_fig_sub_event_job)


def erp_like_fig_for_poster(sub, **p):

    savefolder = base_folder / 'figures' / 'icp_co2' / 'figs_poster'

    baseline_pre_win_size_mins = p['baseline_pre_win_size_mins']
    post_per_win_size_mins = p['post_per_win_size_mins']

    df_sub = load_table_nory(sub)
    df_sub = df_sub.dropna(subset = ['baseline_start_heure']).reset_index(drop = True) # remove events where don't have baseline start hour
    n_events = df_sub.shape[0] # count number of events of the patient

    side_lesion = df_sub['lateralite_lesion'].unique()[0] # get lesion side
    diagnostic = df_sub['diagnostic_initial'].unique()[0]

    cns_reader = CnsReader(base_folder / data_path / sub) # initialize reader of patient data

    da_prx = prx_job.get(sub)['prx']
    prx_trace = da_prx.values
    dates_prx = da_prx['date'].values

    da_psi = psi_job.get(sub)['psi']
    psi_trace = da_psi.values
    dates_psi = da_psi['date'].values

    all_streams = cns_reader.streams.keys()
    if 'ABP_Mean' in all_streams:
        abp_stream_name = 'ABP'
    elif 'ART_Mean' in all_streams:
        abp_stream_name = 'ART'
    icp_stream_name = 'ICP'
    co2_stream_name = 'CO2'

    stream_names = [abp_stream_name,co2_stream_name,icp_stream_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    
    dates = ds['times'].values
    abp = ds[abp_stream_name].values
    resp = ds[co2_stream_name].values
    icp = ds[icp_stream_name].values

    icp_detection = detect_icp_job.get(sub).to_dataframe()
    icp_rises = icp_detection['rise_amplitude'].values
    icp_rises_dates = icp_detection['peak_date'].values

    co2_stream = cns_reader.streams[co2_stream_name]
    co2_srate = co2_stream.sample_rate
    co2_trace = co2_stream.get_data()
    co2_times = np.arange(co2_trace.size) / co2_srate

    dates_relative_var_icp_pulse_resp, relative_var_icp_pulse_resp = compute_relative_variation_of_icp_pulse_modulation_by_resp2(sub, co2_times)
    dates_relative_var_icp_sig_resp, relative_var_icp_sig_resp = compute_relative_variation_of_icp_modulation_by_resp(sub)
    
    for i in range(n_events): # loop over events = over icp rises
        start_per, stop_per = format_window_dates(df_sub=df_sub, i=i, window='per', win_duration=baseline_pre_win_size_mins) # get start and stop dates of the window
        start_win_date = start_per - np.timedelta64(baseline_pre_win_size_mins, 'm')
        stop_win_date = stop_per + np.timedelta64(post_per_win_size_mins, 'm')
        t0 = np.datetime64(start_win_date)
        t1 = np.datetime64(stop_win_date)
        mask_dates = (dates > t0) & (dates < t1)
        local_dates = dates[mask_dates]
        local_icp = icp[mask_dates]
        local_resp = resp[mask_dates]
        local_abp = abp[mask_dates]
        local_ppc = local_abp - local_icp

        mask_dates_prx = (dates_prx > t0) & (dates_prx < t1)
        local_dates_prx = dates_prx[mask_dates_prx]
        local_prx = prx_trace[mask_dates_prx]

        mask_dates_psi = (dates_psi > t0) & (dates_psi < t1)
        local_dates_psi = dates_psi[mask_dates_psi]
        local_psi = psi_trace[mask_dates_psi]

        mask_icp_rises_dates = (icp_rises_dates > t0) & (icp_rises_dates < t1)
        local_icp_rises = icp_rises[mask_icp_rises_dates]
        local_icp_rises_dates = icp_rises_dates[mask_icp_rises_dates]

        mask_relative_var_icp_pulse_resp_dates = (dates_relative_var_icp_pulse_resp > t0) & (dates_relative_var_icp_pulse_resp < t1)
        local_relative_var_icp_pulse_resp = relative_var_icp_pulse_resp[mask_relative_var_icp_pulse_resp_dates]
        local_relative_var_icp_pulse_resp_dates = dates_relative_var_icp_pulse_resp[mask_relative_var_icp_pulse_resp_dates]

        mask_relative_var_icp_sig_resp_dates = (dates_relative_var_icp_sig_resp > t0) & (dates_relative_var_icp_sig_resp < t1)
        local_relative_var_icp_sig_resp_dates = dates_relative_var_icp_sig_resp[mask_relative_var_icp_sig_resp_dates]
        local_relative_var_icp_sig_resp = relative_var_icp_sig_resp[mask_relative_var_icp_sig_resp_dates]

        res = compute_heart_resp_spectral_ratio_in_icp3(icp = local_icp, srate= srate, wsize_secs=p['spectrogram_win_size_secs'], resp_fband=p['resp_fband'], heart_fband=p['heart_fband'], rolling_N_time=p['rolling_N_time_spectrogram'])
        dates_spectrum  = (res['times_spectrum_s'] * 1e6) * np.timedelta64(1, 'us') + local_dates[0]

        traces = [local_icp, res['heart_in_icp_spectrum'], res['resp_in_icp_spectrum'], local_icp_rises, local_relative_var_icp_sig_resp]
        date_vectors = [local_dates, dates_spectrum,dates_spectrum, local_icp_rises_dates, local_relative_var_icp_sig_resp_dates]
        labels = ['icp','heart_in_icp_spectrum','resp_in_icp_spectrum','heart_in_icp_tempo','relative_var_icp_sig_resp']

        nrows = len(traces)
        row_inds = range(nrows)

        fig, axs = plt.subplots(nrows = nrows, sharex=False, figsize = (20, int(nrows * 3)), constrained_layout = True)
        fig.suptitle(f'{sub}\nEvent n°{i+1}\ndg : {diagnostic} - side lesion : {side_lesion}', fontsize = 12)
        for r, trace, date_vector, label in zip(row_inds,traces,date_vectors,labels):
            ax = axs[r]
            if label in ['icp','abp','ppc']:
                trace = iirfilt(trace, srate, highcut = p['highcut_icp_abp_ppc_plot'])
            
            if label in ['heart_in_icp_tempo']:
                trace = pd.Series(trace).rolling(20).median()
                trace = trace.ffill()
                trace = trace.bfill()
                trace = trace.values

            if label in ['relative_var_icp_pulse_resp','relative_var_icp_sig_resp']:
                trace = pd.Series(trace).rolling(10).median()
                trace = trace.ffill()
                trace = trace.bfill()
                trace = trace.values
            
            ax.plot(date_vector, trace, color = 'k', lw = 2)
            ax.axvspan(start_per, stop_per, label = 'Per ICP Rise', alpha = 0.1, color = 'r')

            if label in ['icp','abp','ppc']:
                ylabel = f'{label}\n(mmHg)'
            else:
                ylabel =  f'{label}'

            if label == 'prx':
                ax.set_ylim(-1,1)
            if label == 'psi':
                ax.set_ylim(-0.1,4.1)

            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, loc = 1)

        fig.savefig(savefolder / f'{sub}_event{i+1}.png', bbox_inches = 'tight', dpi = 500)
        plt.close(fig)

# COMPLIANCE OVERVIEW
def compliance_overview(sub, **p):

    savefolder = base_folder / 'figures' / 'icp_co2' / 'compliance_overview'

    meta = pd.read_excel(base_folder / 'overview_data_pycns.xlsx').set_index('Patient')
    gcs_sortie = meta.loc[sub,'GCS_sortie']
    mrs_sortie = meta.loc[sub,'mRS_sortie']
    diagnostic = meta.loc[sub,'motif']

    cns_reader = CnsReader(base_folder / data_path / sub) # initialize reader of patient data
    icp_stream = cns_reader.streams['ICP']
    srate = icp_stream.sample_rate
    raw_icp, dates_icp = icp_stream.get_data(with_times = True, apply_gain = True)
    times_icp = np.arange(raw_icp.size) / srate

    assert np.any(~np.isnan(raw_icp)), np.isnan(raw_icp).sum()
    icp_plot = iirfilt(raw_icp, srate, highcut = 0.01)
    decim_factor = 100
    icp_plot = scipy.signal.decimate(icp_plot, q=decim_factor)
    new_srate_icp = srate / decim_factor
    dates_icp_plot = dates_icp[::decim_factor]
    times_icp_plot = times_icp[::decim_factor]
    icp_plot = pd.Series(icp_plot).rolling(int(new_srate_icp * 60 * 10)).mean().bfill().ffill().values

    res = compute_heart_resp_spectral_ratio_in_icp3(icp = raw_icp, srate=srate, wsize_secs=p['spectrogram_win_size_secs'], resp_fband=p['resp_fband'], heart_fband=p['heart_fband'], rolling_N_time=p['rolling_N_time_spectrogram'])
    dates_spectrum = dates_icp[np.searchsorted(times_icp, res['times_spectrum_s'])]

    da_psi = psi_job.get(sub)['psi']
    psi_trace = da_psi.values
    dates_psi = da_psi['date'].values

    dates_plot = [dates_icp_plot, dates_psi, dates_spectrum, dates_spectrum, dates_spectrum]
    traces_plot = [icp_plot, psi_trace, res['heart_in_icp_spectrum'], res['resp_in_icp_spectrum'] ,res['ratio_heart_resp_in_icp_spectrum']]
    ylabels = ['icp','psi','heart in icp','resp in icp','ratio h/r in icp']
    ylims = [(0,40),(-0.5,4.5),(-1,60),(-1,15),(0,15)]

    sizes = {label:trace for label,trace in zip(ylabels, traces_plot)}
    min_size_label = min(sizes)
    dates_corr_resample = dates_plot[ylabels.index(min_size_label)]

    rows = []

    nrows = len(dates_plot)
    fig, axs = plt.subplots(nrows=nrows, figsize = (12,nrows*3), constrained_layout = True)
    fig.suptitle(f'{sub} - dg : {diagnostic} - gcs sortie : {gcs_sortie} - mrs sortie : {mrs_sortie}', fontsize = 12)
    for r in range(len(dates_plot)):
        ax = axs[r]
        label = ylabels[r]
        dates_outcome = dates_plot[r]
        trace = traces_plot[r]
        ax.plot(dates_outcome, trace, lw = 2, color = 'k')
        ax.set_ylabel(label)
        ax.set_ylim(ylims[r][0],ylims[r][1])
        ax.set_xlim(dates_spectrum[0], dates_spectrum[-1])

        for trace_target, date_target, label_target in zip([icp_plot, psi_trace],[dates_icp_plot,dates_psi],['icp','psi']):
            trace_outcome = trace.copy()
            if trace_outcome.size > trace_target.size:
                keep_inds = np.searchsorted(dates_outcome, date_target)
                trace_outcome = trace_outcome[keep_inds - 1]
            elif trace_outcome.size < trace_target.size:
                keep_inds = np.searchsorted(date_target, dates_outcome)
                trace_target = trace_target[keep_inds - 1]
            elif trace_outcome.size == trace_target.size:
                trace_outcome = trace_outcome.copy()
                trace_target = trace_target.copy()
            assert trace_outcome.size == trace_target.size, f'{trace_outcome.size} - {trace_target.size}'

            res_corr = scipy.stats.pearsonr(trace_target, trace_outcome)
            r_pearson = res_corr.statistic
            res_corr = scipy.stats.spearmanr(trace_target, trace_outcome)
            r_spearman = res_corr.correlation
            rows.append([sub, label, label_target, r_pearson, r_spearman])
    fig.savefig(savefolder / f'{sub}.png', dpi = 300, bbox_inches = 'tight')
    plt.close(fig)

    df = pd.DataFrame(rows, columns = ['patient','x','y','r_pearson','r_spearman'])
    ds = xr.Dataset(df)
    return ds

def test_compliance_overview(sub):
    print(sub)
    ds = compliance_overview(sub, **compliance_overview_params)
    print(ds.to_dataframe())

compliance_overview_job = jobtools.Job(precomputedir, 'compliance_overview', compliance_overview_params, compliance_overview)
jobtools.register_job(compliance_overview_job)

def concat_compliance_overview_res_and_make_stats():
    df = pd.concat([compliance_overview_job.get(sub).to_dataframe() for sub in get_patient_list(['ICP'])]).reset_index(drop = True)
    # print(df.groupby(['x','y'])[['r_pearson','r_spearman']].mean())

    savefolder = base_folder / 'figures' / 'icp_co2' / 'compliance_overview'
    fig, ax = plt.subplots(figsize = (10,5), constrained_layout = True)
    sns.pointplot(data = df, 
                  x = 'x', 
                  y = 'r_spearman',
                  hue = 'y',
                  ax=ax
                  )
    fig.savefig(savefolder / 'global.png', dpi = 300 , bbox_inches = 'tight')
    plt.close(fig)
# 

def concat_stats_and_make_overview_figures():
    subs = get_nory_sub_keys()
    concat = []
    bug_subs = []
    for sub in subs:
        try:
            df_sub = erp_like_fig_sub_event_job.get(sub).to_dataframe()
        except:
            bug_subs.append(sub)
        else:
            concat.append(df_sub)
    # print(bug_subs)
    data = pd.concat(concat)
    # print(data)
    data_without_nan = data.dropna()

    for target_corr_label in ['icp','psi']:
        data_clean = data_without_nan[data_without_nan['Target_Metric'] == target_corr_label]
        data_clean = data_clean[data_clean['Metric_to_corr'] != target_corr_label].reset_index(drop = True)

        N_subs = len(subs)
        N_events = data_clean.groupby(['Patient','N_Event']).mean(True).shape[0]

        savefolder = base_folder / 'figures' / 'icp_co2' / 'erp_like_figs_sub_event' / 'overview_stats'

        fig, ax = plt.subplots(constrained_layout = True, figsize = (10,6))
        auto_stats(df = data_clean,
                predictor = 'Metric_to_corr',
                outcome = 'R',
                ax=ax,
                design = 'between',
                )
        fig.savefig(savefolder  / f'stats_R_against_{target_corr_label}.png', dpi = 500, bbox_inches = 'tight')
        plt.close(fig)

        fig, ax = plt.subplots(constrained_layout = True)
        proportion_significant_metric = data_clean.groupby('Metric_to_corr')['Is_Significant'].mean().sort_values(ascending = False)
        proportion_significant_metric.round(2).plot.bar(ax=ax)
        ax.set_title(f'Proportion of Significative Correlations vs {target_corr_label} by Metric')
        for bar in ax.containers:
            ax.bar_label(bar)
        fig.savefig(savefolder  / f'stats_proportion_signif_{target_corr_label}.png', dpi = 500, bbox_inches = 'tight')
        plt.close(fig)

        fig, ax = plt.subplots(constrained_layout = True)
        # R_values = data_clean.groupby('Metric')['R'].mean().sort_values(ascending = False)
        R_values = data_clean.groupby('Metric_to_corr')['R'].median().sort_values(ascending = False)
        R_values.round(2).plot.bar(ax=ax)
        ax.set_title(f'Mean R (vs {target_corr_label}) values by Metric')
        for bar in ax.containers:
            ax.bar_label(bar)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(f'Mean correlation coefficient (vs {target_corr_label})')
        fig.savefig(savefolder  / f'stats_sort_R_against_{target_corr_label}.png', dpi = 500, bbox_inches = 'tight')
        plt.close(fig)

        summary_stats = auto_stats_summary(df = data_clean,
                                        predictor = 'Metric_to_corr',
                                        outcome = 'R',
                                        design = 'between',
                                        )
        summary_stats['descriptive_stats']['N_Patients'] = N_subs
        summary_stats['descriptive_stats']['N_Events'] = N_events
        save_auto_stats_summary(stats_dict=summary_stats, path = savefolder / f'summary_stats_{target_corr_label}.xlsx')

        fig, ax = plt.subplots(constrained_layout = True, figsize = (20,6))
        auto_stats(df = data_clean,
                predictor = ['Metric_to_corr','Diagnostic'],
                outcome = 'R',
                ax=ax,
                design = 'between',
                )
        fig.savefig(savefolder  / f'stats_R_metric_x_diagnostic_{target_corr_label}.png', dpi = 500, bbox_inches = 'tight')
        plt.close(fig)


        fig, ax = plt.subplots(constrained_layout = True)
        order = data_clean.groupby('Metric_to_corr')['R'].mean().sort_values(ascending = False).index
        # print(order)
        sns.barplot(data = data_clean,
                    x = 'Metric_to_corr', 
                    y = 'R',
                    ax=ax,
                    order = order
                    )
        ax.set_title(f'Mean R (vs {target_corr_label}) values by Metric')
        ax.set_ylim(-1, 1)
        ax.set_ylabel(f'Mean correlation coefficient (vs {target_corr_label})')
        ax.set_xticks(ax.get_xticks(), labels = ax.get_xticklabels(), rotation = 90)
        fig.savefig(savefolder  / f'stats_sort_R_against_{target_corr_label}_barplot_seaborn.png', dpi = 500, bbox_inches = 'tight')
        plt.close(fig)
    

#####
def get_nory_sub_keys():
    df_nory = load_table_nory() # load event file
    subs = list(df_nory['ID_pseudo'].unique())
    bug_subs = ['P1','P14','P20','JR10','P58','P68'] 
    return [sub for sub in subs if not sub in bug_subs]

def compute_all():
    # run_keys_1 = [(sub,) for sub in get_patient_list(['CO2','ICP','ABP','ECG_II'])]
    # run_keys_2 = [(sub,) for sub in get_patient_list(['CO2','ICP','ART','ECG_II'])]
    # run_keys = list(set(run_keys_1 + run_keys_2))
    # print(len(run_keys))
    # run_keys = [('P3',)]
    # jobtools.compute_job_list(plot_icp_resp_job, run_keys, force_recompute=True, engine='loop')
    # jobtools.compute_job_list(plot_icp_resp_job, run_keys, force_recompute=True, engine='joblib', n_jobs=5)
    # jobtools.compute_job_list(plot_icp_resp_job, run_keys, force_recompute=True, engine='slurm',
    #                           slurm_params={'cpus-per-task':'1', 'mem':'3G', },
    #                           module_name='compute_impedances')
    # jobtools.compute_job_list(plot_compliance_metrics_correlations_job, run_keys, force_recompute=True, engine='joblib' ,n_jobs= 2)
    # jobtools.compute_job_list(plot_compliance_metrics_correlations_job, run_keys, force_recompute=True, engine='joblib',n_jobs = 10)
    # jobtools.compute_job_list(erp_like_fig_sub_event_job, run_keys, force_recompute=True, engine='loop')

    # nory_keys = [(sub,) for sub in get_nory_sub_keys()]
    # jobtools.compute_job_list(erp_like_fig_sub_event_job, nory_keys, force_recompute=False, engine='slurm',
    #                               slurm_params={'cpus-per-task':'10', 'mem':'25G', },
    #                           module_name='ICP_to_resp_analysis')
    
    run_keys = [(sub,) for sub in get_patient_list(['ICP'])]
    jobtools.compute_job_list(compliance_overview_job, run_keys, force_recompute=True, engine = 'joblib', n_jobs = 5)


if __name__ == "__main__":
    # test_plot_icp_resp('P2')
    # test_plot_compliance_metrics_correlations()
    
    # save_abstract_neurotrauma_results(n_dec = 2)
    # figure_sfar('P6')
    # figure_sfar('P14')
    # figure_sfar('P12')
    # figure_sfar('GA9')
    # figure_sfar('P4')

    # test_erp_like_fig_sub_event('BM3')
    # test_compliance_overview('MJ18')

    # compute_all()

    # print(load_table_nory('P3'))

    # concat_stats_and_make_overview_figures()

    # concat_compliance_overview_res_and_make_stats()

    erp_like_fig_for_poster('P11', **erp_like_fig_sub_event_params)

    
    

