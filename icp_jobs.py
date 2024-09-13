import numpy as np
import xarray as xr
import pandas as pd
import physio
import pycns
import sys
import os
from tools import *
from configuration import *
from params import *
import jobtools
import joblib
import json

# DETECT ICP JOB

def detect_icp(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    icp_stream = cns_reader.streams[p['icp_chan_name'][sub]]
    srate_icp = icp_stream.sample_rate
    raw_icp, dates = icp_stream.get_data(with_times=True, apply_gain=True)
    icp_features = compute_icp(raw_icp, srate_icp, date_vector = dates, lowcut = p['lowcut'], highcut = p['highcut'], order = p['order'], ftype = p['ftype'], exclude_sweep_ms=p['exclude_sweep_ms'])
    return xr.Dataset(icp_features)

def test_detect_icp(sub):
    print(sub)
    ds = detect_icp(sub, **detect_icp_params).to_dataframe()
    print(ds)

detect_icp_job = jobtools.Job(precomputedir, 'detect_icp', detect_icp_params, detect_icp)
jobtools.register_job(detect_icp_job)

# PSI 
def psi(sub, **p):
    cns_reader = pycns.CnsReader(data_path / sub)
    icp_stream = cns_reader.streams[p['icp_chan_name'][sub]]

    # Add the plugin directory to the system path, for us it is in the plugin/pulse_detection directory
    plugin_dir = base_folder_neuro_rea / 'ICMPWaveformClassificationPlugin' / 'plugin' / 'pulse_detection'
    if str(plugin_dir) not in sys.path:
        sys.path.append(str(plugin_dir))

    # Import the necessary plugin modules
    from classifier_pipeline import ProcessingPipeline
    from pulse_detector import Segmenter
    from pulse_classifier import Classifier

    srate = icp_stream.sample_rate
    raw_signal, dates = icp_stream.get_data(with_times = True, apply_gain = True)
    time = np.arange(raw_signal.size) / srate
    assert np.any(~np.isnan(raw_signal))
    pipeline = ProcessingPipeline()
    classes, times = pipeline.process_signal(raw_signal, time)
    classification_results = np.argmax(classes, axis=1)

    # Remove the artefact class from the classification results
    non_artefact_mask = classification_results != 4
    non_artefact_classes = classification_results[non_artefact_mask]
    non_artefact_times = np.array(times)[non_artefact_mask]

    # Use rolling window to calculate PSI
    window_length = 5 * 60
    window_step = 10
    starting_time = non_artefact_times[0]

    psi_vector = []
    psi_times = []

    for win_start in np.arange(starting_time, non_artefact_times[-1] - window_length, window_step):
        # Get the classes in the time window
        win_end = win_start + window_length
        win_mask = (non_artefact_times >= win_start) & (non_artefact_times < win_end)
        win_classes = non_artefact_classes[win_mask]

        # Calculate the PSI
        class_counts = np.unique(win_classes, return_counts=True)
        psi = 0
        if len(win_classes) > 0:
            sum_count = np.sum(class_counts[1])
            for c, count in zip(class_counts[0], class_counts[1]):
                psi += (c+1) * count / sum_count

        # Append the PSI to the vector
        psi_vector.append(psi)
        psi_times.append(win_start + window_length / 2)
    psi_times = np.array(psi_times)
    psi_vector = np.array(psi_vector)
    psi_dates = dates[np.searchsorted(time, psi_times)]

    psi_da = xr.DataArray(data = psi_vector, dims = ['date'], coords=  {'date':psi_dates})
    ds = xr.Dataset()
    ds['psi'] = psi_da
    return ds

def test_psi(sub):
    print(sub)
    ds = psi(sub, **psi_params)
    print(ds['psi'])

psi_job = jobtools.Job(precomputedir, 'psi', psi_params, psi)
jobtools.register_job(psi_job)

# 
def interpolate_samples(data, data_times, time_vector, kind = 'linear'):
    f = scipy.interpolate.interp1d(data_times, data, fill_value="extrapolate", kind = kind)
    xnew = time_vector
    ynew = f(xnew)
    return ynew

def compute_heart_resp_spectral_ratio_in_icp(icp, srate, sub_name, wsize_secs = 50, resp_fband = (0.12,0.6), heart_fband = (0.8,2.5), rolling_N_time = 5, show_and_save = True):
    
    nperseg = int(wsize_secs * srate)
    nfft = int(nperseg)

    # Compute spectro ICP
    freqs, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(icp, fs = srate, nperseg =  nperseg, nfft = nfft)
    Sxx_icp = np.sqrt(Sxx_icp)
    da = xr.DataArray(data = Sxx_icp, dims = ['freq','time'], coords = {'freq':freqs, 'time':times_spectrum_s})
    resp_amplitude = da.loc[resp_fband[0]:resp_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    heart_amplitude = da.loc[heart_fband[0]:heart_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    if show_and_save:
        resp_freq = da.loc[resp_fband[0]:resp_fband[1],:].idxmax('freq')
        heart_freq = da.loc[heart_fband[0]:heart_fband[1],:].idxmax('freq')

        flim_max = heart_fband[1]
        f_mask = (freqs < flim_max)

        fig, axs = plt.subplots(nrows = 3, figsize = (9, 8), constrained_layout = True)
        fig.suptitle(sub_name, fontsize = 20)

        ax = axs[0]
        da_sel = da.loc[resp_fband[0]:heart_fband[1],:].mean('time')
        ax.plot(da_sel['freq'], da_sel.values, color = 'k', lw = 2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (mmHg)')

        ax = axs[1]
        q = 0.01
        lw = 0.5

        to_plot = Sxx_icp[f_mask, :]
        vmin = np.quantile(to_plot, q)
        vmax = np.quantile(to_plot, 1-q)
        ax.pcolormesh(times_spectrum_s, freqs[f_mask], to_plot, vmin=vmin, vmax=vmax)
        ax.plot(times_spectrum_s, resp_freq, color = 'b', lw = lw)
        ax.plot(times_spectrum_s, heart_freq, color = 'r', lw = lw)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')

        ax = axs[2]
        ax.plot(times_spectrum_s, resp_amplitude, color = 'b')
        ax.plot(times_spectrum_s, heart_amplitude, color = 'r')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')

        fig.savefig(base_folder / 'results' / 'spectrograms_icp_verif' / f'{sub_name}.png' , dpi = 200, bbox_inches = 'tight')
        plt.close(fig)

    ratio_heart_resp = heart_amplitude / resp_amplitude
    res = {'times_spectrum_s':times_spectrum_s,'heart_in_icp_spectrum':heart_amplitude.values, 'resp_in_icp_spectrum':resp_amplitude.values,'ratio_heart_resp_in_icp_spectrum':ratio_heart_resp.values}
    return res

def heart_resp_spectral_peaks(sub, **p):
    cns_reader = pycns.CnsReader(data_path / sub)
    icp_stream = cns_reader.streams[p['icp_chan_name'][sub]]
    srate = icp_stream.sample_rate
    raw_signal, dates = icp_stream.get_data(with_times = True, apply_gain = True)
    res = compute_heart_resp_spectral_ratio_in_icp(raw_signal, srate, sub, wsize_secs = 50, resp_fband = p['resp_fband'], heart_fband = p['heart_fband'], rolling_N_time = p['rolling_N_time_spectrogram'], show_and_save = p['savefig'])
    return xr.Dataset()

def test_heart_resp_spectral_peaks(sub):
    print(sub)
    ds = heart_resp_spectral_peaks(sub, **heart_resp_spectral_peaks_params)
    # print(ds['heart_resp_spectral_peaks_params'])

heart_resp_spectral_peaks_job = jobtools.Job(precomputedir, 'heart_resp_spectral_peaks', heart_resp_spectral_peaks_params, heart_resp_spectral_peaks)
jobtools.register_job(heart_resp_spectral_peaks_job)

def compute_all():
    run_keys = [(sub,) for sub in subs]
    # jobtools.compute_job_list(detect_icp_job, run_keys, force_recompute=False, engine = 'loop')
    jobtools.compute_job_list(psi_job, run_keys, force_recompute=False, engine = 'loop')

if __name__ == "__main__":
    # test_detect_icp('Patient_2024_May_16__9_33_08_427295')
    # test_psi('Patient_2024_May_16__9_33_08_427295')
    test_heart_resp_spectral_peaks('Patient_2024_May_16__9_33_08_427295')
    # compute_all()