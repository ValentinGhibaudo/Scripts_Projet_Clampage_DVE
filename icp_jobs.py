import numpy as np
import xarray as xr
import pandas as pd
import physio
import pycns
import pingouin as pg
import sys
import os
from tools import *
from configuration import *
from params import *
import jobtools
import joblib
import json
import seaborn as sns

# DETECT ICP JOB

def detect_icp(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    icp_chan_name = get_piv_chan_name(sub)
    icp_stream = cns_reader.streams[icp_chan_name]
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
    icp_chan_name = get_piv_chan_name(sub)
    icp_stream = cns_reader.streams[icp_chan_name]

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

# HEART RESP SPECTRAL PEAKS
def heart_resp_spectral_peaks(sub, **p):
    
    cns_reader = pycns.CnsReader(data_path / sub)
    icp_chan_name = get_piv_chan_name(sub)
    icp_stream = cns_reader.streams[icp_chan_name]
    srate = icp_stream.sample_rate

    raw_icp, dates = icp_stream.get_data(with_times = True, apply_gain = True)

    resp_fband = p['resp_fband']
    heart_fband = p['heart_fband']
    nperseg = int(p['spectrogram_win_size_secs'] * srate)
    rolling_N_time = p['rolling_N_time_spectrogram']
    nfft = int(nperseg)

    # Compute spectro ICP
    freqs, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(raw_icp, fs = srate, nperseg =  nperseg, nfft = nfft)
    dates_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + dates[0]
    Sxx_icp = np.sqrt(Sxx_icp)
    da = xr.DataArray(data = Sxx_icp, dims = ['freq','time'], coords = {'freq':freqs, 'time':times_spectrum_s})
    resp_amplitude = da.loc[resp_fband[0]:resp_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    heart_amplitude = da.loc[heart_fband[0]:heart_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    if p['savefig']:
        meta = get_metadata(sub)
        has_dvi = meta['DVI']

        resp_freq = da.loc[resp_fband[0]:resp_fband[1],:].idxmax('freq')
        heart_freq = da.loc[heart_fband[0]:heart_fband[1],:].idxmax('freq')

        flim_max = heart_fband[1]
        f_mask = (freqs < flim_max)

        fig, axs = plt.subplots(nrows = 3, figsize = (9, 8), constrained_layout = True)
        fig.suptitle(f'{sub} (DVI = {has_dvi})', fontsize = 20)

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
        ax.pcolormesh(dates_spectrum, freqs[f_mask], to_plot, vmin=vmin, vmax=vmax)
        ax.plot(dates_spectrum, resp_freq, color = 'b', lw = lw)
        ax.plot(dates_spectrum, heart_freq, color = 'r', lw = lw)
        ax.set_xlabel('Date')
        ax.set_ylabel('Frequency (Hz)')

        ax = axs[2]
        ax.plot(dates_spectrum, resp_amplitude, color = 'b')
        ax.plot(dates_spectrum, heart_amplitude, color = 'r')
        ax.set_xlabel('Date')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlim(dates_spectrum[0], dates_spectrum[-1])

        start_dates = get_date_windows_gmt(sub)
        win_size_hours = 1
        for win_label, c in zip(start_dates.keys(),['r','g']):
            start_span = start_dates[win_label]
            stop_span = start_span + np.timedelta64(win_size_hours, 'h')
            ax.axvspan(start_span, stop_span, color = c, alpha = 0.2, label = win_label)
        ax.legend()

        fig.savefig(base_folder / 'results' / 'spectrograms_icp_verif' / f'{sub}.png' , dpi = 200, bbox_inches = 'tight')
        plt.close(fig)

    ratio_heart_resp = heart_amplitude / resp_amplitude
    da_spectral_features = xr.DataArray(data = np.array([heart_amplitude.values, resp_amplitude.values, ratio_heart_resp.values]),
                                        dims = ['feature','date'],
                                        coords = {'feature':['heart_in_icp','resp_in_icp','ratio'], 'date':dates_spectrum}
                                        )
    ds = xr.Dataset()
    ds['spectral_features'] = da_spectral_features
    return ds

def test_heart_resp_spectral_peaks(sub):
    print(sub)
    ds = heart_resp_spectral_peaks(sub, **heart_resp_spectral_peaks_params)
    print(ds['spectral_features'])

heart_resp_spectral_peaks_job = jobtools.Job(precomputedir, 'heart_resp_spectral_peaks', heart_resp_spectral_peaks_params, heart_resp_spectral_peaks)
jobtools.register_job(heart_resp_spectral_peaks_job)

# RATIO P1 P2 
def ratio_P1P2(sub, **p):
    meta = get_metadata(sub)
    has_dvi = meta['DVI']
    cns_reader = pycns.CnsReader(data_path / sub)
    stream_name = get_piv_chan_name(sub)
    icp_stream = cns_reader.streams[stream_name]
    icp_mean_stream = cns_reader.streams[f'{stream_name}_Mean']
    icp_mean, dates_mean = icp_mean_stream.get_data(with_times = True, apply_gain = True)

    # Add the plugin directory to the system path, for us it is in the plugin/pulse_detection directory
    plugin_dir = base_folder_neuro_rea / 'package_P2_P1' 
    if str(plugin_dir) not in sys.path:
        sys.path.append(str(plugin_dir))

    # Import the necessary plugin modules
    from p2p1.subpeaks import SubPeakDetector

    srate = icp_stream.sample_rate
    raw_signal, dates = icp_stream.get_data(with_times = True, apply_gain = True)
    raw_signal[np.isnan(raw_signal)] = np.nanmedian(raw_signal) # signal must not contain Nan
    time = np.arange(raw_signal.size) / srate

    sd = SubPeakDetector(all_preds=False)
    srate_detect = int(np.round(srate))
    sd.detect_pulses(signal = raw_signal, fs = srate_detect) 
    onsets_inds, ratio_P1P2_vector = sd.compute_ratio()
    onsets_times = onsets_inds / srate_detect
    onsets_dates = dates[np.searchsorted(time, onsets_times)]
    ratio_P1P2_vector = pd.Series(ratio_P1P2_vector).rolling(window=p['N_pulse_sliding_window_fig']).mean().values
    if onsets_dates.size == ratio_P1P2_vector.size + 1:
        onsets_dates = onsets_dates[:-1]
    elif onsets_dates.size == ratio_P1P2_vector.size - 1:
        ratio_P1P2_vector = ratio_P1P2_vector[:-1]


    start_dates = get_date_windows_gmt(sub)

    fig, axs = plt.subplots(nrows = 2, constrained_layout = True)
    fig.suptitle(f'{sub} (DVI = {has_dvi})')

    ax = axs[0]
    ax.plot(onsets_dates, ratio_P1P2_vector)
    ax.set_title('P1/P2 ratios')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    start_dates = get_date_windows_gmt(sub)
    win_size_hours = 1
    for win_label, c in zip(start_dates.keys(),['r','g']):
        start_span = start_dates[win_label]
        stop_span = start_span + np.timedelta64(win_size_hours, 'h')
        ax.axvspan(start_span, stop_span, color = c, alpha = 0.2, label = win_label)
    ax.legend()

    ax = axs[1]
    ax.plot(dates_mean, icp_mean, color = 'k')
    ax.set_title('ICP Mean')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    for win_label, c in zip(start_dates.keys(),['r','g']):
        start_span = start_dates[win_label]
        stop_span = start_span + np.timedelta64(win_size_hours, 'h')
        ax.axvspan(start_span, stop_span, color = c, alpha = 0.2, label = win_label)
    ax.legend()

    fig.savefig(base_folder / 'results' / 'ratio_p1p2_verif' / f'{sub}.png', dpi = 200, bbox_inches = 'tight')
    plt.close(fig)

    ratio_P1P2_da = xr.DataArray(data = ratio_P1P2_vector, dims = ['date'], coords=  {'date':onsets_dates})
    ds = xr.Dataset()
    ds['ratio_P1P2'] = ratio_P1P2_da
    return ds

def test_ratio_P1P2(sub):
    print(sub)
    ds = ratio_P1P2(sub, **ratio_P1P2_params)
    print(ds['ratio_P1P2'])

ratio_P1P2_job = jobtools.Job(precomputedir, 'ratio_P1P2', ratio_P1P2_params, ratio_P1P2)
jobtools.register_job(ratio_P1P2_job)

# METRICS
def metrics(sub, **p):
    icp_features = detect_icp_job.get(sub).to_dataframe()
    psi_da = psi_job.get(sub)['psi']
    p1p2_da = ratio_P1P2_job.get(sub)['ratio_P1P2']
    spectral_features_da = heart_resp_spectral_peaks_job.get(sub)['spectral_features']

    cns_reader = pycns.CnsReader(data_path / sub)
    stream_name = get_piv_chan_name(sub)
    icp_mean_stream = cns_reader.streams[f'{stream_name}_Mean']
    icp_mean, dates_mean = icp_mean_stream.get_data(with_times = True, apply_gain = True)

    meta = get_metadata(sub)
    has_dvi = meta['DVI']
    start_dates = get_date_windows_gmt(sub)
    rows = []
    for win_label, start_analysis in start_dates.items():
        stop_analysis = start_analysis + np.timedelta64(p['analyzing_window_duration_hours'], 'h')

        local_icp_mmHg = np.nanmedian(icp_mean[(dates_mean > start_analysis) & (dates_mean < stop_analysis)])

        local_icp_features = icp_features[(icp_features['peak_date'] > start_analysis) & (icp_features['peak_date'] < stop_analysis)]
        local_icp_peak_amplitude_mean_mmHg = local_icp_features['rise_amplitude'].median()

        local_psi_mean = float(psi_da.loc[start_analysis:stop_analysis].median('date'))

        local_p1p2ratio_mean = float(p1p2_da.loc[start_analysis:stop_analysis].median('date'))

        local_spectral_features_da = spectral_features_da.loc[:,start_analysis:stop_analysis].median('date')
        local_heart_amplitude_mean_mmHg = float(local_spectral_features_da.loc['heart_in_icp'])
        local_resp_amplitude_mean_mmHg = float(local_spectral_features_da.loc['resp_in_icp'])
        local_ratio_hr_amplitude_mean = float(local_spectral_features_da.loc['ratio'])

        row = [sub, has_dvi, meta['Age'], meta['Sexe'], meta['Duree_sejour'], meta['Duree_DVE'], win_label,
               local_icp_mmHg, local_icp_peak_amplitude_mean_mmHg, local_psi_mean, local_p1p2ratio_mean, local_heart_amplitude_mean_mmHg, local_resp_amplitude_mean_mmHg, local_ratio_hr_amplitude_mean
               ]
        rows.append(row)
    columns = ['Patient','DVI','Age','Sexe','Duree_sejour','Duree_DVE','Période','ICP_mmHg','Pulse_Amplitude_mmHg','PSI','P1P2_ratio','Heart_Amplitude_mmHg','Resp_Amplitude_mmHg','RatioHR']
    metrics = pd.DataFrame(rows, columns = columns)
    return xr.Dataset(metrics)

def test_metrics(sub):
    print(sub)
    ds = metrics(sub, **metrics_params).to_dataframe()
    print(ds)

metrics_job = jobtools.Job(precomputedir, 'metrics', metrics_params, metrics)
jobtools.register_job(metrics_job)

# CONCAT METRICS
def concat_metrics(key, **p):
    concat = pd.concat([metrics_job.get(sub).to_dataframe() for sub in subs])
    concat = concat.reset_index(drop = True)
    return xr.Dataset(concat)

def test_concat_metrics():
    ds = concat_metrics('global_key', **concat_metrics_params).to_dataframe()
    print(ds)

concat_metrics_job = jobtools.Job(precomputedir, 'concat_metrics', concat_metrics_params, concat_metrics)
jobtools.register_job(concat_metrics_job)

def save_results_and_stats():
    metrics = concat_metrics_job.get('global_key').to_dataframe()
    metrics.to_excel(base_folder / 'results' / 'res_metrics.xlsx')
    predictors = ['DVI','Période']
    outcomes = ['ICP_mmHg','Pulse_Amplitude_mmHg','PSI','P1P2_ratio','Heart_Amplitude_mmHg','Resp_Amplitude_mmHg','RatioHR']
    outcomes = outcomes

    concat_aov = []

    nrows = len(outcomes)
    fig, axs = plt.subplots(nrows=nrows, figsize = (6, nrows * 3), constrained_layout = True)
    for r, outcome in enumerate(outcomes):
        ax = axs[r]
        keep_cols = ['Patient'] + predictors + [outcome]
        metrics_stats = metrics[keep_cols].dropna()

        aov = pg.mixed_anova(dv=outcome, between='DVI',within='Période', subject='Patient', data=metrics_stats)
        aov['outcome'] = outcome
        
        concat_aov.append(aov)
        
        sns.boxplot(data = metrics_stats,
                        x = 'DVI',
                        y = outcome,
                        hue = 'Période',
                        ax=ax,
                        palette = sns.color_palette("pastel"),
                        whis=5
                        # bw=0.3
                        )
        sns.stripplot(data = metrics_stats,
                        x = 'DVI',
                        y = outcome,
                        hue = 'Période',
                        ax=ax,
                        dodge = True,
                        size = 10,
                        legend = False
                        )
        ax.legend(loc = 'upper left', fontsize = 10, ncols = 1)
    fig.savefig(base_folder / 'results' / 'figs_interaction.png' , dpi = 500, bbox_inches = 'tight')
    plt.close(fig)

    res = pd.concat(concat_aov)
    res['p-corr'] = pg.multicomp(res['p-unc'], method = 'holm')[1]
    res['signif-unc'] = res['p-unc'].apply(lambda x:0 if x > 0.05 else 1)
    res['signif-corr'] = res['p-corr'].apply(lambda x:0 if x > 0.05 else 1)
    res.to_excel(base_folder / 'results' / 'mixed_anovas.xlsx')


        

def compute_all():
    run_keys = [(sub,) for sub in subs]
    # jobtools.compute_job_list(detect_icp_job, run_keys, force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(psi_job, run_keys, force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(heart_resp_spectral_peaks_job, run_keys, force_recompute=True, engine = 'loop')
    # jobtools.compute_job_list(ratio_P1P2_job, run_keys, force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(ratio_P1P2_job, run_keys, force_recompute=True, engine = 'slurm', 
    #                           slurm_params={'cpus-per-task':'5', 'mem':'60G', }, module_name='icp_jobs')

    jobtools.compute_job_list(metrics_job, run_keys, force_recompute=True, engine = 'loop')
    jobtools.compute_job_list(concat_metrics_job, [('global_key',)], force_recompute=True, engine = 'loop')


if __name__ == "__main__":
    # test_detect_icp('Patient_2024_May_16__9_33_08_427295')
    # test_psi('Patient_2024_May_16__9_33_08_427295')
    # test_heart_resp_spectral_peaks('Patient_2024_May_16__9_33_08_427295')
    # test_ratio_P1P2('Patient_2024_May_16__9_33_08_427295')
    # test_metrics('Patient_2024_May_16__9_33_08_427295')
    # test_concat_metrics()

    compute_all()

    # save_results_and_stats()