import numpy as np
import xarray as xr
import pandas as pd
import physio
import pycns
import sys
import os
from tools import *
from configuration import *
from params_physio_csd import *
import jobtools
import joblib
import json

# LOAD ONE EEG
def load_one_eeg(sub, chan, **p):
    eeg_stream = CnsReader(data_path / sub).streams['EEG']
    raw_sig, srate = load_one_eeg_chan(eeg_stream, chan, win_load_duration_hours=p['hours_load'], apply_gain=p['apply_gain'])
    dates = eeg_stream.get_times()
    dates = dates.astype('datetime64[ns]')
    da = xr.DataArray(data = raw_sig, dims = ['time'], coords = {'time':dates}, attrs = {'srate':srate, 'name':chan})
    ds = xr.Dataset()
    ds['eeg'] = da
    return ds

def test_load_one_eeg(sub, chan):
    print(sub, chan)
    ds = load_one_eeg(sub, chan, **load_one_eeg_params)
    print(ds)

load_one_eeg_job = jobtools.Job(precomputedir, 'load_one_eeg', load_one_eeg_params, load_one_eeg)
jobtools.register_job(load_one_eeg_job)

# DETECT RESP JOB

def detect_resp(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    co2_stream = cns_reader.streams['CO2']
    dates = co2_stream.get_times()
    raw_co2 = co2_stream.get_data(with_times=False, apply_gain=False)
    srate = co2_stream.sample_rate
    co2, resp_cycles = physio.compute_respiration(raw_co2, srate, parameter_preset = 'human_co2')
    resp_cycles['inspi_date'] = dates[resp_cycles['inspi_index']]
    resp_cycles['expi_date'] = dates[resp_cycles['expi_index']]
    return xr.Dataset(resp_cycles)

def test_detect_resp(sub):
    print(sub)
    ds = detect_resp(sub, **detect_resp_params).to_dataframe()
    print(ds)

detect_resp_job = jobtools.Job(precomputedir, 'detect_resp', detect_resp_params, detect_resp)
jobtools.register_job(detect_resp_job)

# DETECT ECG JOB

def detect_ecg(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    ecg_stream = cns_reader.streams['ECG_II']
    dates = ecg_stream.get_times()
    raw_ecg = ecg_stream.get_data(with_times=False, apply_gain=False)
    srate = ecg_stream.sample_rate
    ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_preset = 'human_ecg')
    ecg_peaks['peak_date'] = dates[ecg_peaks['peak_index']]
    return xr.Dataset(ecg_peaks)

def test_detect_ecg(sub):
    print(sub)
    ds = detect_ecg(sub, **detect_ecg_params).to_dataframe()
    print(ds)

detect_ecg_job = jobtools.Job(precomputedir, 'detect_ecg', detect_ecg_params, detect_ecg)
jobtools.register_job(detect_ecg_job)

# RSA JOB

def rsa(sub, **p):
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    ecg_peaks = detect_ecg_job.get(sub).to_dataframe()
    rsa_cycles, cyclic_cardiac_rate = physio.compute_rsa(resp_cycles,
                                                        ecg_peaks,
                                                        srate=10.,
                                                        two_segment=True,
                                                        points_per_cycle=40,
                                                        )
    rsa_cycles['cycle_date'] = resp_cycles['inspi_date']
    return xr.Dataset(rsa_cycles)

def test_rsa(sub):
    print(sub)
    ds = rsa(sub, **rsa_params).to_dataframe()
    print(ds)

rsa_job = jobtools.Job(precomputedir, 'rsa', rsa_params, rsa)
jobtools.register_job(rsa_job)

# DETECT ABP JOB

def detect_abp(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    all_streams = cns_reader.streams.keys()
    if 'ABP' in all_streams:
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    abp_stream = cns_reader.streams[abp_name]
    srate_abp = abp_stream.sample_rate
    raw_abp, dates = abp_stream.get_data(with_times=True, apply_gain=True)
    abp_features = compute_abp(raw_abp, srate_abp, date_vector = dates, lowcut = p['lowcut'], highcut = p['highcut'], order = p['order'], ftype = p['ftype'])
    return xr.Dataset(abp_features)

def test_detect_abp(sub):
    print(sub)
    ds = detect_abp(sub, **detect_abp_params).to_dataframe()
    print(ds)

detect_abp_job = jobtools.Job(precomputedir, 'detect_abp', detect_abp_params, detect_abp)
jobtools.register_job(detect_abp_job)

# DETECT ICP JOB

def detect_icp(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    icp_stream = cns_reader.streams['ICP']
    srate_icp = icp_stream.sample_rate
    raw_icp, dates = icp_stream.get_data(with_times=True, apply_gain=True)
    icp_features = compute_icp(raw_icp, srate_icp, date_vector = dates, lowcut = p['lowcut'], highcut = p['highcut'], order = p['order'], ftype = p['ftype'])
    return xr.Dataset(icp_features)

def test_detect_icp(sub):
    print(sub)
    ds = detect_icp(sub, **detect_icp_params).to_dataframe()
    print(ds)

detect_icp_job = jobtools.Job(precomputedir, 'detect_icp', detect_icp_params, detect_icp)
jobtools.register_job(detect_icp_job)

# DETECT PRX JOB

def prx(sub, **p):
    cns_reader = CnsReader(data_path / sub)
    prx_r, prx_pval, dates = compute_prx_and_keep_nans(cns_reader, wsize_mean_secs = p['wsize_mean_secs'], wsize_corr_mins = p['wsize_corr_mins'], overlap_corr_prop = p['overlap_corr_prop'])
    da = xr.DataArray(data = prx_r, dims = ['date'], coords = {'date':dates})
    ds = xr.Dataset()
    ds['prx'] = da
    return xr.Dataset(ds)

def test_prx(sub):
    print(sub)
    ds = prx(sub, **prx_params).to_dataframe()
    print(ds['prx'])

prx_job = jobtools.Job(precomputedir, 'prx', prx_params, prx)
jobtools.register_job(prx_job)

# PSI 
def psi(sub, **p):
    cns_reader = pycns.CnsReader(data_path / sub, event_time_zone='Europe/Paris')
    icp_stream = cns_reader.streams['ICP']

    # Add the plugin directory to the system path, for us it is in the plugin/pulse_detection directory
    plugin_dir = base_folder / 'ICMPWaveformClassificationPlugin' / 'plugin' / 'pulse_detection'
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

# CardioRespPhaseSynchro
def crps(sub, **p):
    """
    crps = carido-respiratory phase synchronization
    """
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    inspi_ratio = resp_cycles['cycle_ratio'].median()
    cycle_times = resp_cycles[['inspi_time', 'expi_time', 'next_inspi_time']].values

    ecg_peaks = detect_ecg_job.get(sub).to_dataframe()

    rpeak_phase_resp = physio.time_to_cycle(ecg_peaks['peak_time'].values, cycle_times, segment_ratios=[inspi_ratio])
    ecg_peaks['rpeak_resp_phase'] = rpeak_phase_resp
    ecg_peaks['resp_phase'] = rpeak_phase_resp % 1
    ecg_peaks['resp_cycle'] = np.floor(rpeak_phase_resp)
    return xr.Dataset(ecg_peaks)

def test_crps(sub):
    print(sub)
    ds = crps(sub, **crps_params)
    print(ds.to_dataframe())

crps_job = jobtools.Job(precomputedir, 'crps', crps_params, crps)
jobtools.register_job(crps_job)
#

def sub_eeg_chan_keys():
    keys = []
    for sub in get_patient_list(['Scalp','ECoG'], patient_type='SD_ICU'):
        chans = CnsReader(data_path / sub).streams['EEG'].channel_names
        for chan in chans:
            key = (sub, chan)
            keys.append(key)
    return keys

def compute_all():
    # jobtools.compute_job_list(load_one_eeg_job, sub_eeg_chan_keys(), force_recompute=False, engine='loop')
    # jobtools.compute_job_list(load_one_eeg_job, sub_eeg_chan_keys(), force_recompute=False, engine='slurm',
    #                           slurm_params={'cpus-per-task':'4', 'mem':'50G', },
    #                           module_name='compute_artifacts')
    # jobtools.compute_job_list(detect_resp_job, [(sub,) for sub in get_patient_list(['CO2'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(detect_ecg_job, [(sub,) for sub in get_patient_list(['ECG_II'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(detect_ecg_job, [(sub,) for sub in get_patient_list(['ECG_II'])], force_recompute=False, engine = 'slurm',
    #                           slurm_params={'cpus-per-task':'10', 'mem':'50G', },
    #                           module_name='compute_physio_csd')

    # jobtools.compute_job_list(rsa_job, [(sub,) for sub in get_patient_list(['ECG_II','CO2'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(rsa_job, [(sub,) for sub in get_patient_list(['ECG_II','CO2'])], force_recompute=False, engine = 'slurm',
    #                         slurm_params={'cpus-per-task':'10', 'mem':'50G', },
    #                           module_name='multi_projects_jobs')

    # abp_art_subs = get_patient_list(['ABP']) + get_patient_list(['ART'])
    # abp_art_subs = list(set(abp_art_subs))
    # jobtools.compute_job_list(detect_abp_job, [(sub,) for sub in abp_art_subs], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(detect_abp_job, [(sub,) for sub in abp_art_subs], force_recompute=False, engine = 'joblib', n_jobs = 5)
    # jobtools.compute_job_list(detect_abp_job, [(sub,) for sub in abp_art_subs], force_recompute=False, engine = 'slurm',
    #                         slurm_params={'cpus-per-task':'5', 'mem':'30G', },
    #                           module_name='multi_projects_jobs')

    # jobtools.compute_job_list(detect_icp_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(detect_icp_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'joblib', n_jobs = 5)
    # jobtools.compute_job_list(detect_icp_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'slurm',
    #                         slurm_params={'cpus-per-task':'5', 'mem':'30G', },
    #                           module_name='multi_projects_jobs')

    # icp_abp_art_subs = get_patient_list(['ICP','ABP']) + get_patient_list(['ICP','ART'])
    # icp_abp_art_subs = list(set(icp_abp_art_subs))
    # jobtools.compute_job_list(prx_job, [(sub,) for sub in icp_abp_art_subs], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(prx_job, [(sub,) for sub in icp_abp_art_subs], force_recompute=False, engine = 'slurm',
    #                         slurm_params={'cpus-per-task':'5', 'mem':'30G', },
    #                           module_name='multi_projects_jobs')
    

    # jobtools.compute_job_list(psi_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(psi_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'joblib', n_jobs = 5)
    # jobtools.compute_job_list(psi_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'slurm',
    #                         slurm_params={'cpus-per-task':'10', 'mem':'50G', },
    #                           module_name='multi_projects_jobs')

    jobtools.compute_job_list(crps_job, [(sub,) for sub in get_patient_list(['ECG_II','CO2'])], force_recompute=False, engine = 'loop')

if __name__ == "__main__":
    # test_load_one_eeg('MF12','ECoGA4')
    test_detect_resp('P61')
    # test_detect_ecg('MR21')
    # test_rsa('MF12')
    # test_detect_abp('P10')
    # test_detect_icp('MF12')
    # test_prx('P73')
    # test_psi('P43')
    # test_crps('MF12')

    # compute_all()