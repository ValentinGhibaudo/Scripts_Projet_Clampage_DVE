import pandas as pd
import pycns
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import xarray as xr
from configuration import *
from scipy import signal
import physio
import matplotlib.dates as mdates
from pycns import CnsReader
import xmltodict
import scipy

def notch_filter(sig, srate, bandcut = (48,52), order = 4, ftype = 'butter', show = False, axis = -1):

    """
    IIR-Filter to notch/cut 50 Hz of signal
    """

    band = [bandcut[0], bandcut[1]]
    Wn = [e / srate * 2 for e in band]
    sos = signal.iirfilter(order, Wn, analog=False, btype='bandstop', ftype=ftype, output='sos')
    filtered_sig = signal.sosfiltfilt(sos, sig, axis=axis)

    if show:
        w, h = signal.sosfreqz(sos,fs=srate, worN = 2**18)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.scatter(w, np.abs(h), color = 'k', alpha = 0.5)
        full_energy = w[np.abs(h) >= 0.99]
        ax.axvspan(xmin = full_energy[0], xmax = full_energy[-1], alpha = 0.1)
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig

def get_patient_dates(patient):
    raw_folder = data_path / patient
    cns_reader = CnsReader(raw_folder)
    stream_keys = cns_reader.streams.keys()
    if len(stream_keys) == 0:
        print(f'{patient} : no stream available to compute dates of patient journey')
        return None, None
    else:
        start = np.min([cns_reader.streams[name].get_times()[0] for name in stream_keys])
        stop = max([cns_reader.streams[name].get_times()[-1] for name in stream_keys])
        start = np.datetime64(start, 'us')
        stop = np.datetime64(stop, 'us')
        return start,stop

def get_metadata(sub = None):
    """
    Inputs
        sub : str id of patient to get its metadata or None if all metadata. Default is None
    Ouputs 
        pd.DataFrame or pd.Series
    """
    path = base_folder / 'tab_base_neuromonito.xlsx'
    if sub is None:
        return pd.read_excel(path)
    else:
        return pd.read_excel(path).set_index('ID_pseudo').loc[sub,:]
    
def get_patient_ids():
    path = base_folder / 'tab_base_neuromonito.xlsx'
    return list(pd.read_excel(path)['ID_pseudo'])
    
def iirfilt(sig, srate, lowcut=None, highcut=None, order = 4, ftype = 'butter', verbose = False, show = False, axis = -1):

    """
    IIR-Filter of signal
    -------------------
    Inputs : 
    - sig : 1D numpy vector
    - srate : sampling rate of the signal
    - lowcut : lowcut of the filter. Lowpass filter if lowcut is None and highcut is not None
    - highcut : highcut of the filter. Highpass filter if highcut is None and low is not None
    - order : N-th order of the filter (the more the order the more the slope of the filter)
    - ftype : Type of the IIR filter, could be butter or bessel
    - verbose : if True, will print information of type of filter and order (default is False)
    - show : if True, will show plot of frequency response of the filter (default is False)
    """

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)
    filtered_sig = signal.sosfiltfilt(sos, sig, axis=axis)

    if verbose:
        print(f'{ftype} iirfilter of {order}th-order')
        print(f'btype : {btype}')


    if show:
        w, h = signal.sosfreqz(sos,fs=srate)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig


def plot_frequency_response(srate, lowcut=None, highcut=None, order = 4, ftype = 'butter'):
    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    w, h = signal.sosfreqz(sos,fs=srate)
    fig, ax = plt.subplots()
    ax.plot(w, np.abs(h))
    ax.set_title('Frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')
    plt.show()

def get_amp(sig, axis = -1):
    analytic_signal = signal.hilbert(sig, axis = axis)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def sliding_mean(sig, nwin, mode = 'same'):
    """
    Sliding mean
    ------
    Inputs =
    - sig : 1D np vector
    - nwin : N samples in the sliding window
    - mode : default = 'same' = size of the output (could be 'valid' or 'full', see doc scipy.signal.fftconvolve)
    Output =
    - smoothed_sig : signal smoothed
    """

    kernel = np.ones(nwin)/nwin
    smoothed_sig = signal.fftconvolve(sig, kernel , mode = mode)
    return smoothed_sig

def compute_rms(x):
    """Fast root mean square."""
    n = x.size
    ms = 0
    for i in range(n):
        ms += x[i] ** 2
    ms /= n
    return np.sqrt(ms)

def sliding_rms(x, sf, window=0.5, step=0.2, interp=True):
    halfdur = window / 2
    n = x.size
    total_dur = n / sf
    last = n - 1
    idx = np.arange(0, total_dur, step)
    out = np.zeros(idx.size)

    # Define beginning, end and time (centered) vector
    beg = ((idx - halfdur) * sf).astype(int)
    end = ((idx + halfdur) * sf).astype(int)
    beg[beg < 0] = 0
    end[end > last] = last
    # Alternatively, to cut off incomplete windows (comment the 2 lines above)
    # mask = ~((beg < 0) | (end > last))
    # beg, end = beg[mask], end[mask]
    t = np.column_stack((beg, end)).mean(1) / sf



    # Now loop over successive epochs
    for i in range(idx.size):
        out[i] = compute_rms(x[beg[i] : end[i]])

    # Finally interpolate
    if interp and step != 1 / sf:
        f = interpolate.interp1d(t, out, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True)
        t = np.arange(n) / sf
        out = f(t)

    return out

def get_wsize(srate, lowest_freq , n_cycles=5):
    nperseg = ( n_cycles / lowest_freq) * srate
    return int(nperseg)

def crosscorrelogram(a, b, bins):
    """
    Lazy implementation of crosscorrelogram.
    """
    diff = a[:, np.newaxis] - b[np.newaxis, :]
    count, bins = np.histogram(diff, bins)
    return count, bins

def get_rate_variablity(cycles, rate_bins, bin_size_min, colname_date, colname_time, units):
    times = cycles[colname_date].values

    start = times[0]
    stop = times[-1]
    delta = np.timedelta64(int(bin_size_min*60), 's')
    time_bins = np.arange(start, stop, delta)

    rate_dist = np.zeros((time_bins.size - 1, rate_bins.size - 1)) * np.nan
    rate = np.zeros(time_bins.size - 1) * np.nan
    rate_variability = np.zeros(time_bins.size - 1) * np.nan

    for i in range(time_bins.size - 1):

        t0, t1 = time_bins[i], time_bins[i+1]

        keep = (cycles[colname_date] > t0) & (cycles[colname_date] < t1)
        cycles_keep = cycles[keep]

        if cycles_keep.shape[0] < 2:
            continue

        d = np.diff(cycles_keep[colname_time].values)
        if units == 'Hz':
            r = 1 / d
        elif  units == 'bpm':
            r = 60 / d
        else:
            raise ValueError(f'bad units {units}')

        count, bins = np.histogram(r, bins=rate_bins, density=True)
        rate_dist[i, :] = count
        rate[i], rate_variability[i] = physio.compute_median_mad(r)

    results = dict(
    time_bins=time_bins,
    rate_bins=rate_bins,
    rate_dist=rate_dist,
    rate=rate,
    rate_variability=rate_variability,
    units=units,
    )
    return results

def plot_variability(results, ratio_saturation=4, ax=None, plot_type = '2d', color='red'):
    globals().update(results)
    
    if ax is None:
        fig, ax = plt.subplots()
        
    if plot_type == '2d':
        
        im = ax.imshow(rate_dist.T, origin='lower', aspect='auto', interpolation='None',
                 extent=[mdates.date2num(time_bins[0]), mdates.date2num(time_bins[-1]),
                         rate_bins[0], rate_bins[-1]])
        ax.plot(mdates.date2num(time_bins[:-1]), rate, color=color)
        ax.set_ylabel(f'rate [{units}]')

        im.set_clim(0, np.nanmax(rate_dist) / ratio_saturation)
    
    elif plot_type == '1d':
        ax.plot(mdates.date2num(time_bins[:-1]), rate_variability, color=color)
        ax.set_ylabel(f'rate variability [{units}]')  
    return ax

def compute_icp(raw_icp, srate, date_vector = None, show = False, lowcut = 0.08, highcut = 10, order = 4, ftype = 'butter'):
    
    icp_filt = iirfilt(raw_icp, srate, lowcut = lowcut, highcut = highcut, order = order, ftype = ftype)
    
    raw_peak_inds = physio.detect_peak(icp_filt, srate, thresh = 0.1, exclude_sweep_ms = 100)
    raw_trough_inds = physio.detect_peak(-icp_filt, srate, thresh = 0.00001, exclude_sweep_ms = 100)
    
    troughs_before_peak_sel, = np.nonzero(np.diff(np.searchsorted(raw_peak_inds, raw_trough_inds)) > 0)
    troughs_before_peak_inds = raw_trough_inds[troughs_before_peak_sel]

    peaks_after_trough_sel, = np.nonzero(np.diff(np.searchsorted(troughs_before_peak_inds, raw_peak_inds)) > 0)
    peaks_after_trough_inds = raw_peak_inds[peaks_after_trough_sel]

    if peaks_after_trough_inds[0] < troughs_before_peak_inds[0]: # first point detected has to be a trough
        peaks_after_trough_inds = peaks_after_trough_inds[1:] # so remove the first peak if is before first trough
    if troughs_before_peak_inds[-1] > peaks_after_trough_inds[-1]: # last point detected has to be a peak
        troughs_before_peak_inds = troughs_before_peak_inds[:-1] # so remove the last trough if is after last peak
    
    peaks_removed = raw_peak_inds[~np.isin(raw_peak_inds, peaks_after_trough_inds)]
    troughs_removed = raw_trough_inds[~np.isin(raw_trough_inds, troughs_before_peak_inds)]
    
    if show:
        fig, ax = plt.subplots()
        ax.plot(t, raw_icp, color = 'k')
        ax.scatter(t[raw_peak_inds], raw_icp[raw_peak_inds], color = 'g')
        ax.scatter(t[raw_trough_inds], raw_icp[raw_trough_inds], color = 'r')

        ax.scatter(t[peaks_removed], raw_icp[peaks_removed], marker = 'x', color = 'm', lw = 10)
        ax.scatter(t[troughs_removed], raw_icp[troughs_removed], marker = 'x', color = 'k', lw = 10)

        plt.show()
        
    detection = pd.DataFrame()
    detection['trough_ind'] = troughs_before_peak_inds
    detection['trough_time'] =  detection['trough_ind'] / srate
    next_trough_inds = troughs_before_peak_inds[1:]
    next_trough_inds = np.append(next_trough_inds, np.nan)
    detection['next_trough_ind'] = next_trough_inds.astype(int)
    detection['next_trough_time'] =  detection['next_trough_ind'] / srate
    detection['peak_ind'] = peaks_after_trough_inds
    detection['peak_time'] =  detection['peak_ind'] / srate
    detection = detection.iloc[:-1,:]
    detection['rise_duration'] = detection['peak_time'] - detection['trough_time']
    detection['decay_duration'] = detection['next_trough_time'] - detection['peak_time']
    detection['total_duration'] = detection['rise_duration'] + detection['decay_duration']

    detection['amplitude_at_trough'] = raw_icp[detection['trough_ind']]
    detection['amplitude_at_peak'] = raw_icp[detection['peak_ind']]
    detection['amplitude_at_next_trough'] = raw_icp[detection['next_trough_ind']]

    detection['rise_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_trough']
    detection['decay_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_next_trough']
    
    if not date_vector is None:
        detection['trough_date'] =  date_vector[detection['trough_ind']]
        detection['peak_date'] =  date_vector[detection['peak_ind']]
        
    return detection

def compute_abp(raw_abp, srate, date_vector = None, show = False, lowcut = 0.08, highcut = 10, order = 4, ftype = 'butter'):
    
    abp_filt = iirfilt(raw_abp, srate, lowcut = lowcut, highcut = highcut, order = order, ftype = ftype)
    raw_peak_inds = physio.detect_peak(abp_filt, srate, thresh = 0.1, exclude_sweep_ms = 100)
    raw_trough_inds = physio.detect_peak(-abp_filt, srate, thresh = 0.00001, exclude_sweep_ms = 100)
    
    troughs_before_peak_sel, = np.nonzero(np.diff(np.searchsorted(raw_peak_inds, raw_trough_inds)) > 0)
    troughs_before_peak_inds = raw_trough_inds[troughs_before_peak_sel]

    peaks_after_trough_sel, = np.nonzero(np.diff(np.searchsorted(troughs_before_peak_inds, raw_peak_inds)) > 0)
    peaks_after_trough_inds = raw_peak_inds[peaks_after_trough_sel]

    if peaks_after_trough_inds[0] < troughs_before_peak_inds[0]: # first point detected has to be a trough
        peaks_after_trough_inds = peaks_after_trough_inds[1:] # so remove the first peak if is before first trough
    if troughs_before_peak_inds[-1] > peaks_after_trough_inds[-1]: # last point detected has to be a peak
        troughs_before_peak_inds = troughs_before_peak_inds[:-1] # so remove the last trough if is after last peak
    
    peaks_removed = raw_peak_inds[~np.isin(raw_peak_inds, peaks_after_trough_inds)]
    troughs_removed = raw_trough_inds[~np.isin(raw_trough_inds, troughs_before_peak_inds)]
    
    if show:
        fig, ax = plt.subplots()
        ax.plot(t, raw_abp, color = 'k')
        ax.scatter(t[raw_peak_inds], raw_abp[raw_peak_inds], color = 'g')
        ax.scatter(t[raw_trough_inds], raw_abp[raw_trough_inds], color = 'r')

        ax.scatter(t[peaks_removed], raw_abp[peaks_removed], marker = 'x', color = 'm', lw = 10)
        ax.scatter(t[troughs_removed], raw_abp[troughs_removed], marker = 'x', color = 'k', lw = 10)

        plt.show()
        
    detection = pd.DataFrame()
    detection['trough_ind'] = troughs_before_peak_inds
    detection['trough_time'] =  detection['trough_ind'] / srate
    next_trough_inds = troughs_before_peak_inds[1:]
    next_trough_inds = np.append(next_trough_inds, np.nan)
    detection['next_trough_ind'] = next_trough_inds.astype(int)
    detection['next_trough_time'] =  detection['next_trough_ind'] / srate
    detection['peak_ind'] = peaks_after_trough_inds
    detection['peak_time'] =  detection['peak_ind'] / srate
    detection = detection.iloc[:-1,:]
    detection['rise_duration'] = detection['peak_time'] - detection['trough_time']
    detection['decay_duration'] = detection['next_trough_time'] - detection['peak_time']
    detection['total_duration'] = detection['rise_duration'] + detection['decay_duration']
    
    detection['amplitude_at_trough'] = raw_abp[detection['trough_ind']]
    detection['amplitude_at_peak'] = raw_abp[detection['peak_ind']]
    detection['amplitude_at_next_trough'] = raw_abp[detection['next_trough_ind']]

    detection['rise_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_trough']
    detection['decay_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_next_trough']
    
    if not date_vector is None:
        detection['trough_date'] =  date_vector[detection['trough_ind']]
        detection['peak_date'] =  date_vector[detection['peak_ind']]
        
    return detection

def interpolate_samples(data, data_times, time_vector, kind = 'linear'):
    f = scipy.interpolate.interp1d(data_times, data, fill_value="extrapolate", kind = kind)
    xnew = time_vector
    ynew = f(xnew)
    return ynew

def complex_mw(time, n_cycles , freq, a= 1, m = 0): 
    """
    Create a complex morlet wavelet by multiplying a gaussian window to a complex sinewave of a given frequency
    
    ------------------------------
    a = amplitude of the wavelet
    time = time vector of the wavelet
    n_cycles = number of cycles in the wavelet
    freq = frequency of the wavelet
    m = 
    """
    s = n_cycles / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2)) # real gaussian window
    complex_sinewave = np.exp(1j * 2 *np.pi * freq * time) # complex sinusoidal signal
    cmw = GaussWin * complex_sinewave
    return cmw

def morlet_family(srate, f_start, f_stop, n_steps, n_cycles):
    """
    Create a family of morlet wavelets
    
    ------------------------------
    srate : sampling rate
    f_start : lowest frequency of the wavelet family
    f_stop : highest frequency of the wavelet family
    n_steps : number of frequencies from f_start to f_stop
    n_cycles : number of waves in the wavelet
    """
    tmw = np.arange(-5,5,1/srate)
    freqs = np.linspace(f_start,f_stop,n_steps) 
    mw_family = np.zeros((freqs.size, tmw.size), dtype = 'complex')
    for i, fi in enumerate(freqs):
        mw_family[i,:] = complex_mw(tmw, n_cycles = n_cycles, freq = fi)
    return freqs, mw_family

def morlet_power(sig, srate, f_start, f_stop, n_steps, n_cycles, amplitude_exponent=2):
    """
    Compute time-frequency matrix by convoluting wavelets on a signal
    
    ------------------------------
    Inputs =
    - sig : the signal (1D np vector)
    - srate : sampling rate
    - f_start : lowest frequency of the wavelet family
    - f_stop : highest frequency of the wavelet family
    - n_steps : number of frequencies from f_start to f_stop
    - n_cycles : number of waves in the wavelet
    - amplitude_exponent : amplitude values extracted from the length of the complex vector will be raised to this exponent factor (default = 2 = V**2 as unit)

    Outputs = 
    - freqs : frequency 1D np vector
    - power : 2D np array , axis 0 = freq, axis 1 = time

    """
    freqs, family = morlet_family(srate, f_start = f_start, f_stop = f_stop, n_steps = n_steps, n_cycles = n_cycles)
    sigs = np.tile(sig, (n_steps,1))
    tf = signal.fftconvolve(sigs, family, mode = 'same', axes = 1)
    power = np.abs(tf) ** amplitude_exponent
    return freqs , power

def compute_spectrum_log_slope(spectrum, freqs, freq_range = [1,40], show = False):
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    f_log = np.log(freqs[mask])
    spectrum_log = np.log(spectrum[mask])

    res = scipy.stats.linregress(f_log, spectrum_log)
    a = res.slope
    
    if show:
        b = res.intercept
        fit_log = a * f_log + b
        fit = np.exp(a * f_log + b)
        
        fig, axs = plt.subplots(nrows = 2, figsize = (8,6))
        ax = axs[0]
        ax.plot(f_log, spectrum_log)
        ax.plot(f_log,  fit_log)
        ax.set_title('Slope : {:.3f}'.format(a))
        
        ax = axs[1]
        ax.semilogy(freqs[mask], spectrum[mask])
        ax.semilogy(freqs[mask],  fit)
        plt.show()
    
    return a

def load_one_eeg_chan(eeg_stream, chan, win_load_duration_hours = 0.1, apply_gain=True):
    assert chan in eeg_stream.channel_names, f'EEG stream do not have channel {chan}'
    srate = eeg_stream.sample_rate
    
    total_load_size = eeg_stream.shape[0]
    win_load_size = int(win_load_duration_hours * 3600 * srate)
    n_wins = total_load_size // win_load_size
    
    chan_ind = eeg_stream.channel_names.index(chan)
    
    
    chan_sig = np.zeros(total_load_size)
    start = 0
    for i in range(n_wins):
        stop = start + win_load_size
        chan_sig[start:stop] = eeg_stream.get_data(isel = slice(start,stop), apply_gain = apply_gain)[:,chan_ind]
        start = stop
    chan_sig[stop:] = eeg_stream.get_data(isel = slice(stop,None))[:,chan_ind]
    return chan_sig, srate

def attribute_subplots(element_list, nrows, ncols):
    assert nrows * ncols >= len(element_list), f'Not enough subplots planned ({nrows*ncols} subplots but {len(element_list)} elements)'
    subplots_pos = {}
    counter = 0
    for r in range(nrows):
        for c in range(ncols):
            if counter == len(element_list):
                break
            subplots_pos[f'{element_list[counter]}'] = [r,c]
            counter += 1
    return subplots_pos  

def get_mcolors():
    from matplotlib.colors import TABLEAU_COLORS
    return list(TABLEAU_COLORS.keys())

def detect_cross(sig, thresh):
    rises, = np.where((sig[:-1] <=thresh) & (sig[1:] >thresh)) # detect where sign inversion from - to +
    decays, = np.where((sig[:-1] >=thresh) & (sig[1:] <thresh)) # detect where sign inversion from + to -
    if rises[0] > decays[0]: # first point detected has to be a rise
        decays = decays[1:] # so remove the first decay if is before first rise
    if rises[-1] > decays[-1]: # last point detected has to be a decay
        rises = rises[:-1] # so remove the last rise if is after last decay
    return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T

def compute_prx(cns_reader, wsize_mean_secs = 10, wsize_corr_mins = 5, overlap_corr_prop = 0.8):
    all_streams = cns_reader.streams.keys()
    if 'ABP' in all_streams:
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    assert 'ICP' in all_streams, 'No ICP stream in data'
    stream_names = ['ICP',abp_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    
    df_sigs = pd.DataFrame()
    df_sigs['icp'] = ds['ICP'].values
    df_sigs['abp'] = ds[abp_name].values
    df_sigs['dates'] = ds['times'].values
    df_sigs = df_sigs.dropna()
    icp = df_sigs['icp'].values
    abp = df_sigs['abp'].values
    dates = df_sigs['dates'].values
    
    wsize_inds = int(srate * wsize_mean_secs)

    starts = np.arange(0, icp.size, wsize_inds)

    icp_down_mean = np.zeros(starts.size)
    abp_down_mean = np.zeros(starts.size)

    for i, start in enumerate(starts):
        stop = start + wsize_inds
        if stop > icp.size:
            break
        icp_down_mean[i] = np.mean(icp[start:stop])
        abp_down_mean[i] = np.mean(abp[start:stop])

    dates = dates[::wsize_inds]
    
    corrs_wsize_secs = wsize_corr_mins * 60
    n_samples_win = int(corrs_wsize_secs / wsize_mean_secs)

    n_samples_between_starts = n_samples_win - int(overlap_corr_prop * n_samples_win)

    start_win_inds = np.arange(0, icp_down_mean.size, n_samples_between_starts)

    prx_r = np.zeros(start_win_inds.size)
    prx_pval = np.zeros(start_win_inds.size)
    for i, start_win_ind in enumerate(start_win_inds):
        stop_win_ind = start_win_ind + n_samples_win
        if stop_win_ind > icp_down_mean.size:
            stop_win_ind = icp_down_mean.size
        actual_win_size = stop_win_ind - start_win_ind
        if actual_win_size > 2: # in case when last window too short to compute correlation ...
            res = scipy.stats.pearsonr(icp_down_mean[start_win_ind:stop_win_ind], abp_down_mean[start_win_ind:stop_win_ind])
            prx_r[i] = res.statistic
            prx_pval[i] = res.pvalue
        else: # ... fill last value with pre-last value
            prx_r[i] = prx_r[i-1]
            prx_pval[i] = prx_pval[i-1]
            
        
    dates = dates[start_win_inds]
    # print(np.nanmean(prx_r), np.nanstd(prx_r))
    return prx_r, prx_pval, dates

def compute_prx_and_keep_nans(cns_reader, wsize_mean_secs = 10, wsize_corr_mins = 5, overlap_corr_prop = 0.8):
    all_streams = cns_reader.streams.keys() # get all stream names

    # check if ABP or ART stream in available streams
    if 'ABP' in all_streams: 
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    assert 'ICP' in all_streams, 'No ICP stream in data' # check if ICP stream in available streams
    stream_names = ['ICP',abp_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names]) # compute srate for upsampling based on the most sampled stream
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate) # load ICP and blood pressure streams with same datetime basis
    icp = ds['ICP'].values # icp : dataset to numpy
    abp = ds[abp_name].values # abp : dataset to numpy
    dates = ds['times'].values # datetimes : dataset to numpy

    wsize_inds = int(srate * wsize_mean_secs) # compute window size in points for the local averaging

    start_mean_inds = np.arange(0, icp.size, wsize_inds) # compute start inds
    stop_mean_inds = start_mean_inds + wsize_inds # stop inds = start inds + window size in points
    stop_mean_inds[-1] = icp.size - 1 # last stop ind is replaced by the size of original signal to not slice too far

    icp_down_local_mean = np.zeros(stop_mean_inds.size) # initialize local mean icp signal
    abp_down_local_mean = np.zeros(stop_mean_inds.size) # initialize local mean abp signal
    dates_local_mean = dates[stop_mean_inds] # compute date vector of local means by slicing original dates by stop inds
    
    for i, start, stop in zip(np.arange(start_mean_inds.size), start_mean_inds, stop_mean_inds): # loop over start and stop inds
        icp_down_local_mean[i] = np.mean(icp[start:stop]) # compute local mean icp (return Nan if Nan in the window)
        abp_down_local_mean[i] = np.mean(abp[start:stop]) # compute local mean abp (return Nan if Nan in the window)

    n_samples_by_corr_win = int(wsize_corr_mins * 60 / wsize_mean_secs) # compute number of points by correlation window (seconds durations of corr win / seconds duration of local mean win) (= 30 if corr win = 5 mins and local mean win = 10 secs)
    n_samples_between_start_prx_inds = n_samples_by_corr_win - int(overlap_corr_prop * n_samples_by_corr_win) # = 6 if overlap = 80% and n_samples_by_corr_win = 30
    start_prx_inds = np.arange(0, icp_down_local_mean.size, n_samples_between_start_prx_inds) # compute start inds of prx

    prx_r = np.zeros(start_prx_inds.size) # initialize prx vector of shape start_prx_inds.size
    prx_pval = np.zeros(start_prx_inds.size) # initialize prx pval vector of shape start_prx_inds.size
    dates_prx = [] # initialize a list to store datetimes of prx computing
    for i, start_win_ind in enumerate(start_prx_inds): # loop over start inds
        stop_win_ind = start_win_ind + n_samples_by_corr_win  # compute stop ind = start ind + n_samples_by_corr_win
        if stop_win_ind > icp_down_local_mean.size: # if stop win index higher that size of local mean sig ...
            stop_win_ind = icp_down_local_mean.size # ... computing window will end at the last local mean sig point
            dates_prx.append(dates_local_mean[stop_win_ind-1]) # add a datetime corresponding to local mean date vector sliced with current stop ind - 1
        else:
            dates_prx.append(dates_local_mean[stop_win_ind]) # add a datetime corresponding to local mean date vector sliced with current stop ind
        actual_win_size = stop_win_ind - start_win_ind # compute the window size in points
        if actual_win_size > 2: # check if window size has at least two points to correlate ...
            icp_sig_win = icp_down_local_mean[start_win_ind:stop_win_ind] # slice the local mean icp sig
            abp_sig_win = abp_down_local_mean[start_win_ind:stop_win_ind] # slice the local mean abp sig
            if np.any(np.isnan(icp_sig_win)) or np.any(np.isnan(abp_sig_win)): # check if nan in the slice of local mean sig and fill with nan if it is the case
                prx_r[i] = np.nan
                prx_pval[i] = np.nan
            else: # if no nan, compute pearson correlation from scipy
                res = scipy.stats.pearsonr(icp_sig_win, abp_sig_win)
                prx_r[i] = res.statistic
                prx_pval[i] = res.pvalue
        else: # ... else fill with a nan if no two points available to correlate
            prx_r[i] = np.nan
            prx_pval[i] = np.nan
    
    dates_prx = np.array(dates_prx).astype('datetime64')
    # print(np.nanmean(prx_r), np.nanstd(prx_r))
    return prx_r, prx_pval, dates_prx

def compute_homemade_prx(cns_reader, win_size_rolling_mins = 5, highcut_Hz=0.1, ftype = 'bessel', order = 4):
    all_streams = cns_reader.streams.keys()
    if 'ABP' in all_streams:
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    assert 'ICP' in all_streams, 'No ICP stream in data'
    stream_names = ['ICP',abp_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    df_sigs = pd.DataFrame()
    df_sigs['icp'] = ds['ICP'].values
    df_sigs['abp'] = ds[abp_name].values
    df_sigs['dates'] = ds['times'].values
    df_sigs = df_sigs.dropna()
    df_sigs['icp'] = iirfilt(df_sigs['icp'], srate, highcut = highcut_Hz, ftype = ftype, order = order)
    df_sigs['abp'] = iirfilt(df_sigs['abp'], srate, highcut = highcut_Hz, ftype = ftype, order = order)
    down_samp_compute = int(srate / (highcut_Hz * 20))
    down_samp_compute = 1 if down_samp_compute < 1 else down_samp_compute
    new_srate = srate / down_samp_compute
    df_sigs = df_sigs.iloc[::down_samp_compute]
    corr = df_sigs['abp'].rolling(int(win_size_rolling_mins * 60 * srate)).corr(df_sigs['icp'])
    corr.index = df_sigs['dates']
    return corr


def init_da(coords, name = None, values = np.nan):
    dims = list(coords.keys())
    coords = coords

    def size_of(element):
        element = np.array(element)
        size = element.size
        return size

    shape = tuple([size_of(element) for element in list(coords.values())])
    data = np.full(shape, values)
    da = xr.DataArray(data=data, dims=dims, coords=coords, name = name)
    return da

def compute_suppression_ratio(sig, srate, threshold_µV = 5, win_size_sec_epoch = 0.240, win_size_sec_moving = 63):
    mask_not_suppressed = np.abs(sig) > threshold_µV
    t = np.arange(sig.size) / srate
    start_wins = np.arange(0, t[-1], win_size_sec_epoch)
    rows = []
    for i, start_win in enumerate(start_wins):
        stop_win = start_win + win_size_sec_epoch
        start_win_ind = int(start_win * srate)
        stop_win_ind = start_win_ind + int(win_size_sec_epoch * srate)
        if stop_win_ind > sig.size:
            break
        mask_not_suppressed_win = mask_not_suppressed[start_win_ind:stop_win_ind]
        is_suppressed = 0 if np.sum(mask_not_suppressed_win) > 0 else 1
        rows.append([start_win, stop_win, is_suppressed])
    df_suppression = pd.DataFrame(rows, columns = ['start_t','stop_t','is_suppressed'])

    if t[-1] < win_size_sec_moving:
        suppression_ratio = df_suppression['is_suppressed'].mean()
        start_wins = np.array([0])
    else:
        start_wins = np.arange(0, t[-1], win_size_sec_moving)
        suppression_ratio = np.zeros(start_wins.size)
        for i, start_win in enumerate(start_wins):
            stop_win = start_win + win_size_sec_moving
            if stop_win > t[-1]:
                stop_win = t[-1]
            local_df_suppression = df_suppression[(df_suppression['start_t'] >= start_win) & (df_suppression['start_t'] < stop_win)]
            suppression_ratio[i] = local_df_suppression['is_suppressed'].mean()
    suppression_ratio *= 100 # transform ratio into percentage
    return suppression_ratio, start_wins

def compute_suppression_ratio_homemade(sig, srate, threshold_µV = 5, lowcut=0.5, highcut = 40):
    sig_filtered = iirfilt(sig, srate, lowcut, highcut)
    sig_amp = get_amp(sig_filtered)
    return (np.sum(sig_amp < threshold_μV) / sig_amp.size) * 100

def compute_spectral_entropy(power, normalized = True):
    # Normalize the power spectrum
    power /= np.sum(power)
    # Compute entropy
    entropy = -np.sum(power * np.log2(power))
    if normalized:
        entropy = entropy / np.log2(power.size)
    return entropy

def get_crest_line(freqs, Sxx, freq_axis = 0):
    argmax_freqs = np.argmax(Sxx, axis = freq_axis)
    fmax_freqs = np.apply_along_axis(lambda i:freqs[i], axis = freq_axis, arr = argmax_freqs)
    return fmax_freqs

if __name__ == "__main__":
    print(get_patient_ids())
