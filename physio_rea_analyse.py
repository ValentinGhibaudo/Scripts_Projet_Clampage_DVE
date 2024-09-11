import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import xarray as xr

import pycns
import physio


def detect_ecg_peak(ecg_stream):
    ecg_raw = ecg_stream.get_data(with_times=False, apply_gain=False)
    srate = ecg_stream.sample_rate
    ecg, r_peaks = physio.compute_ecg(ecg_raw, srate)
    return ecg, r_peaks


def detect_resp_cycles(co2_stream):
    co2_raw = co2_stream.get_data(with_times=False, apply_gain=False)
    srate = co2_stream.sample_rate

    co2, cycles = physio.compute_respiration(co2_raw, srate, parameter_preset = 'human_co2')

    return cycles

    
    

def analyse_rate_variability(cycle_inds, times, srate, rate_bins, 
                            bin_size_min=30., units='Hz'):
    """
    Compute rate variability.
    This works for both ECG and respiration.
    """
    start = times[0]
    stop = times[-1]
    delta = np.timedelta64(int(bin_size_min*60), 's')
    time_bins = np.arange(start, stop, delta)

    rate_dist = np.zeros((time_bins.size - 1, rate_bins.size - 1)) * np.nan
    rate = np.zeros(time_bins.size - 1) * np.nan
    rate_varibility = np.zeros(time_bins.size - 1) * np.nan
    for i in range(time_bins.size - 1):
        t0, t1 = time_bins[i], time_bins[i+1]
        i0 = np.searchsorted(times, t0)
        i1 = np.searchsorted(times, t1)
        keep = (cycle_inds >=i0) & (cycle_inds <i1)
        cycles = cycle_inds[keep]
        
        if cycles.size < 2:
            continue

        cycles_s = cycles / srate
        d = np.diff(cycles_s)
        if units == 'Hz':
            r = 1 / d
        elif  units == 'bpm':
            r = 60 / d
        else:
            raise ValueError(f'bad units {units}')
        
        rate_dist[i, :], _ = np.histogram(r, bins=rate_bins, density=True)
        rate[i], rate_varibility[i] = physio.compute_median_mad(r)
        
    results = dict(
        time_bins=time_bins,
        rate_bins=rate_bins,
        rate_dist=rate_dist,
        rate=rate,
        rate_varibility=rate_varibility,
        units=units,
    )
    
    return results
    


def plot_rate_variability(results, title=None):
    """
    Plot the results from analyse_rate_variability()
    
    """

    # horrible trick to inject the results dict in locals variables
    globals().update(results)
    
    
    fig, axs = plt.subplots(nrows=2, sharex=True)

    ax = axs[0]
    im = ax.imshow(rate_dist.T, origin='lower', aspect='auto', interpolation='None',
             extent=[mdates.date2num(time_bins[0]), mdates.date2num(time_bins[-1]),
                     rate_bins[0], rate_bins[-1]])
    ax.plot(mdates.date2num(time_bins[:-1]), rate, color='red')
    ax.set_ylabel(f'rate [{units}]')
    
    # fig.colorbar(im)
    
    im.set_clim(0, np.nanmax(rate_dist) / 4)
    
    ax = axs[1]
    ax.plot(mdates.date2num(time_bins[:-1]), rate_varibility, color='red')
    ax.set_ylabel(f'rate variability [{units}]')

    ax.xaxis_date()
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)

    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=15, horizontalalignment='right')
    
    
    if title is not None:
        ax = axs[0]
        ax.set_title(title)

    return fig


