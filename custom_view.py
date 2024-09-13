import numpy as np
from tools import *
from params import *
import scipy
import physio

class ECG_Detections:
    name = 'ECG\nDetections\nmV'
    
    def __init__(self, stream, ecg_features):
        self.stream = stream
        self.ecg_features = ecg_features
        
    def plot(self, ax, t0, t1):
        sig, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True, apply_gain=True)
        
        if not sig is None:
            ecg_features = self.ecg_features
            local_peaks = ecg_features[(ecg_features['peak_date'] > t0) & (ecg_features['peak_date'] < t1)]

            local_peak_dates = local_peaks['peak_date'].values
            local_peak_inds = np.searchsorted(dates, local_peak_dates)

            ax.plot(dates, sig, color='k')
            ax.scatter(local_peak_dates, sig[local_peak_inds], color='m')
        else:
            ax.plot()
        
        
class Resp_Detections:
    name = 'Resp\nDetections\nAU'
    
    def __init__(self, stream, resp_features):
        self.stream = stream
        self.resp_features = resp_features
        
    def plot(self, ax, t0, t1):
        sig, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True, apply_gain=True)
        
        if not sig is None:
            resp_features = self.resp_features
            local_resp_features = resp_features[(resp_features['inspi_date'] > t0) & (resp_features['expi_date'] < t1)]

            local_inspi_dates = local_resp_features['inspi_date'].values
            local_expi_dates = local_resp_features['expi_date'].values
            local_inspi_inds= np.searchsorted(dates, local_inspi_dates)
            local_expi_inds = np.searchsorted(dates, local_expi_dates)

            ax.plot(dates, sig, color='k')
            ax.scatter(local_inspi_dates, sig[local_inspi_inds], color='g')
            ax.scatter(local_expi_dates, sig[local_expi_inds], color='r')
        else:
            ax.plot()
            
class ABP_Detections:
    name = 'ABP\nDetections\nmmHG'
    
    def __init__(self, stream, abp_features):
        self.stream = stream
        self.abp_features = abp_features
        
    def plot(self, ax, t0, t1):
        sig, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True, apply_gain=True)
        
        if not sig is None:
            abp_features = self.abp_features
            local_abp_features = abp_features[(abp_features['trough_date'] > t0) & (abp_features['peak_date'] < t1)]

            local_trough_dates = local_abp_features['trough_date'].values
            local_peak_dates = local_abp_features['peak_date'].values
            local_trough_inds= np.searchsorted(dates, local_trough_dates)
            local_peak_inds = np.searchsorted(dates, local_peak_dates)

            ax.plot(dates, sig, color='r')
            ax.scatter(local_trough_dates, sig[local_trough_inds], color='y')
            ax.scatter(local_peak_dates, sig[local_peak_inds], color='g')
        else:
            ax.plot()
            
class ICP_Detections:
    name = 'ICP\nDetections\nmmHg'
    
    def __init__(self, stream, icp_features):
        self.stream = stream
        self.icp_features = icp_features
        
    def plot(self, ax, t0, t1):
        sig, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True, apply_gain=True)
        
        if not sig is None:
            icp_features = self.icp_features
            local_icp_features = icp_features[(icp_features['trough_date'] > t0) & (icp_features['peak_date'] < t1)]

            local_trough_dates = local_icp_features['trough_date'].values
            local_peak_dates = local_icp_features['peak_date'].values
            local_trough_inds= np.searchsorted(dates, local_trough_dates)
            local_peak_inds = np.searchsorted(dates, local_peak_dates)

            ax.plot(dates, sig, color='maroon')
            ax.scatter(local_trough_dates, sig[local_trough_inds], color='y')
            ax.scatter(local_peak_dates, sig[local_peak_inds], color='g')
        else:
            ax.plot()
        ax.set_ylim(-5, 50)

class Bipolar:
    name = 'Bipolar'

    def __init__(self, stream, chan1, chan2, down_sampling_factor = 1, lowcut = None, highcut = None, centering = True):
        self.stream = stream
        self.chan1_name = chan1
        self.chan2_name = chan2
        if isinstance(chan1, str):
            chan1 = stream.channel_names.index(chan1)
        if isinstance(chan2, str):
            chan2 = stream.channel_names.index(chan2)
        self.ind1 = chan1
        self.ind2 = chan2
        self.down_sampling_factor = down_sampling_factor
        self.lowcut = lowcut
        self.highcut = highcut
        self.centering = centering
       
    def plot(self, ax, t0, t1):
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        srate = self.stream.sample_rate
        sig = sigs[:, self.ind2] - sigs[:, self.ind1]
        # if 'ECoG' in self.chan_name1:
        sig = sig / 1000 # µV to mV
        unit = 'mV'

        if self.centering:
            sig = sig - np.mean(sig)
            # sig = sig - sig[0]
        if not self.lowcut is None or not self.highcut is None:
            sig = iirfilt(sig, srate, lowcut = self.lowcut, highcut = self.highcut, ftype = 'bessel')
        sig = scipy.signal.decimate(sig, q=self.down_sampling_factor)
        times = times[::self.down_sampling_factor]
        
        ax.plot(times, sig, color='r', lw = 0.8)
        ax.set_ylim(sig.min(), sig.max())
        ax.set_ylabel(f'{self.chan1_name}\n-\n{self.chan2_name}\n[{unit}]')
        
class Amplitude_Envelope:
    name = 'Amp Env'

    def __init__(self, stream, chan_name, lowcut = 0.5, highcut = 40, highcut_amp = 0.01):
        self.stream = stream
        self.lowcut = lowcut
        self.highcut = highcut
        self.highcut_amp = highcut_amp
        self.chan_name = chan_name
        
    def plot(self, ax, t0, t1):
        stream = self.stream
        srate = stream.sample_rate

        lowest_freq = min([self.lowcut, self.highcut_amp])
        compute_margin_secs = 1 / lowest_freq * 2 # 5 cycles of the lowest oscillation of the resulting signal as margin
        t0_with_margin = np.datetime64(t0) - np.timedelta64(int(compute_margin_secs) , 's')
        t1_with_margin = np.datetime64(t1) + np.timedelta64(int(compute_margin_secs) , 's')
        sigs, dates_with_margin = self.stream.get_data(sel=slice(t0_with_margin, t1_with_margin), with_times=True,
                                          apply_gain=True)
        sigs = sigs - np.median(sigs, axis = 1)[:,None]

        chan_name = self.chan_name
        if not '-' in chan_name: # monopolar case
            sig = sigs[:,stream.channel_names.index(chan_name)]
        else:
            chan_name1, chan_name2 = chan_name.split('-')
            sig = sigs[:,stream.channel_names.index(chan_name1)] - sigs[:,stream.channel_names.index(chan_name2)]
        if self.highcut > 48:
            sig = notch_filter(sig, srate)
        if self.highcut > 98:
            sig = notch_filter(sig, srate, bandcut = (98,102))
        ac = iirfilt(sig, srate, lowcut = self.lowcut, highcut = self.highcut, ftype = 'bessel')
        ac_envelope = get_amp(ac)
        ac_env_smoothed = iirfilt(ac_envelope, srate, lowcut = None, highcut = self.highcut_amp)
        down_sampling_factor = int(srate / (self.highcut_amp * 10))
        down_sampling_factor = 1 if down_sampling_factor < 1 else down_sampling_factor
        ac_env_smoothed = ac_env_smoothed[::down_sampling_factor]
        dates_with_margin = dates_with_margin[::down_sampling_factor]
        mask_dates_without_margin = (dates_with_margin >= np.datetime64(t0)) & (dates_with_margin < np.datetime64(t1))
        dates_plot = dates_with_margin[mask_dates_without_margin]
        ac_env_smoothed_plot = ac_env_smoothed[mask_dates_without_margin]
        ax.plot(dates_plot, ac_env_smoothed_plot, color='k', lw = 1)
        vmin, vmax = ac_env_smoothed_plot.min(), ac_env_smoothed_plot.max()
        ax.set_ylim(vmin - vmin / 10, vmax + vmax / 10)
        ax.set_ylabel(f'Amp\n{self.chan_name}\n[µV]')

class Amplitude_Envelope_Show_Automated_Detection:
    name = 'Amp Env'

    def __init__(self, stream, chan, depols_detections, lowcut = 0.001, highcut = 0.01):
        self.stream = stream
        self.lowcut = lowcut
        self.highcut = highcut
        self.chan_name = chan
        self.depols_detections = depols_detections
        
    def plot(self, ax, t0, t1):
        stream = self.stream
        srate = stream.sample_rate

        compute_margin_secs = 1 / self.lowcut * 2 # 5 cycles of the lowest oscillation of the resulting signal as margin
        t0_with_margin = np.datetime64(t0) - np.timedelta64(int(compute_margin_secs) , 's')
        t1_with_margin = np.datetime64(t1) + np.timedelta64(int(compute_margin_secs) , 's')
        sigs, dates_with_margin = self.stream.get_data(sel=slice(t0_with_margin, t1_with_margin), with_times=True,
                                          apply_gain=True)

        chan_name = self.chan_name
        if not '-' in chan_name: # monopolar case
            sig = sigs[:,stream.channel_names.index(chan_name)]
        else:
            chan_name1, chan_name2 = chan_name.split('-')
            sig = sigs[:,stream.channel_names.index(chan_name1)] - sigs[:,stream.channel_names.index(chan_name2)]
        
        ac = iirfilt(sig, srate, lowcut = self.lowcut, highcut = self.highcut, ftype = 'bessel')
        ac_envelope = get_amp(ac)
        down_sampling_factor = int(srate / (self.highcut * 10))
        down_sampling_factor = 1 if down_sampling_factor < 1 else down_sampling_factor
        ac_envelope = ac_envelope[::down_sampling_factor]
        dates_with_margin = dates_with_margin[::down_sampling_factor]
        mask_dates_without_margin = (dates_with_margin >= np.datetime64(t0)) & (dates_with_margin < np.datetime64(t1))
        dates_plot = dates_with_margin[mask_dates_without_margin]
        ac_env_smoothed_plot = ac_envelope[mask_dates_without_margin]
        ax.plot(dates_plot, ac_env_smoothed_plot, color='k', lw = 1)
        vmin, vmax = ac_env_smoothed_plot.min(), ac_env_smoothed_plot.max()
        ax.set_ylim(vmin - vmin / 10, vmax + vmax / 10)
        ax.set_ylabel(f'Amp\n{self.chan_name}\n[µV]')

        depols = self.depols_detections
        local_depols = depols[(depols['depol_start_date'] > dates_plot[0]) & (depols['depol_stop_date'] < dates_plot[-1])]
        local_start_depols_dates = local_depols['depol_start_date'].values
        local_stop_depols_dates = local_depols['depol_stop_date'].values
        local_start_depols_inds= np.searchsorted(dates_plot, local_start_depols_dates)
        local_stop_depols_inds = np.searchsorted(dates_plot, local_stop_depols_dates)
        ax.scatter(local_start_depols_dates, ac_env_smoothed_plot[local_start_depols_inds], color='g')
        ax.scatter(local_stop_depols_dates, ac_env_smoothed_plot[local_stop_depols_inds], color='r')

class Amplitude_Envelope_MultiChan:
    name = 'Amp Env Multi'

    def __init__(self, stream, chan_type, lowcut = 0.001, highcut = 0.01, quantile_saturation = 0.01, plot_mode = '2D', exclude_chans = None):
        self.stream = stream
        self.lowcut = lowcut
        self.highcut = highcut
        self.chan_type = chan_type
        self.quantile_saturation = quantile_saturation
        self.plot_mode = plot_mode
        self.exclude_chans = exclude_chans
        
    def plot(self, ax, t0, t1):
        stream = self.stream
        sigs, dates = stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        srate = stream.sample_rate
        down_sampling_factor = int(srate / (self.highcut * 4))
        down_sampling_factor = 1 if down_sampling_factor < 1 else down_sampling_factor
        # print(down_sampling_factor)
        sigs = sigs[::down_sampling_factor,:]
        dates = dates[::down_sampling_factor]
        new_srate = srate / down_sampling_factor
        # print(new_srate)
        all_chans = stream.channel_names
        chan_type = self.chan_type
        if "ECoG" in chan_type:
            sel_chans = [chan for chan in all_chans if "ECoG" in chan]
        else:
            sel_chans = [chan for chan in all_chans if not "ECoG" in chan]
        if self.exclude_chans is not None:
            sel_chans = [chan for chan in sel_chans if not chan in self.exclude_chans]
        sel_chans_inds = [all_chans.index(chan) for chan in sel_chans]
        sigs = sigs[:,sel_chans_inds]
        try:
            sigs_filtered = iirfilt(sigs, new_srate, lowcut = self.lowcut, highcut = self.highcut, ftype = 'bessel', axis = 0)
            sigs_amplitude_envelope = get_amp(sigs_filtered, axis = 0)
            sigs_amplitude_envelope_smoothed = sigs_amplitude_envelope.copy()
            # sigs_amplitude_envelope_smoothed = np.log(sigs_amplitude_envelope_smoothed)
            if self.plot_mode == '2D':
                vmin = np.quantile(sigs_amplitude_envelope_smoothed, self.quantile_saturation)
                vmax = np.quantile(sigs_amplitude_envelope_smoothed, 1 - self.quantile_saturation)
                ax.pcolormesh(dates, sel_chans, sigs_amplitude_envelope_smoothed.T, cmap = 'viridis', vmin = vmin, vmax=vmax)
                ax.set_ylabel(f'{self.chan_type}\n{round(vmin,1)}-{round(vmax,1)}\n[µV]')
            elif self.plot_mode == '1D':
                for chan_ind, chan_name in enumerate(sel_chans):
                    ax.plot(dates, sigs_amplitude_envelope_smoothed[:,chan_ind], label = chan_name)
                ax.legend(fontsize = 6)
                ax.set_ylabel(f'Amplitude Envelope\n[µV]')
                # vmin = np.min(sigs_amplitude_envelope_smoothed)
                # vmax = np.max(sigs_amplitude_envelope_smoothed)
                # ax.set_ylim(vmin - vmin/10, vmax + vmax/10)
                ax.set_ylim(0, 6000) # µV
            else:
                ax.plot()
        except:
            ax.plot()
        
class Spectrogram_eeg:
    name = 'Spectro eeg'

    def __init__(self, stream, chan_name, wsize, lf=None, hf=None, scaling = 'log', saturation_quantile = None, cmap = 'viridis'):
        self.stream = stream
        self.wsize = wsize
        self.chan_name = chan_name
        self.lf = lf
        self.hf = hf
        self.saturation_quantile = saturation_quantile
        self.scaling = scaling
        self.cmap = cmap
        
    def plot(self, ax, t0, t1):
        stream = self.stream
        srate = stream.sample_rate
        chan_name = self.chan_name
        if not '-' in chan_name:
            chan_ind = stream.channel_names.index(chan_name)
            sigs, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                              apply_gain=True)
            sig = sigs[:,chan_ind]
        else:
            chan1, chan2 = chan_name.split('-')
            chan_ind1, chan_ind2 = stream.channel_names.index(chan1), stream.channel_names.index(chan2)
            sigs, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                              apply_gain=True)
            sig = sigs[:,chan_ind1] -  sigs[:,chan_ind2]     
        
        lf = self.lf
        hf = self.hf
        
        down_sample_factor = int(srate / (hf * 3))
        if down_sample_factor >=1:
            sig = scipy.signal.decimate(sig, q = down_sample_factor)
            srate /= down_sample_factor
        freqs, times_spectrum_s, Sxx = scipy.signal.spectrogram(sig, fs = srate, nperseg = int(self.wsize * srate))

        times_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + dates[0]

        if lf is None and hf is None:
            f_mask = (freqs>=freqs[0]) and (freqs<=freqs[-1])
        elif lf is None and not hf is None:
            f_mask = (freqs<=hf)
        elif not lf is None and hf is None:
            f_mask = (freqs>=lf)
        else:
            f_mask = (freqs>=lf) & (freqs<=hf)
        
        relative_cumsum = np.cumsum(Sxx, axis = 0) / np.sum(Sxx, axis = 0)
        median_freq = np.apply_along_axis(lambda x:freqs[np.searchsorted(x, 0.5)], arr = relative_cumsum, axis = 0)
        spectral_edge_freq = np.apply_along_axis(lambda x:freqs[np.searchsorted(x, 0.95)], arr = relative_cumsum, axis = 0)
        
        data = Sxx[f_mask,:]
        
        assert self.scaling in ['log','dB',None], f'{self.scaling} is not a valid parameter (log or dB or None)'
        if self.scaling == 'log':
            data = np.log(data)
        elif self.scaling == 'dB':
            data = 10 * np.log10(data)
        elif self.scaling is None:
            data = data
            
        if not self.saturation_quantile is None:
            vmin = np.quantile(data, self.saturation_quantile)
            vmax = np.quantile(data, 1 - self.saturation_quantile)
            ax.pcolormesh(times_spectrum, freqs[f_mask], data, vmin=vmin , vmax=vmax, cmap = self.cmap)
        else:
            ax.pcolormesh(times_spectrum, freqs[f_mask], data, cmap = self.cmap)
        # ax.plot(times_spectrum, median_freq, color = 'k', ls = '--', alpha = 0.3)
        # ax.plot(times_spectrum, spectral_edge_freq, color = 'k', alpha = 0.3)
        ax.set_ylim(lf, hf)
        ax.set_ylabel(f'Spectro EEG\n{chan_name}\nFrequency (Hz)')
        # ax.set_yscale('log')
        # for i in [4,8,12]:
        #     ax.axhline(i, color = 'r')
        # ax.set_ylim(lf, hf)
        
class Total_power:
    name = 'Total_power'

    def __init__(self, stream, chan_name, wsize, lf=None, hf=None):
        self.stream = stream
        self.srate = stream.sample_rate
        self.chan_ind = stream.channel_names.index(chan_name)
        self.wsize = wsize
        self.lf = lf
        self.hf = hf
        
    def plot(self, ax, t0, t1):
        lf = self.lf
        hf = self.hf
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        sig = sigs[:,self.chan_ind]
        
        freqs, times_spectrum_s, Sxx = scipy.signal.spectrogram(sig, fs = self.srate, nperseg = int(self.wsize * self.srate))
        times_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + times[0]

        if lf is None and hf is None:
            f_mask = (freqs>=freqs[0]) & (freqs<=freqs[-1])
        elif lf is None and not hf is None:
            f_mask = (freqs<=hf)
        elif not lf is None and hf is None:
            f_mask = (freqs>=lf)
        else:
            f_mask = (freqs>=lf) & (freqs<=hf)

        data = Sxx[f_mask,:]
        total_power = data.sum(axis = 0) 
        total_power = np.log(total_power)
        ax.axhline(20, color = 'r', ls = '--')
        ax.plot(times_spectrum, total_power, color = 'y')
        # ax.set_ylim(total_power.min(), total_power.max())
        ax.set_ylim(8, 30)
        
class Spectrum_Slope:
    name = 'Spectrum_Slope'

    def __init__(self, stream, chan_name, wsize):
        self.stream = stream
        self.srate = stream.sample_rate
        self.chan_ind = stream.channel_names.index(chan_name)
        self.wsize = wsize
        
    def plot(self, ax, t0, t1):
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        sig = sigs[:,self.chan_ind]
        
        freqs, times_spectrum_s, Sxx = scipy.signal.spectrogram(sig, fs = self.srate, nperseg = int(self.wsize * self.srate))
        times_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + times[0]

        slopes = np.apply_along_axis(compute_spectrum_log_slope, axis = 0, arr=Sxx, freqs=freqs)
        ax.plot(times_spectrum, slopes, color = 'tab:purple')
        ax.axhline(-1, color = 'r', ls = '--')
        ax.set_ylim(-3.2, 1.2)
                
class Spectrogram_bio:
    name = 'Spectro bio'

    def __init__(self, stream, wsize, lf=None, hf=None, log_transfo = False, saturation_quantile = None, overlap_prct = 0.5, nfft_factor = 1, power_or_amplitude = 'power'):
        self.stream = stream
        self.srate = stream.sample_rate
        self.wsize = wsize
        self.lf = lf
        self.hf = hf
        self.overlap_prct = overlap_prct
        self.nfft_factor = nfft_factor
        self.log_transfo = log_transfo
        self.saturation_quantile = saturation_quantile
        self.power_or_amplitude = power_or_amplitude
        
    def plot(self, ax, t0, t1):
        lf = self.lf
        hf = self.hf
        
        times = self.stream.get_times()
        
        i0 = np.searchsorted(times, np.datetime64(t0))
        i1 = np.searchsorted(times, np.datetime64(t1))
        
        sig, times = self.stream.get_data(isel=slice(i0, i1), with_times=True,
                                          apply_gain=True)
        
        if sig.shape[0] == 0 or np.any(np.isnan(sig)):
            ax.plot()
        else:
            nperseg = int(self.wsize * self.srate)
            noverlap = int(nperseg * self.overlap_prct)
            nfft = int(nperseg * self.nfft_factor)
            freqs, times_spectrum_s, Sxx = scipy.signal.spectrogram(sig, fs = self.srate, nperseg =  nperseg, noverlap = noverlap, nfft = nfft)
            if self.power_or_amplitude == 'amplitude': 
                Sxx = np.sqrt(Sxx)
            times_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + times[0]
            
            if lf is None and hf is None:
                f_mask = (freqs>=freqs[0]) and (freqs<=freqs[-1])
            elif lf is None and not hf is None:
                f_mask = (freqs<=hf)
            elif not lf is None and hf is None:
                f_mask = (freqs>=lf)
            else:
                f_mask = (freqs>=lf) & (freqs<=hf)

            if self.log_transfo:
                data = np.log(Sxx[f_mask,:])
            else:
                data = Sxx[f_mask,:]

            if not self.saturation_quantile is None:
                vmin = np.quantile(data, self.saturation_quantile)
                vmax = np.quantile(data, 1 - self.saturation_quantile)
                ax.pcolormesh(times_spectrum, freqs[f_mask], data, vmin=vmin , vmax=vmax)
            else:
                ax.pcolormesh(times_spectrum, freqs[f_mask], data)
            
class Respi_Rate:
    name = 'Respi_Rate'

    def __init__(self, resp_features, rate_bins_resp = np.arange(5, 30, 0.5), resp_wsize_in_mins = 4, ratio_sat = 4, units = 'bpm'):
        self.rate_bins_resp = rate_bins_resp
        self.resp_wsize_in_mins = resp_wsize_in_mins
        self.ratio_sat = ratio_sat
        self.resp_features = resp_features
        self.units = units
        
    def plot(self, ax, t0, t1):

        resp_features = self.resp_features
        local_resp_features = resp_features[(resp_features['inspi_date'] > t0) & (resp_features['inspi_date'] < t1)]
        
        if not local_resp_features.shape[0] == 0:
            try:
                res = get_rate_variablity(cycles = local_resp_features, 
                                        rate_bins = self.rate_bins_resp, 
                                        bin_size_min = self.resp_wsize_in_mins, 
                                        colname_date = 'inspi_date', 
                                        colname_time = 'inspi_time', 
                                        units = self.units
                                        )

                plot_variability(res, ax=ax, ratio_saturation = self.ratio_sat, plot_type = '2d')
                ax.set_ylim(self.rate_bins_resp[0], self.rate_bins_resp[-1])
                ax.set_ylabel(f'Respi\nrate\n[{self.units}]')
            except:
                ax.plot()

            
class Heart_Rate:
    name = 'Heart_Rate'

    def __init__(self, ecg_peaks, step_bins_ecg = 2,  hrv_wsize_in_mins = 2, ratio_sat = 4, plot_type='2d'):
        self.step_bins_ecg = step_bins_ecg
        self.hrv_wsize_in_mins = hrv_wsize_in_mins
        self.ratio_sat = ratio_sat
        self.ecg_peaks = ecg_peaks
        self.plot_type = plot_type
        
    def plot(self, ax, t0, t1):
        
        ecg_peaks = self.ecg_peaks
        local_peaks = ecg_peaks[(ecg_peaks['peak_date'] > t0) & (ecg_peaks['peak_date'] < t1)]
        rri = 60 / np.diff(local_peaks['peak_time'].values)
        min_rate = np.quantile(rri, 0.001) - 5
        max_rate = np.quantile(rri, 0.999) + 5
        
        rate_bins_ecg = np.arange(min_rate, max_rate, self.step_bins_ecg)
        
        if not local_peaks.shape[0] == 0:
            try:
                res = get_rate_variablity(cycles = local_peaks, 
                                        rate_bins = rate_bins_ecg, 
                                        bin_size_min = self.hrv_wsize_in_mins, 
                                        colname_date = 'peak_date', 
                                        colname_time = 'peak_time', 
                                        units = 'bpm')
                

                plot_variability(res, ax=ax, ratio_saturation = self.ratio_sat, plot_type = self.plot_type)
                if self.plot_type == '2d':
                    ax.set_ylim(min_rate, max_rate)
                elif self.plot_type == '1d':
                    # ax.set_ylim(0, np.quantile(res['rate_variability'], 0.999))
                    ax.set_ylim(0, 10)
                ax.set_ylabel(f'Heart\nrate\n[bpm]')
            except:
                ax.plot()

class CRPS:
    name = 'CRPS'

    def __init__(self, ecg_peaks_resp, plot_type='2d', step_secs_1d = 30, win_duration_mins_1d = 3):
        self.ecg_peaks_resp = ecg_peaks_resp
        self.plot_type = plot_type
        self.step_secs_1d = step_secs_1d
        self.win_duration_mins_1d = win_duration_mins_1d
        
    def plot(self, ax, t0, t1):
        
        ecg_peaks_resp = self.ecg_peaks_resp
        local_peaks = ecg_peaks_resp[(ecg_peaks_resp['peak_date'] > t0) & (ecg_peaks_resp['peak_date'] < t1)]
        
        # rate_bins_ecg = np.arange(min_rate, max_rate, self.step_bins_ecg)
        
        if not local_peaks.shape[0] == 0:
            try:
                if self.plot_type == '2d':
                    ax.scatter(local_peaks['peak_date'], local_peaks['resp_phase'], s=0.5, color = 'k')
                    ax.set_ylabel('resp phase')
                elif self.plot_type == '1d':
                    start_compute = np.datetime64(t0)
                    stop_compute = np.datetime64(t1)
                    starts = np.arange(start_compute, stop_compute, np.timedelta64(self.step_secs_1d, "s"))
                    stops = starts + np.timedelta64(self.win_duration_mins_1d, "m")
                    bins = np.arange(0,1,0.05)
                    crpss = np.zeros((starts.size))
                    Ns = []
                    # count_matrix = np.zeros((bins.size - 1, starts.size))
                    for i, start, stop in zip(range(starts.size), starts, stops):
                        if stop < stop_compute:
                            mask = (ecg_peaks_resp['peak_date'] > start) & (ecg_peaks_resp['peak_date'] <= stop)
                            local_peaks = ecg_peaks_resp[mask]
                            Ns.append(local_peaks.shape[0])
                            count, bins = np.histogram(local_peaks['resp_phase'], bins)
                            # count_matrix[:,i] = count
                            count = count / np.sum(count)
                            crpss[i] = Modulation_Index(count)
                        else:
                            crpss[i] = np.nan
                            # count_matrix[:,i] = np.nan
                            
                    ax.plot(starts, crpss, color = 'k')
                    ax.set_ylim(-0.005, 0.15)
                    ax.set_ylabel('MI CRPS')
            except:
                ax.plot()

class HeartResp_FreqRatio:
    name = 'HR_Ratio'

    def __init__(self, ecg_peaks, resp_cycles, step_secs = 60):
        self.ecg_peaks = ecg_peaks
        self.resp_cycles = resp_cycles
        self.step_secs = step_secs
        
    def plot(self, ax, t0, t1):
        
        ecg_peaks = self.ecg_peaks
        local_peaks = ecg_peaks[(ecg_peaks['peak_date'] > t0) & (ecg_peaks['peak_date'] < t1)]

        resp_cycles = self.resp_cycles
        local_cycles = resp_cycles[(resp_cycles['inspi_date'] > t0) & (resp_cycles['inspi_date'] < t1)]
        
        if not local_peaks.shape[0] == 0 and not local_cycles.shape[0] == 0:
            try:
                start_compute = np.datetime64(t0)
                stop_compute = np.datetime64(t1)
                starts = np.arange(start_compute, stop_compute, np.timedelta64(self.step_secs, "s"))
                stops = starts + np.timedelta64(self.step_secs, "s")
                ratios = np.zeros((starts.size))
                for i, start, stop in zip(range(starts.size), starts, stops):
                    if stop < stop_compute:
                        mask_ecg = (local_peaks['peak_date'] > start) & (local_peaks['peak_date'] <= stop)
                        mask_resp = (local_cycles['inspi_date'] > start) & (local_cycles['inspi_date'] <= stop)
                        freq_heart = np.median(1 / np.diff(local_peaks[mask_ecg]['peak_time'].values))
                        freq_resp = np.median(local_cycles['cycle_freq'].values)
                        ratios[i] = freq_heart / freq_resp
                    else:
                        ratios[i] = np.nan
                            
                # ax.plot(starts, ratios, color = 'k')
                # ax.set_ylim(3, 6)
                # ax.set_ylabel('HR Ratio')
                ax.plot(starts, np.abs(ratios - np.round(ratios)), color = 'k')
                ax.set_ylim(-0.01, 0.51)
                ax.set_ylabel('Mod HR Ratio')
            except:
                ax.plot()
            
class RSA:
    name = 'RSA'

    def __init__(self, rsa_cycles, win_cycles_smooth = 20, n_mads_cleaning = 5):
        self.rsa_cycles = rsa_cycles
        self.win_cycles_smooth = win_cycles_smooth
        self.n_mads_cleaning = n_mads_cleaning
        
    def plot(self, ax, t0, t1):
        
        rsa_cycles = self.rsa_cycles
        local_rsa_cycles = rsa_cycles[(rsa_cycles['cycle_date'] > t0) & (rsa_cycles['cycle_date'] < t1)]
        
        ymax = 20
        if not local_rsa_cycles.shape[0] == 0:
            try:
                by_cycle_rsa_amplitudes = local_rsa_cycles['decay_amplitude'].values
                # by_cycle_rsa_amplitudes = local_rsa_cycles['rising_amplitude'].values
                med, mad = physio.compute_median_mad(by_cycle_rsa_amplitudes)
                threshold = med + mad * self.n_mads_cleaning
                inds_bad_cycles = np.nonzero(by_cycle_rsa_amplitudes > threshold)[0]
                by_cycle_rsa_amplitudes[inds_bad_cycles] = med # replace bad cycles by median value
                
                if not self.win_cycles_smooth is None:
                    # plot_rsa = sliding_mean(by_cycle_rsa_amplitudes, self.win_cycles_smooth)
                    plot_rsa = pd.Series(by_cycle_rsa_amplitudes).rolling(self.win_cycles_smooth).mean().values
                else:
                    plot_rsa = by_cycle_rsa_amplitudes.copy()
                # plot_rsa = by_cycle_rsa_amplitudes.copy()
                plot_dates = local_rsa_cycles['cycle_date']
                ax.plot(plot_dates, plot_rsa, lw = 0.8, color = 'r')
                
                max_ = np.nanmax(plot_rsa)
                if max_ < 5:
                    ymax = 5
                elif max_ > 5 and max_ < 10:
                    ymax = 10
                else:
                    ymax = 20
            except:
                ax.plot()

        ax.set_ylabel(f'RSA\n[bpm]')
        ax.set_ylim(0, ymax)
        
class Spreading_depol_mono:
    name = 'Spreading_depol_mono'

    def __init__(self, stream, detections = None, down_sampling_factor = 2, lowcut_dc = 0.001, highcut_dc = 0.1, global_gain = 0.1):
        self.stream = stream
        self.down_sampling_factor = down_sampling_factor
        self.lowcut_dc = lowcut_dc
        self.highcut_dc = highcut_dc
        self.detections = detections
        self.global_gain = global_gain

    def plot(self, ax, t0, t1):
        chans = self.stream.channel_names
        srate = self.stream.sample_rate
        ecog_chans = [chan for chan in chans if 'ECoG' in chan]
        ecog_chan_inds = [chans.index(chan) for chan in ecog_chans]
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        times = times[::self.down_sampling_factor]
        new_srate = srate / self.down_sampling_factor
        
        raw_sigs = sigs[:,ecog_chan_inds]
        dcs = iirfilt(raw_sigs, srate, lowcut = self.lowcut_dc, highcut = self.highcut_dc, ftype = 'bessel', axis = 0)
        acs = iirfilt(raw_sigs, srate, lowcut = 0.5, highcut = 40, ftype = 'bessel', axis = 0)
        acs = acs - np.median(acs, axis = 1)[:,None]
        # acs = acs - np.mean(acs, axis = 1)[:,None]

        offsets_dc, gains_dc = physio.compute_median_mad(dcs, axis = 0)
        offsets_ac, gains_ac = physio.compute_median_mad(acs, axis = 0)
        
        offsets_dc_plot = np.arange(gains_dc.size) * np.max(gains_dc)
        dcs_plot = dcs * self.global_gain - offsets_dc[None,:] + offsets_dc_plot[None,:]
        
        acs_normalized = (acs - offsets_ac[None,:]) / gains_ac[None,:]
        acs_plot = acs_normalized * np.median(gains_dc) * self.global_gain + offsets_dc_plot[None,:]
        
        dcs_plot = scipy.signal.decimate(dcs_plot, q=self.down_sampling_factor, axis = 0)
        acs_plot = scipy.signal.decimate(acs_plot, q=self.down_sampling_factor, axis = 0)

        ax.plot(times, acs_plot, color = 'k', lw = 0.5, alpha = 0.9)
        ax.plot(times, dcs_plot, color = 'r', lw = 1)
        
        vmin = min([np.min(dcs_plot),np.min(acs_plot)])
        vmax = max([np.max(dcs_plot),np.max(acs_plot)])
        ax.set_ylim(vmin - vmin / 10, vmax + vmax / 10)
        ax.set_yticks(offsets_dc_plot, labels=ecog_chans)
        ax.set_ylabel(f'ECoG\n[µV]')
        
        detections = self.detections
        if not detections is None:
            detections = detections[detections['name'].apply(lambda x:'SD' in x)]
            detections = detections.sort_values(by = 'start_time')
            local_detections = detections[(detections['start_time'].values > np.datetime64(t0)) & (detections['start_time'].values < np.datetime64(t1))]
            if local_detections.shape[0] > 0:
                for i, row in local_detections.iterrows():
                    start = row['start_time']
                    duration = float(row['duration'])
                    stop = start + pd.Timedelta(duration, 's')
                ax.axvspan(start, stop, color = 'k', alpha = 0.05)
        
class Spreading_depol_bipol:
    name = 'ECoG Bipolar'

    def __init__(self, stream, detections = None, down_sampling_factor = 2, lowcut_dc = 0.001, highcut_dc = 0.1, global_gain = 0.5):
        self.stream = stream
        self.down_sampling_factor = down_sampling_factor
        self.lowcut_dc = lowcut_dc
        self.highcut_dc = highcut_dc
        self.detections = detections
        self.global_gain = global_gain

    def plot(self, ax, t0, t1):
        chans = self.stream.channel_names
        srate = self.stream.sample_rate
        ecog_chan_names = [chan for chan in chans if 'ECoG' in chan]
        sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        times = times[::self.down_sampling_factor]
        sigs = sigs / 1000 # µV to mV
        new_srate = self.stream.sample_rate / self.down_sampling_factor
        
        bipol_ecog_chan_names = [f'{ecog_chan_names[i]}-{ecog_chan_names[i-1]}' for i in np.arange(len(ecog_chan_names)-1,0,-1)]
        
        raw_sigs = np.zeros((sigs.shape[0], len(bipol_ecog_chan_names)))
        for i, bipol_chan in enumerate(bipol_ecog_chan_names):
            chan1, chan2 = bipol_chan.split('-')
            chan1_ind, chan2_ind = chans.index(chan1), chans.index(chan2)
            raw_sigs[:,i] =  sigs[:,chan1_ind] -  sigs[:,chan2_ind]
        
        dcs = iirfilt(raw_sigs, srate, lowcut = self.lowcut_dc, highcut = self.highcut_dc, ftype = 'bessel', axis = 0)
        acs = iirfilt(raw_sigs, srate, lowcut = 0.5, highcut = 40, ftype = 'bessel', axis = 0)
        
        offsets_dc, gains_dc = physio.compute_median_mad(dcs, axis = 0)
        offsets_ac, gains_ac = physio.compute_median_mad(acs, axis = 0)
        
        offsets_dc_plot = np.arange(gains_dc.size) * np.max(gains_dc)
        dcs_plot = dcs * self.global_gain - offsets_dc[None,:] + offsets_dc_plot[None,:]
        
        acs_normalized = (acs - offsets_ac[None,:]) / gains_ac[None,:]
        acs_plot = acs_normalized * np.median(gains_dc) * self.global_gain + offsets_dc_plot[None,:]
        
        dcs_plot = scipy.signal.decimate(dcs_plot, q=self.down_sampling_factor, axis = 0)
        acs_plot = scipy.signal.decimate(acs_plot, q=self.down_sampling_factor, axis = 0)

        ax.plot(times, acs_plot, color = 'k', lw = 0.5, alpha = 0.9)
        ax.plot(times, dcs_plot, color = 'r', lw = 1)
        
        vmin = min([np.min(dcs_plot),np.min(acs_plot)])
        vmax = max([np.max(dcs_plot),np.max(acs_plot)])
        ax.set_ylim(vmin - vmin / 10, vmax + vmax / 10)
        ax.set_yticks(offsets_dc_plot, labels=bipol_ecog_chan_names)
        ax.set_ylabel(f'ECoG Bipol\n[µV]')
        
        detections = self.detections
        if not detections is None:
            detections = detections[detections['name'].apply(lambda x:'SD' in x)]
            detections = detections.sort_values(by = 'start_time')
            local_detections = detections[(detections['start_time'].values > np.datetime64(t0)) & (detections['start_time'].values < np.datetime64(t1))]
            if local_detections.shape[0] > 0:
                for i, row in local_detections.iterrows():
                    start = row['start_time']
                    duration = float(row['duration'])
                    stop = start + pd.Timedelta(duration, 's')
                ax.axvspan(start, stop, color = 'k', alpha = 0.05)

class Spreading_depol_scalp:
    name = 'Spreading_depol_scalp'

    def __init__(self, stream, down_sampling_factor = 5):
        self.stream = stream
        self.down_sampling_factor = down_sampling_factor

    def plot(self, ax, t0, t1):
        chans = self.stream.channel_names
        srate = self.stream.sample_rate
        scalp_chans = [chan for chan in chans if not 'ECoG' in chan]
        raw_sigs, times = self.stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        times = times[::self.down_sampling_factor]
        new_srate = self.stream.sample_rate / self.down_sampling_factor
        
        dcs = iirfilt(raw_sigs, srate, lowcut = self.lowcut_dc, highcut = 0.1, ftype = 'bessel', axis = 0)
        acs = iirfilt(raw_sigs, srate, lowcut = 0.5, highcut = 40, ftype = 'bessel', axis = 0)
        
        offsets_dc, gains_dc = physio.compute_median_mad(dcs, axis = 0)
        offsets_ac, gains_ac = physio.compute_median_mad(acs, axis = 0)
        
        offsets_dc_plot = np.arange(gains_dc.size) * np.max(gains_dc)
        dcs_plot = dcs / 10 - offsets_dc[None,:] + offsets_dc_plot[None,:]
        
        acs_normalized = (acs - offsets_ac[None,:]) / gains_ac[None,:]
        acs_plot = acs_normalized * np.median(gains_dc) / 10 + offsets_dc_plot[None,:]
        
        dcs_plot = scipy.signal.decimate(dcs_plot, q=self.down_sampling_factor, axis = 0)
        acs_plot = scipy.signal.decimate(acs_plot, q=self.down_sampling_factor, axis = 0)

        ax.plot(times, acs_plot, color = 'k', lw = 0.5, alpha = 0.9)
        ax.plot(times, dcs_plot, color = 'r', lw = 1)
        
        vmin = min([np.min(dcs_plot),np.min(acs_plot)])
        vmax = max([np.max(dcs_plot),np.max(acs_plot)])
        ax.set_ylim(vmin - vmin / 10, vmax + vmax / 10)
        ax.set_yticks(offsets_dc_plot, labels=scalp_chans)
        ax.set_ylabel(f'ECoG Bipol\n[µV]')
        
        detections = self.detections
        if not detections is None:
            detections = detections[detections['name'].apply(lambda x:'SD' in x)]
            detections = detections.sort_values(by = 'start_time')
            local_detections = detections[(detections['start_time'].values > np.datetime64(t0)) & (detections['start_time'].values < np.datetime64(t1))]
            if local_detections.shape[0] > 0:
                for i, row in local_detections.iterrows():
                    start = row['start_time']
                    duration = float(row['duration'])
                    stop = start + pd.Timedelta(duration, 's')
                ax.axvspan(start, stop, color = 'k', alpha = 0.05)
        
class Pulse_Pressure:
    name = 'Pulse_Pressure'

    def __init__(self, abp_features, n_mads_cleaning=5, n_mads_ylim=4):
        self.abp_features = abp_features
        self.n_mads_cleaning = n_mads_cleaning
        self.n_mads_ylim = n_mads_ylim

    def plot(self, ax, t0, t1):
        abp = self.abp_features
        local_abp = abp[(abp['peak_date'] >=t0) & (abp['peak_date'] <t1)]
        pulse_pressure = local_abp['rise_amplitude'].values
        med, mad = physio.compute_median_mad(pulse_pressure)
        threshold = med + mad * self.n_mads_cleaning
        inds_bad = np.nonzero(pulse_pressure > threshold)[0]
        pulse_pressure[inds_bad] = med # replace bad cycles by median value
        ax.plot(local_abp['peak_date'], pulse_pressure, color = 'g')
        ax.set_ylabel('Pulse Pressure')
        # s = np.std(pulse_pressure)
        # ax.set_ylim(np.min(pulse_pressure) - s, np.max(pulse_pressure) + s)
        med, mad = physio.compute_median_mad(pulse_pressure)
        ax.set_ylim(med - mad * self.n_mads_ylim, med + mad * self.n_mads_ylim)
      
class Traube_Herring:
    name = 'Traube_Herring'

    def __init__(self, abp, co2, srate, dates, wsize = 50):
        self.abp = abp
        self.co2 = co2
        self.srate = srate
        self.dates = dates
        self.wsize = wsize

    def plot(self, ax, t0, t1):
        dates = self.dates
        raw_abp = self.abp
        raw_co2 = self.co2
        srate = self.srate
        
        local_mask = (dates >= t0) & (dates < t1)
        local_abp = raw_abp[local_mask]
        local_co2 = raw_co2[local_mask]
        
        nperseg = int(self.wsize * srate)
        freqs, times_spectrum_s, Sxx_abp = scipy.signal.spectrogram(local_abp, fs = srate, nperseg =  nperseg)
        freqs, times_spectrum_s, Sxx_co2 = scipy.signal.spectrogram(local_co2, fs = srate, nperseg =  nperseg)
        dates_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + t0
        mask_f = (freqs > 0.05) & (freqs < 0.6)
        freqs = freqs[mask_f]
        Sxx_abp = Sxx_abp[mask_f,:]
        Sxx_co2 = Sxx_co2[mask_f,:]
        
        inds_where_traube_is = np.argmax(Sxx_co2, axis = 0)
        # fmax_freqs = np.apply_along_axis(lambda i:Sxx_abp[i], axis = 0, arr = inds_where_traube_is)
        traube = np.zeros(inds_where_traube_is.shape)
        for i, ind in enumerate(inds_where_traube_is):
            traube[i] = Sxx_abp[ind,i]
        ax.plot(dates_spectrum, traube, color = 'm')
        s = np.std(traube)
        ax.set_ylim(0, np.max(traube) + s)
        ax.set_ylabel('Traube-Herring')
        
class Wavelet_Power:
    name = 'Wavelet_Power'

    def __init__(self, stream, chan,  f_start, f_stop, n_steps, n_cycles, amplitude_exponent=2, quantile_saturation = 0.01, scaling = None, compute_margin_secs = 10, down_samp_plot = 3):
        self.stream = stream
        self.chan_name = chan
        self.chan_ind = stream.channel_names.index(chan)
        self.f_start = f_start
        self.f_stop = f_stop
        self.n_steps = n_steps
        self.n_cycles = n_cycles
        self.amplitude_exponent = amplitude_exponent
        self.quantile_saturation = quantile_saturation
        self.scaling = scaling
        self.compute_margin_secs = compute_margin_secs
        self.down_samp_plot = down_samp_plot

    def plot(self, ax, t0, t1):
        t0_with_margin = np.datetime64(t0) - np.timedelta64(self.compute_margin_secs , 's')
        t1_with_margin = np.datetime64(t1) + np.timedelta64(self.compute_margin_secs , 's')
        sigs, dates_with_margin = self.stream.get_data(sel=slice(t0_with_margin, t1_with_margin), with_times=True,
                                          apply_gain=True)
        srate = self.stream.sample_rate
        down_samp_compute = int(srate / (self.f_stop * 4))
        down_samp_compute = 1 if down_samp_compute < 1 else down_samp_compute
        sig_with_margin = sigs[:, self.chan_ind]
        sig_with_margin = scipy.signal.decimate(sig_with_margin, q = down_samp_compute)
        dates_with_margin = dates_with_margin[::down_samp_compute]
        new_srate = srate / down_samp_compute
        f, power_with_margin = morlet_power(sig_with_margin, new_srate, self.f_start, self.f_stop, self.n_steps, self.n_cycles, self.amplitude_exponent)
        mask_dates_without_margin = (dates_with_margin >= np.datetime64(t0)) & (dates_with_margin < np.datetime64(t1))
        dates_without_margin = dates_with_margin[mask_dates_without_margin]
        power_without_margin = power_with_margin[:,mask_dates_without_margin]
        down_samp_plot = self.down_samp_plot
        power_plot = scipy.signal.decimate(power_without_margin, q = down_samp_plot, axis = 1)
        dates_plot = dates_without_margin[::down_samp_plot]
        srate_plot = new_srate / down_samp_plot
        if self.scaling == 'fit':
            f_log = np.log(f)
            spectrum = np.mean(power_plot, axis = 1)
            spectrum_log = np.log(spectrum)
            res = scipy.stats.linregress(f_log, spectrum_log)
            a = res.slope
            b = res.intercept
            fit_log = a * f_log + b
            fit = np.exp(a * f_log + b)
            power_plot = power_plot - fit[:,None]
        elif self.scaling == 'baseline':
            baseline_duration_secs = 0.5
            spectrum_baseline = np.median(power_plot[:,:int(srate_plot * baseline_duration_secs)], axis = 1)
            power_plot = power_plot - spectrum_baseline[:,None]            
        vmin = np.quantile(power_plot, self.quantile_saturation)
        vmax = np.quantile(power_plot, 1-self.quantile_saturation)
        ax.pcolormesh(dates_plot, f, power_plot, vmin = vmin, vmax=vmax)
        ax.set_ylabel(f'{self.chan_name}\nPower') 


class Wavelet_Power_bio:
    name = 'Wavelet_Power_bio'

    def __init__(self, stream,  f_start, f_stop, n_steps, n_cycles, amplitude_exponent=2, quantile_saturation = 0.01, scaling = None, compute_margin_secs = 10, down_samp_plot = 3):
        self.stream = stream
        self.f_start = f_start
        self.f_stop = f_stop
        self.n_steps = n_steps
        self.n_cycles = n_cycles
        self.amplitude_exponent = amplitude_exponent
        self.quantile_saturation = quantile_saturation
        self.scaling = scaling
        self.compute_margin_secs = compute_margin_secs
        self.down_samp_plot = down_samp_plot

    def plot(self, ax, t0, t1):
        t0_with_margin = np.datetime64(t0) - np.timedelta64(self.compute_margin_secs , 's')
        t1_with_margin = np.datetime64(t1) + np.timedelta64(self.compute_margin_secs , 's')
        sig_with_margin, dates_with_margin = self.stream.get_data(sel=slice(t0_with_margin, t1_with_margin), with_times=True,
                                          apply_gain=True)
        srate = self.stream.sample_rate
        down_samp_compute = int(srate / (self.f_stop * 4))
        down_samp_compute = 1 if down_samp_compute < 1 else down_samp_compute
        sig_with_margin = scipy.signal.decimate(sig_with_margin, q = down_samp_compute)
        dates_with_margin = dates_with_margin[::down_samp_compute]
        new_srate = srate / down_samp_compute
        f, power_with_margin = morlet_power(sig_with_margin, new_srate, self.f_start, self.f_stop, self.n_steps, self.n_cycles, self.amplitude_exponent)
        mask_dates_without_margin = (dates_with_margin >= np.datetime64(t0)) & (dates_with_margin < np.datetime64(t1))
        dates_without_margin = dates_with_margin[mask_dates_without_margin]
        power_without_margin = power_with_margin[:,mask_dates_without_margin]
        down_samp_plot = self.down_samp_plot
        power_plot = scipy.signal.decimate(power_without_margin, q = down_samp_plot, axis = 1)
        dates_plot = dates_without_margin[::down_samp_plot]
        srate_plot = new_srate / down_samp_plot
        if self.scaling == 'fit':
            f_log = np.log(f)
            spectrum = np.mean(power_plot, axis = 1)
            spectrum_log = np.log(spectrum)
            res = scipy.stats.linregress(f_log, spectrum_log)
            a = res.slope
            b = res.intercept
            fit_log = a * f_log + b
            fit = np.exp(a * f_log + b)
            power_plot = power_plot - fit[:,None]
        elif self.scaling == 'baseline':
            baseline_duration_secs = 0.5
            spectrum_baseline = np.median(power_plot[:,:int(srate_plot * baseline_duration_secs)], axis = 1)
            power_plot = power_plot - spectrum_baseline[:,None]            
        vmin = np.quantile(power_plot, self.quantile_saturation)
        vmax = np.quantile(power_plot, 1-self.quantile_saturation)
        ax.pcolormesh(dates_plot, f, power_plot, vmin = vmin, vmax=vmax)
        ax.set_ylabel(f'Freq (Hz)')  
        
        
class SD_Detection:
    name = 'SD_Detection'

    def __init__(self, stream, chan, down_sampling_factor = 100, lowcut = 0.001, highcut = 0.02, threshold_dc = 10, threshold_ac = 0.5):
        self.stream = stream
        self.chan_name = chan
        self.chan_ind = stream.channel_names.index(chan)
        self.down_sampling_factor = down_sampling_factor
        self.lowcut = lowcut
        self.highcut = highcut
        self.threshold_dc = threshold_dc
        self.threshold_ac = threshold_ac

    def plot(self, ax, t0, t1):        
        # MODE BOOLEAN
        
        stream = self.stream
        chan_names = stream.channel_names
        ecog_chan_names = [chan for chan in chan_names if 'ECoG' in chan] 
        ecog_chan_inds = [chan_names.index(chan) for chan in ecog_chan_names] 
        sigs, dates = stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        srate = stream.sample_rate
        
        sigs = sigs[:,ecog_chan_inds]
        sigs = sigs / 1000 # µV to mV
        if not self.lowcut is None or not self.highcut is None:
            dcs = iirfilt(sigs, srate, lowcut = self.lowcut, highcut = self.highcut, ftype = 'bessel', axis = 0)
        
        acs = iirfilt(sigs, srate, lowcut = 0.5, highcut = 40, ftype = 'bessel', axis = 0)
        acs = get_amp(acs, axis = 0)
        acs = iirfilt(acs, srate, lowcut = None, highcut = 0.01, ftype = 'bessel', axis = 0)

        acs = acs[::self.down_sampling_factor,:]
        
        dcs = get_amp(dcs, axis = 0)
        
        dcs = dcs[::self.down_sampling_factor,:]
        dates = dates[::self.down_sampling_factor]        
        
        bool_dc = dcs > self.threshold_dc
        bool_ac = acs < self.threshold_ac
        
        bool_concat = np.zeros(shape = (2, bool_dc.shape[0], bool_dc.shape[1]), dtype = 'bool') # dc/ac * time * chan
        bool_concat[0,:,:] = bool_dc
        bool_concat[1,:,:] = bool_ac     
        print(bool_concat.shape)
        
        # timecourse_sd_chans = np.sum(bool_concat, axis = 1)
        # bool_timecourse_sd_chans = (timecourse_sd_chans >= 2) & (timecourse_sd_chans <= 5)
        
        # ax.plot(dates, timecourse_sd_chans, color='k', lw = 0.8)
        # ax.set_ylim(-1, len(ecog_chan_inds) + 1)
        
        bool_2d = np.all(bool_concat, axis = 0)
        ax.pcolormesh(dates, ecog_chan_names, bool_2d.T, vmin = 0, vmax = 1)
        # ax.set_ylim(-1, len(ecog_chan_inds) + 1)
        
        # ax.plot(dates, bool_timecourse_sd_chans, color='k', lw = 0.8)
        # ax.set_ylim(-0.1, 1.5)
        ax.set_ylabel('CSD or not')
        
        # MODE AMPLITUDE
        
#         stream = self.stream
#         srate = stream.sample_rate
#         sigs, dates = stream.get_data(sel=slice(t0, t1), with_times=True,
#                                           apply_gain=True)           
#         sig = sigs[:, self.chan_ind]
#         sig = sig / 1000 # µV to mV
#         unit = 'mV'

#         if not self.lowcut is None or not self.highcut is None:
#             sig = iirfilt(sig, srate, lowcut = self.lowcut, highcut = self.highcut, ftype = 'bessel')
        
#         sig = sig[::self.down_sampling_factor]
#         dates = dates[::self.down_sampling_factor]
#         sig = get_amp(sig)
#         # med, mad = physio.compute_median_mad(sig)
#         # thresh = 10 # mV
#         ax.plot(dates, sig, color='k', lw = 0.8)
#         # ax.axhline(thresh, color = 'r')
#         ax.set_ylim(sig.min(), sig.max())
#         ax.set_ylabel(f'{self.chan_name}\n[{unit}]')
        
        # MODE AMPLITUDES
        
#         stream = self.stream
#         chan_names = stream.channel_names
#         ecog_chan_names = [chan for chan in chan_names if 'ECoG' in chan]
#         ecog_chan_inds = [chan_names.index(chan) for chan in ecog_chan_names]
#         sigs, dates = stream.get_data(sel=slice(t0, t1), with_times=True,
#                                           apply_gain=True)
#         srate = stream.sample_rate
        
#         sigs = sigs[:,ecog_chan_inds]
        
#         bipol_ecog_chan_names = [f'{ecog_chan_names[i]}-{ecog_chan_names[i-1]}' for i in np.arange(len(ecog_chan_names)-1,0,-1)]
        
#         sigs_bipol = np.zeros((sigs.shape[0], len(bipol_ecog_chan_names)))
#         for i, bipol_chan in enumerate(bipol_ecog_chan_names):
#             chan1, chan2 = bipol_chan.split('-')
#             chan1_ind, chan2_ind = ecog_chan_names.index(chan1), ecog_chan_names.index(chan2)
#             sigs_bipol[:,i] =  sigs[:,chan1_ind] -  sigs[:,chan2_ind]
        
#         sigs = sigs_bipol.copy()
#         del sigs_bipol
        
#         if not self.lowcut is None or not self.highcut is None:
#             sigs = iirfilt(sigs, srate, lowcut = self.lowcut, highcut = self.highcut, ftype = 'bessel', axis = 0) 
#         sigs = sigs / 1000 # µV to mV
#         unit = 'mV'
        
#         sigs = sigs[::self.down_sampling_factor,:]
#         dates = dates[::self.down_sampling_factor]
#         sig_amps = get_amp(sigs, axis = 0)
        
#         mcolors = get_mcolors()
#         # for ind, chan in enumerate(ecog_chan_names):
#         for ind, chan in enumerate(bipol_ecog_chan_names):
#             ax.plot(dates, sig_amps[:,ind], label = chan, color = mcolors[ind])
#         ax.legend(fontsize = 6)
#         ax.set_ylim(sig_amps.min(), sig_amps.max())
        # sum_amps = sig_amps.sum(axis = 1)
        # med, mad = physio.compute_median_mad(sum_amps)
        # thresh = med + mad * 3
        # ax.plot(dates, sum_amps, color = 'k')
        # ax.axhline(thresh, color = 'r')
        # ax.set_ylim(sum_amps.min(), sum_amps.max())
    
class Cereral_Perfusion_Pressure:
    name = 'Cereral_Perfusion_Pressure'

    def __init__(self, icp_mean_stream, abp_mean_stream, down_sampling_factor=10):
        self.icp_mean_stream = icp_mean_stream
        self.abp_mean_stream = abp_mean_stream
        self.down_sampling_factor = down_sampling_factor
        
    def plot(self, ax, t0, t1):

        icp, dates_icp = self.icp_mean_stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        
        abp, dates_abp = self.abp_mean_stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        
        ymax = 170
        if icp.size == abp.size:
                
            cpp = abp - icp

            ax.plot(dates_abp, cpp, color='k', lw = 1)
            ax.set_ylabel(f'CPP (mmHg)')
            ax.set_ylim(30, ymax)
            
        else:
            ax.plot()
            # ax.text(x = dates_abp[dates_abp.size // 2], y = 100, s = 'ICP or ABP do not have the same shape', ha = 'center')
            ax.set_ylabel(f'CPP (mmHg)')
            ax.set_ylim(30, ymax)
            
        # ax.axhspan(0,50, color = 'r', alpha = 0.1)
        # ax.axhspan(50,150, color = 'tab:blue', alpha = 0.1)
        # ax.axhspan(150,ymax, color = 'r', alpha = 0.1)
        
        ax.axhline(50, color = 'r')
        ax.axhline(150, color = 'r')
        
class PRx:
    name = 'PRx'

    def __init__(self, prx, prx_dates, wsize_smooth = None):
        self.prx = prx
        self.prx_dates = prx_dates
        self.wsize_smooth = wsize_smooth
        
    def plot(self, ax, t0, t1):
        prx = self.prx
        prx_dates = self.prx_dates
        if not self.wsize_smooth is None:
            moving_corr = pd.Series(moving_corr).rolling(window=self.wsize_smooth).mean().values
            
        local_mask = (prx_dates > np.datetime64(t0)) & (prx_dates <= np.datetime64(t1))
        
        local_dates = prx_dates[local_mask]
        local_prx = prx[local_mask]
        
        ax.plot(local_dates, local_prx, color = 'k')
        ax.set_ylim(-1.05, 1.05)

class PRx_homemade:
    name = 'PRx_homemade'

    def __init__(self, prx_da):
        self.prx_da = prx_da
        
    def plot(self, ax, t0, t1):
        prx = self.prx_da.values
        prx_dates = self.prx_da['date'].values

        local_mask = (prx_dates > np.datetime64(t0)) & (prx_dates <= np.datetime64(t1))
        
        local_dates = prx_dates[local_mask]
        local_prx = prx[local_mask]
        
        ax.plot(local_dates, local_prx, color = 'k')
        ax.set_ylim(-1.05, 1.05)
        
        
class Heart_Resp_in_ICP:
    name = 'Heart_Resp_in_ICP'

    def __init__(self, icp_stream, wsize_secs = 30, resp_band = (0.1, 0.5), heart_band = (0.8, 2.5)):
        self.icp_stream = icp_stream
        self.wsize_secs = wsize_secs
        self.resp_band = resp_band
        self.heart_band = heart_band

    def plot(self, ax, t0, t1):
        icp_stream = self.icp_stream 
        raw_icp, dates = icp_stream.get_data(sel = slice(t0, t1), with_times = True, apply_gain = True)
        srate = icp_stream.sample_rate
        
        try:
            freqs, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(raw_icp, fs = srate, nperseg =  int(self.wsize_secs * srate), scaling = 'spectrum')
            dates_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + t0

            max_amplitudes = []
            for band, color in zip([self.resp_band, self.heart_band],['tab:blue','r']):
                mask_f = (freqs > band[0]) & (freqs <= band[1])
                Sxx_icp_band = Sxx_icp[mask_f,:]
                max_amplitude = np.sqrt(np.max(Sxx_icp_band, axis = 0))
                max_amplitudes.append(max_amplitude)
                ax.plot(dates_spectrum, max_amplitude, color = color)
            max_ = max(np.concatenate(max_amplitudes))
            ax2 = ax.twinx()
            ax2.plot(dates_spectrum, max_amplitudes[1] / max_amplitudes[0], color = 'k')
            ax2.set_yticks([])
            # ax2.set_ylabel('H/R')
            ax.set_ylabel('Resp (blue)\nHeart (red)\nin ICP (mmHg)')
            ax.set_ylim(0, max_ + max_/10)
        except:
            ax.plot()
            ax.set_ylabel('Resp (blue)\nHeart (red)\nin ICP (mmHg)')

class Heart_Resp_in_ABP:
    name = 'Heart_Resp_in_ABP'

    def __init__(self, abp_stream, wsize_secs = 30, resp_band = (0.1, 0.5), heart_band = (0.8, 2.5)):
        self.abp_stream = abp_stream
        self.wsize_secs = wsize_secs
        self.resp_band = resp_band
        self.heart_band = heart_band

    def plot(self, ax, t0, t1):
        abp_stream = self.abp_stream 
        raw_abp, dates = abp_stream.get_data(sel = slice(t0, t1), with_times = True, apply_gain = True)
        srate = abp_stream.sample_rate
        
        try:
            freqs, times_spectrum_s, Sxx_abp = scipy.signal.spectrogram(raw_abp, fs = srate, nperseg =  int(self.wsize_secs * srate), scaling = 'spectrum')
            dates_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + t0

            max_amplitudes = []
            for band, color in zip([self.resp_band, self.heart_band],['tab:blue','r']):
                mask_f = (freqs > band[0]) & (freqs <= band[1])
                Sxx_abp_band = Sxx_abp[mask_f,:]
                max_amplitude = np.sqrt(np.max(Sxx_abp_band, axis = 0))
                max_amplitudes.append(max_amplitude)
                ax.plot(dates_spectrum, max_amplitude, color = color)
            max_ = max(np.concatenate(max_amplitudes))
            ax2 = ax.twinx()
            ax2.plot(dates_spectrum, max_amplitudes[1] / max_amplitudes[0], color = 'k')
            ax2.set_yticks([])
            # ax2.set_ylabel('H/R')
            ax.set_ylabel('Resp (blue)\nHeart (red)\nin ABP (mmHg)')
            ax.set_ylim(0, max_ + max_/10)
        except:
            ax.plot()
            ax.set_ylabel('Resp (blue)\nHeart (red)\nin ABP (mmHg)')

class Heart_Resp_freq_from_ICP:
    name = 'Heart_Resp_freq_from_ICP'

    def __init__(self, icp_stream, wsize_secs = 30, resp_band = (0.1, 0.5), heart_band = (0.8, 2.5)):
        self.icp_stream = icp_stream
        self.wsize_secs = wsize_secs
        self.resp_band = resp_band
        self.heart_band = heart_band

    def plot(self, ax, t0, t1):
        icp_stream = self.icp_stream 
        raw_icp, dates = icp_stream.get_data(sel = slice(t0, t1), with_times = True, apply_gain = True)
        srate = icp_stream.sample_rate
        
        try:
            freqs, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(raw_icp, fs = srate, nperseg =  int(self.wsize_secs * srate), scaling = 'spectrum')
            dates_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + t0

            for band, color in zip([self.resp_band, self.heart_band],['tab:blue','r']):
                mask_f = (freqs > band[0]) & (freqs <= band[1])
                Sxx_icp_band = Sxx_icp[mask_f,:]
                freqs_masked = freqs[mask_f]
                instantaneous_freq = get_crest_line(freqs_masked, Sxx_icp_band)
                ax.plot(dates_spectrum, instantaneous_freq, color = color)
            ax.set_ylabel('Resp (blue)\nHeart (red)\nHz')
            ax.set_ylim(0, 2.2)
        except:
            ax.plot()
            ax.set_ylabel('Resp (blue)\nHeart (red)\nHz')
        
        
class Muscle_artifacts:
    name = 'Muscle_artifacts'
    
    """
    MNE: https://mne.tools/dev/auto_examples/preprocessing/muscle_detection.html
    This example uses annotate_muscle_zscore() to annotate segments where muscle activity is likely present. This is done by band-pass filtering the data in the 110-140 Hz range. Then, the envelope is taken using the hilbert analytical signal to only consider the absolute amplitude and not the phase of the high frequency signal. The envelope is z-scored and summed across channels and divided by the square root of the number of channels. Because muscle artifacts last several hundred milliseconds, a low-pass filter is applied on the averaged z-scores at 4 Hz, to remove transient peaks. Segments above a set threshold are annotated as BAD_muscle.
    """

    def __init__(self, eeg_stream, used_chans = None, lowcut = 30, highcut_zscore = 1):
        self.stream = eeg_stream
        self.used_chans = used_chans
        self.lowcut = lowcut
        self.highcut_zscore = highcut_zscore

    def plot(self, ax, t0, t1):
        used_chans = self.used_chans
        srate = self.stream.sample_rate
        raw_sigs, dates = self.stream.get_data(sel=slice(t0, t1), with_times=True, apply_gain=True)
        chans = self.stream.channel_names
        if not used_chans is None:
            chan_inds = [chans.index(chan) for chan in used_chans]
            raw_sigs = raw_sigs[:,chan_inds]
            
        sigs_filtered = iirfilt(raw_sigs, srate, lowcut = self.lowcut, ftype = 'butter', axis = 0)
        sigs_amplitude = get_amp(sigs_filtered, axis = 0)
        amplitudes_zscored = (sigs_amplitude - np.mean(sigs_amplitude, axis = 0)[None,:]) / np.std(sigs_amplitude, axis = 0)[None,:]
        zscores_filtered = iirfilt(amplitudes_zscored, srate, lowcut = None, highcut = self.highcut_zscore, ftype = 'butter', axis = 0)
        zscores_sqrt_averaged = np.sum(zscores_filtered, axis = 1) / np.sqrt(zscores_filtered.shape[1])
        
        ax.plot(dates, zscores_sqrt_averaged, color = 'm', lw = 1)
        ax.set_ylim(np.min(zscores_sqrt_averaged), np.max(zscores_sqrt_averaged))
        ax.set_ylabel('Muscle_artifacts')
        
class Muscle_artifacts_precompute:
    name = 'Muscle_artifacts_precompute'
    
    """
    MNE: https://mne.tools/dev/auto_examples/preprocessing/muscle_detection.html
    This example uses annotate_muscle_zscore() to annotate segments where muscle activity is likely present. This is done by band-pass filtering the data in the 110-140 Hz range. Then, the envelope is taken using the hilbert analytical signal to only consider the absolute amplitude and not the phase of the high frequency signal. The envelope is z-scored and summed across channels and divided by the square root of the number of channels. Because muscle artifacts last several hundred milliseconds, a low-pass filter is applied on the averaged z-scores at 4 Hz, to remove transient peaks. Segments above a set threshold are annotated as BAD_muscle.
    """

    def __init__(self, precomputed_muscle_artifacts, dates, threshold, ylim, mode = 'real'):
        self.precomputed_muscle_artifacts = precomputed_muscle_artifacts
        self.dates = dates
        self.threshold = threshold
        self.ylim = ylim
        self.mode = mode

    def plot(self, ax, t0, t1):
        all_sig = self.precomputed_muscle_artifacts
        dates = self.dates
        mask = (dates >= t0) & (dates < t1)
        dates = dates[mask]
        sig_plot = all_sig[mask]
        
        if self.mode == 'real':
            ax.plot(dates, sig_plot, color = 'm', lw = 1)
            ax.set_ylim(self.ylim[0], self.ylim[1])
            # ax.set_ylim(sig_plot.min(), sig_plot.max())
            ax.axhline(self.threshold, color = 'r')
        elif self.mode == 'bool':
            sig_bool = sig_plot > self.threshold
            ax.plot(dates, sig_bool, color = 'm', lw = 1)
            ax.set_ylim(-0.5, 1.5)
        ax.set_ylabel('Muscle_artifacts\nprecompute')
        
class PRX_Processed:
    name = 'PRX_Processed'
    
    def __init__(self, prx_processed_stream):
        self.prx_processed_stream = prx_processed_stream

    def plot(self, ax, t0, t1):
        prx_processed_stream = self.prx_processed_stream
        sig, dates = prx_processed_stream.get_data(sel = slice(t0,t1), with_times=True, apply_gain=True)
        ax.plot(dates, sig, color = 'm', lw = 1)
        ax.set_ylim(-1,1)
        
        
class Artifact_6Hz:
    name = 'Artifact_6Hz'
    def __init__(self, stream, chan_name, target_frequency = 6, n_cycles = 20, down_sampling_factor = 10):
        self.stream = stream
        self.chan_name = chan_name
        self.target_frequency = target_frequency
        self.n_cycles = n_cycles
        self.down_sampling_factor = down_sampling_factor
        
    def plot(self, ax, t0, t1):
        stream = self.stream
        srate = stream.sample_rate
        sigs, times = stream.get_data(sel=slice(t0, t1), with_times=True,
                                          apply_gain=True)
        chan_name = self.chan_name
        if not '-' in chan_name: # monopolar case
            sig = sigs[:,stream.channel_names.index(chan_name)]
        else:
            chan_name1, chan_name2 = chan_name.split('-')
            sig = sigs[:,stream.channel_names.index(chan_name1)] - sigs[:,stream.channel_names.index(chan_name2)]
        
        wavelet = complex_mw(np.arange(-5,5,1/srate), n_cycles = self.n_cycles, freq = self.target_frequency)
        amplitude = np.abs(signal.fftconvolve(sig, wavelet, mode = 'same', axes = 0))

        amplitude = amplitude[::self.down_sampling_factor]
        amplitude = np.log(amplitude)
        med, mad = physio.compute_median_mad(amplitude)
        times = times[::self.down_sampling_factor]
        ax.plot(times, amplitude, color='k', lw = 0.5)
        # ax.set_ylim(amplitude.min(), amplitude.max())
        ax.set_ylim(0, 20)
        ax.axhline(med + mad * 5, color = 'r')
        ax.set_ylabel(f'6 Hz Amp\n{self.chan_name}\n[mV]')
        
class PSI:
    name = 'PSI'

    def __init__(self, psi_da):
        self.psi_da = psi_da
        
    def plot(self, ax, t0, t1):
        psi_da = self.psi_da
        psi_dates = self.psi_da['date'].values

        local_mask = (psi_dates > np.datetime64(t0)) & (psi_dates <= np.datetime64(t1))
        
        local_dates = psi_dates[local_mask]
        local_psi = psi_da.values[local_mask]
        
        ax.plot(local_dates, local_psi, color = 'k')
        ax.set_ylim(0.5, 4.5)

        
        
        
        
  