import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from datetime import datetime, timedelta
import seaborn as sns
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset


###################################### load miniseed data #########################
def load_miniseed(file_path):
    st = read(file_path)
    return st

###################################### load csv data #########################
def read_catalog(file_path):
    return pd.read_csv(file_path)

###################################### STA/LTA #########################
#filter trace
def filter_trace(st, min_freq, max_freq):
    # Set the minimum frequency
    minfreq = 0.5
    maxfreq = 1.0

    # Going to create a separate trace for the filter data
    st_filt = st.copy()
    st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data
    f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)
    return f, t, sxx, tr_times_filt, tr_data_filt

#sta/lta
def sta_lta(st):
    tr = st.traces[0].copy()
    tr_times = tr.times()
    tr_data = tr.data
    # Sampling frequency of our trace
    df = tr.stats.sampling_rate

    # How long should the short-term and long-term window be, in seconds?
    sta_len = 120
    lta_len = 600

    # Run Obspy's STA/LTA to obtain a characteristic function
    # This function basically calculates the ratio of amplitude between the short-term 
    # and long-term windows, moving consecutively in time across the data
    cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))
    return cft, tr_times, tr_data

###################################### plots #########################
#plot time series and spectogram
def spectogram(f, t, sxx): #output of function 'filter_trace'
    # Plot the time series and spectrogram
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 1, 1)
    # Plot trace
    ax.plot(tr_times_filt,tr_data_filt)

    # Mark detection
    ax.axvline(x = arrival, color='red',label='Detection')
    ax.legend(loc='upper left')

    # Make the plot pretty
    ax.set_xlim([min(tr_times_filt),max(tr_times_filt)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')

    ax2 = plt.subplot(2, 1, 2)
    vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
    ax2.set_xlim([min(tr_times_filt),max(tr_times_filt)])
    ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax2.axvline(x=arrival, c='red')
    cbar = plt.colorbar(vals, orientation='horizontal')
    cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

#plot characteristics function : sta/lta
def plot_char_func(tr_times, cft): #output of sta_lta function
    # Plot characteristic function
    fig,ax = plt.subplots(1,1,figsize=(12,3))
    ax.plot(tr_times,cft)
    ax.set_xlim([min(tr_times),max(tr_times)])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Characteristic function')
    
    
def plot_on_off_triggers(cft, tr_times, tr_data, thr_on=4, thr_off=1.5): # cft, tr_times, tr_data are output of sta_lta function 
    #you can change the on and off thresholds
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))
    # The first column contains the indices where the trigger is turned "on". 
    # The second column contains the indices where the trigger is turned "off".

    # Plot on and off triggers
    fig,ax = plt.subplots(1,1,figsize=(12,3))
    for i in np.arange(0,len(on_off)):
        triggers = on_off[i]

        ax.axvline(x = tr_times[triggers[0]], color='red', label='Trig. On')
        ax.axvline(x = tr_times[triggers[1]], color='purple', label='Trig. Off')

    # Plot seismogram
    ax.plot(tr_times,tr_data)
    ax.set_xlim([min(tr_times),max(tr_times)])
    ax.legend()
    return fig
    
