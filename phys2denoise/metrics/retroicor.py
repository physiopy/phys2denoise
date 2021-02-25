"""implementation of retroicor"""

import argparse as ap
import numpy as np
import scipy as sp
import json
import string
import random
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

"""
This function computes RETROICOR regressors (Glover et al. 2000) 
"""


def compute_card_phase(card_peaks_timings,slice_timings,nscans,TR):
    """

    This function creates cardiac phase from cardiac peaks.
    Assumes that timing of cardiac events are given in same units
    as slice timings, for example seconds.
    
    """

    nscans = np.shape(slice_timings)
    phase_card = np.zeros(nscans)
    for ii in range(nscans):
        # find previous cardiac peaks
        previous_card_peaks = np.asarray(np.nonzero(card_peaks_timings < slice_timings[ii]))
        if np.size(previous_card_peaks) == 0:
            t1 = 0
        else:
            last_peak = previous_card_peaks[0][-1]
            t1 = card_peaks_timings[last_peak]
        
        # find posterior cardiac peaks
        next_card_peaks = np.asarray(np.nonzero(card_peaks_timings > slice_timings[ii]))
        if np.size(next_card_peaks) == 0:
            t2 = nscans * TR
        else:
            next_peak = next_card_peaks[0][0]
            t2 = card_peaks_timings[next_peak]
        
        # compute cardiac phase
        phase_card[ii] = (2*np.math.pi*(slice_timings[ii] - t1))/(t2-t1)

    return phase_card

def compute_resp_phase(resp,sampling_time):

    """
    This function creates respiration phase from resp trace. 
    """


def compute_retroicor_regressors(physio,TR,nscans,slice_timings,n_harmonics,card=FALSE,resp=FALSE):
    nslices = np.shape(slice_timings) # number of slices

    # if respiration, compute histogram and temporal derivative of respiration signal
    if resp:
        resp_hist, resp_hist_bins = plt.hist(physio,bins=100)
        resp_diff = np.diff(physio,n=1)
    
    #initialize output variables
    retroicor_regressors = []
    phase = np.empty((nscans,nslices))

    for jj in range(nslices):
        # Initialize slice timings for current slice
        crslice_timings = TR * np.arange(nscans)+slice_timings[jj]
        # Compute physiological phases using the timings of physio events (e.g. peaks) slice sampling times
        if card:
            phase[,jj] = compute_phase_card(physio,crslice_timings)
        if resp:
            phase[,jj] = compute_phase_resp(resp_diff,resp_hist,resp_hist_bins,crslice_timings)
        # Compute retroicor regressors
        for nn in range(n_harmonics):
            retroicor_regressors[jj][:,2*nn] = np.cos((nn+1)*phase[jj])
            retricor_regressor[jj][:,2*nn+1] = np.sin((nn+1)*phase[jj])

    return retroicor_regressors,phase


        

