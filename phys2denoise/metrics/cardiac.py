"""Denoising metrics for cardio recordings."""
import numpy as np
from scipy import signal


def cpm(Fs, PPGlocs, HR, cardiac):
    """Calculate CPM regressors."""
    Ts = 1 / Fs
    HPF_f = 0.008
    shift_CPM = 0.5
    shift_CPM_VA = 0.5
    shift_RETR = 0
    M_order = 8
    filt_b, filt_a = signal.butter(2, HPF_f * 2 * Ts, "high")

    T_del = 5  # delete first 5 seconds from output
    nDel = int(T_del * Fs)

    HRmean = np.mean(HR)

    voxel = cardiac[nDel:-nDel]
    NV = len(voxel)
    voxel = signal.filtfilt(filt_b, filt_a, voxel)
    N = len(cardiac)
    time = np.arange(0, (N - 1) * Ts, Ts)

    memory = 60 / HRmean
    CPM_IR = func_CPM_cos(Ts, memory, M_order)

    u = np.zeros(time.shape)
    uA = u.copy()
    nPeaks = len(PPGlocs)
    for i in range(nPeaks):
        t = PPGlocs[i]
        val, loc = np.min(np.abs(time - t))
        u[loc] = 1
        uA[loc] = cardiac[loc]

    CPM_regr_all = np.zeros((N, M_order))
    CPM_Amp_regr_all = np.zeros((N, M_order))
    for m in range(M_order * 2):
        u_conv = np.convolve(u, CPM_IR[:, m])
        u_conv = u_conv[:N]
        CPM_regr_all[:, m] = u_conv[:]
        x = np.convolve(uA, CPM_IR[:, m])
        x = x[:N]
        CPM_Amp_regr_all[:, m] = x

    CPM_regr_all = signal.filtfilt(filt_b, filt_a, CPM_regr_all)
    CPM_Amp_regr_all = signal.filtfilt(filt_b, filt_a, CPM_Amp_regr_all)

    RETR_regr_all = RETR_Card_regressors_v2(time, PPGlocs, M_order)
    RETR_regr_all = signal.filtfilt(filt_b, filt_a, RETR_regr_all)

    ind = np.arange(NV)
    ind = np.round(ind + nDel + (shift_CPM * Fs))
    CPM_regr = CPM_regr_all[ind, :]

    ind = np.arange(NV)
    ind = np.round(ind + nDel + (shift_CPM_VA * Fs))
    CPM_Amp_regr = CPM_Amp_regr_all[ind, :]

    ind = np.arange(NV)
    ind = np.round(ind + nDel + (shift_RETR * Fs))
    RETR_regr = RETR_regr_all[ind, :]

    regr_CPM = np.hstack(CPM_regr, np.ones((NV, 1)))
    regr_CPM_Amp = np.hstack(CPM_Amp_regr, np.ones((NV, 1)))
    RETR_regr = np.hstack(RETR_regr, np.ones((NV, 1)))
    return regr_CPM, regr_CPM_Amp, RETR_regr


def RETR_Card_regressors_v2(time, locsECG, M):
    """Calculate RETROICOR cardiac regressors."""
    NV = len(time)
    Phi = np.zeros((NV, 1))
    for i in range(NV):
        t = time[i]
        _, minI = np.min(np.abs(locsECG - t))

        minOnLeft = t - locsECG[minI] > 0
        if (minI == 1) and not minOnLeft:
            t2 = locsECG[minI]
            t1 = t2 - 1
        elif (minI == len(locsECG)) and minOnLeft:
            t1 = locsECG[minI]
            t2 = t1 + 1
        elif minOnLeft:
            t1 = locsECG[minI]
            t2 = locsECG[minI + 1]
        else:
            t1 = locsECG[minI - 1]
            t2 = locsECG[minI]

        Phi[i] = 2 * np.pi * (t - t1) / (t2 - t1)

    Regr = np.zeros((NV, M * 2))
    for i in range(M):
        Regr[:, ((i - 1) * 2) + 1] = np.cos(i * Phi)
        Regr[:, i * 2] = np.sin(i * Phi)

    # Regr=[zeros(1,M*2);diff(Regr)];
    return Regr


def func_CPM_cos(Ts, memory, M):
    """Calculate CPM."""
    t_win = np.arange(0, memory, Ts)
    nIR = len(t_win)

    IR_all = np.zeros((nIR, M * 2))
    for m in range(M):
        IR_all[:, ((m - 1) * 2) + 1] = np.cos(m * 2 * np.pi * t_win / memory) - 1
        IR_all[:, ((m - 1) * 2) + 2] = np.sin(m * 2 * np.pi * t_win / memory)

    return IR_all
