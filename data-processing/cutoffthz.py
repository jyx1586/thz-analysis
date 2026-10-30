# replaces end noise of thz waveform with exponential decay to 0 -- post and shot .dat

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

CUTAT = 4 # ps
file_root = '/Users/yuewang/SLAC/goldthz/Au_data/800nm/processed_data/single_shot/'

def create_cutoffs(path, wp, pos, num):
    file_path = file_root + path
    post_bal_raw = np.loadtxt(file_path + 'THz_post_bal.dat')
    post_bal = np.mean(post_bal_raw, axis=1)  # average over the columns
    shot_bal = np.loadtxt(file_path + 'THz_shot_bal.dat')
    time_axis = np.loadtxt(file_path + 'time_axis.dat')

    # a, b = 5.034, 5.17
    a, b = 4, 4.17
    tMask = (time_axis >= a) & (time_axis <= b)
    tCut = time_axis[tMask] # t values in the range a to b
    tCutoff = tCut[0] # first t value

    # sections of time, post, and shot that are not replaced
    keepMask = time_axis < a
    tKeep = time_axis[keepMask]
    postKeep = post_bal[keepMask] 
    shotKeep = shot_bal[keepMask]

    postCut = post_bal[tMask] # post values corresponding to t values between a and b
    shotCut = shot_bal[tMask] # shot values corresponding to t values between a and b

    def expDecay(x, k, A):
        return A * np.exp(-k* (x))

    tShift = tCut - tCutoff # with this shift, the decay starts at 0, which is needed for exp decay fit

    lower_bounds = [0.001, 0.0]
    upper_bounds = [np.inf, np.inf]
    p_popt, _ = curve_fit(expDecay, tShift, postCut, p0=[0.1, postCut[0]], bounds=(lower_bounds, upper_bounds))
    # s_popt, _= curve_fit(expDecay, tShift, shotCut, p0=[0.1, shotCut[0]], bounds=(lower_bounds, upper_bounds))

    # generate new decay points
    dt = time_axis[1] - time_axis[0]
    EXTEND_PERIOD = 3 # ps 
    tExtend = np.arange(b + dt, b + EXTEND_PERIOD, dt) # add points from after b to b + 3 ps with same dt spacing
    tTotal = np.concatenate([tCut, tExtend]) # set of all t values including extension
    print(tTotal.shape[0])
    tShift = tTotal - tCutoff

    # replace values after time b with decay function
    yDecay = expDecay(tShift, *p_popt)
    postFinal = np.concatenate([postKeep, yDecay])

    if CUTAT == 4:
        yDecay = np.zeros(tTotal.shape[0])
    else:
        yDecay = expDecay(tShift, *s_popt)
    shotFinal = np.concatenate([shotKeep, yDecay])

    tFinal = np.concatenate([tKeep, tTotal])

    # for debugging, set true to see the plots
    PLOT = True
    if PLOT: 
        plt.plot(tFinal, postFinal, label='post')
        plt.plot(tFinal, shotFinal, label='shot')
        plt.title(f"{pos} shot {num}")
        plt.show()

    # save as new dat
    SAVE = False
    if SAVE: 
        np.savetxt(file_path + 'dec_post_4ps.dat', postFinal, delimiter='\t')
        np.savetxt(file_path + 'dec_shot_4ps.dat', shotFinal, delimiter='\t')
        np.savetxt(file_path + 'dec_time_4ps.dat', tFinal, delimiter='\t')

def cycle_through(position: str, p_range, wp='wp43'):
    for digit in p_range:
        pos = position + str(digit)

        shot_num = 6
        for shot in range(shot_num):
            num = '{:0>4}'.format(int(shot + 1))
            path = 'SS_Au800_' + wp + '_' + pos + '_shot_' + num + "/"
            file_path = file_root + path

            print("--------------------------------")
            print(f"File path: {file_path}")
            if not os.path.exists(file_path):
                print(f"File path does not exist: {file_path}")
                continue
            else:
                print(f"creating plot for file: {file_path}")
                create_cutoffs(path, wp, pos, num)

# for file naming convention with neg and pos positions
# neg cycle
cycle_through("neg", [i for i in range(3)])

# pos cycle
cycle_through("pos", [1, 2, 5, '7p5', 10, 12, 15, 20, 25, 30, 50])
