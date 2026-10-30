# Computes mean conductivities and standard deviation from .dat files 
# generated from the raw .bal files.
# Positions: Must be edited to correspond with the delays used in experiment

import matplotlib.pyplot as plt
import numpy.fft as fft
import numpy as np
import os
from pprint import pprint

def renameKeys(old_data): # rename keys to float, then sort by key
    new_data = {}
    for key, value in old_data.items():
        if key.startswith('neg'):
            key = key.replace('neg', '-')
            new_key = float(key)
        elif key.startswith('pos'):
            key = key.replace('pos', '')
            if 'p' in key:
                key = key.replace('p', '.')

            new_key = float(key)
        else:
            new_key = key
        new_data[new_key] = value
    sort_data = dict(sorted(new_data.items()))
    return sort_data

file_root = '/Users/yuewang/SLAC/goldthz/Au/800nm/processed_data/single_shot/'

wp = 'wp43'
pos = 'pos10'

# sigma vars
Z0 = 376.73
nSi3N4 = 2.7

sigmaData = {} # for storing mean sigmas at each pos

positions = ['neg0', 'neg1', 'neg2', 'pos1', 'pos2', 'pos5', 'pos7p5', 'pos10', 'pos12', 'pos15', 'pos20', 'pos25', 'pos30', 'pos50']
for pos in positions:
    realSigmas = [] # for calculating means across shots for each pos

    # set range to maximum trials 
    for shot in range(7):
        if not os.path.exists(file_root + f'SS_Au800_{wp}_{pos}_shot_{shot:04d}/'):
            continue

        file_path = file_root + f'SS_Au800_{wp}_{pos}_shot_{shot:04d}/'
        # load data
        post_bal_raw = np.loadtxt(file_path + 'THz_post_bal.dat')
        post_bal = np.mean(post_bal_raw, axis=1)  # average over the columns
        shot_bal = np.loadtxt(file_path + 'THz_shot_bal.dat')
        time_axis = np.loadtxt(file_path + 'time_axis.dat')

        # compute frequency axis 
        dt = time_axis[1] - time_axis[0]
        N = len(time_axis)

        # zero padding
        Nfft = 5336 # 8 times number of data points

        # padded fft
        ref_fft = fft.fft(post_bal, n=Nfft)
        sig_fft = fft.fft(shot_bal, n=Nfft)
        trans_fft = sig_fft / ref_fft

        freq = fft.fftfreq(Nfft, dt)
        # sort by frequency
        index = np.argsort(freq)
        frq_sort = freq[index]
        ref_sort = ref_fft[index]
        sig_sort = sig_fft[index]

        # generate figure for each shot (double check quality of data)
        plt.figure()
        plt.plot(frq_sort, np.abs(ref_sort), "-b", label="post-shot")
        plt.plot(frq_sort, np.abs(sig_sort), "-r", label="on-shot")
        plt.title(f"post vs on-shot for Au: {wp} {pos} {shot:04d}")
        plt.xlabel("Frequency (THz)")
        plt.ylabel(r"$\sigma_{THz}(\omega)\:(S/m)$")
        plt.xlim(0, 6)

        # finding the mean sigma in a freq range per shot
        sThickness = 30e-9
        transfer = sig_sort/ref_sort
        sigma = ((1 + nSi3N4) / (Z0*sThickness)) * ((1/transfer) - 1)
        sigmaReal = np.real(sigma)
        if shot == 3:
            sigmaMean = np.mean(sigma[(frq_sort >= 0.7) & (frq_sort <= 0.9)])
        else:
            sigmaMean = np.mean(sigma[(frq_sort >= 0.7) & (frq_sort <= 0.9)])
        realSigmas.append(np.real(sigmaMean))

        plt.figure()
        plt.plot(frq_sort, sigmaReal, "-b", label="conductivity")
        plt.title(f"Conductivity vs Frequency for Au: {wp} {pos} {shot:04d}")
        plt.xlabel("Frequency (THz)")
        plt.ylabel(r"$\sigma_{THz}(\omega)\:(S/m)$")
        plt.xlim(0, 6)
        plt.show()

        if not os.path.exists(file_root + '/conductivity plots/' + f'SS_Au800_{wp}_{pos}'):
            os.makedirs(file_root + '/conductivity plots/' + f'SS_Au800_{wp}_{pos}')

        if not os.path.exists(file_root + '/conductivity plots/' + f'SS_Au800_{wp}_{pos}/sigma_shot_{shot:04d}.png'):
            plt.savefig(file_root + '/conductivity plots/' + f'SS_Au800_{wp}_{pos}/sigma_shot_{shot:04d}.png', dpi=300)
        
        
    realSigmaMean = np.mean(realSigmas) # realSigmas holds the mean sigma for each shot at a delay
    realSigmaStd = np.std(realSigmas)
    sigmaData[pos] = {"mean": realSigmaMean, "std": realSigmaStd}

sortedSigmaData = renameKeys(sigmaData)
print("sorted procesed conductivities")
pprint(sortedSigmaData)
# conductivity vs delay figure 
plt.figure(figsize=(9,6))
yErr = [v["std"] for v in sortedSigmaData.values()]
plt.errorbar(list(sortedSigmaData.keys()), [v["mean"] for v in sortedSigmaData.values()], fmt="-ob", yerr=yErr, capsize=5, capthick=2, ms=7, label="Conductivity", lw=3)
plt.title(f"Conductivity vs Delay for Au: {wp}")
plt.xlabel("Pump-probe delay (ps)", fontsize=18, weight='bold')
plt.ylabel(r"$\mathbf{\sigma_{THz}(\omega)\:(S/m)}$", fontsize=18)
plt.xlim(-3, 50)
plt.yticks(fontsize=16)
plt.tick_params(which="both", axis="both", labelsize=14)

plt.xticks(fontsize=16)
plt.yscale('log')
plt.show()