import json
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import cv2 as cv

def saveDict(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def cycle_through(file_root: str, position: str, p_range: list, wp: str, func: function, path: str):
    for digit in p_range:
        pos = position + str(digit)

        shot_num = 3
        for shot in range(shot_num):
            num = '{:0>4}'.format(int(shot + 1))
            path = 'SS_Au800_' + wp + '_' + pos + '_shot_' + num + "/" # default path for shots
            file_path = file_root + path

            print("--------------------------------")
            print(f"File path: {file_path}")
            if not os.path.exists(file_path):
                print(f"File path does not exist: {file_path}")
                continue
            else:
                print(f"creating plot for file: {file_path}")
                func(path, wp, pos, num)

def fft_conductivity(root, positions, wp, graph=False):
    file_root = root
    sigmaData = {}
    # sigma vars
    Z0 = 376.73
    nSi3N4 = 2.7
    for pos in positions:
        plt.figure()
        plt.title(f"Conductivity vs Frequency for Au: {wp} {pos}")
        plt.xlabel("Frequency (THz)")
        plt.ylabel(r"$\sigma_{THz}(\omega)\:(S/m)$")
        plt.xlim(0, 6)
        realSigmas = [] # for calculating means across shots for each pos

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

            # finding the mean sigma in a freq range per shot
            sThickness = 30e-9
            transfer = sig_sort/ref_sort
            sigma = ((1 + nSi3N4) / (Z0*sThickness)) * ((1/transfer) - 1)
            sigmaReal = np.real(sigma)
            sigmaSection = np.mean(sigma[(frq_sort >= 0.73) & (frq_sort <= 0.9)])
            realSigmas.append(np.real(sigmaSection))


            plt.plot(frq_sort, sigmaReal, label=f"shot: {shot}")
            plt.legend()

    print(f"sigma values at {pos}: {realSigmas}")
    if graph: 
        plt.show()
    else:
        plt.close()
        
    realSigmaMean = np.mean(realSigmas) # realSigmas holds the mean sigma for each shot at a delay
    realSigmaStd = np.std(realSigmas)
    sigmaData[pos] = {"mean": realSigmaMean, "std": realSigmaStd}
    sortedSigmaData = renameKeys(sigmaData)

    return sortedSigmaData

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

def zhijiang():
    file_root = "/Users/yuewang/SLAC/goldthz/Au_data/Zhijiang_data_Fig4.txt"

    delay, sigma, upErr, lowErr = np.loadtxt(file_root, unpack=True)
    sigma = sigma * 1e7
    upErr = upErr * 1e7
    lowErr = lowErr * 1e7
    return (delay, sigma, upErr,lowErr)

def exp_slope(sortedSigmaData, t1, t2):
    # determine exponential decay regression for conductivity vs delay between t1 and t2
    delay_range = [t for t in sortedSigmaData.keys() if t1 <= t <= t2]
    mean_values = [sortedSigmaData[t]["mean"] for t in delay_range]
    # fit the logarithm of the data to a linear function
    log_mean_values = np.log(mean_values)
    slope, intercept = np.polyfit(delay_range, log_mean_values, 1)
    yExp1 = [np.exp(slope * t + intercept) for t in delay_range]
    label = f"m = {slope:.4f}"

    return (delay_range, yExp1, label)

def conductivity_13_6():
    delay, sigma, *_ = zhijiang()
    # regression 0-14.5 ps
    x1 = [delay[i] for i in range(len(delay)) if 0 <= delay[i] <= 15]
    y1 = [sigma[i] for i in range(len(delay)) if 0 <= delay[i] <= 15]
    logY1 = np.log(y1)
    slopeZ, interceptZ = np.polyfit(x1, logY1, 1)
    yExp2 = [np.exp(slopeZ * t + interceptZ) for t in x1]
    plt.plot(x1, yExp2, "--", color="tab:red", label=f"m = {slopeZ:.4f}", lw=3, zorder=4)

    x1 = [delay[i] for i in range(len(delay)) if 15 <= delay[i]]
    y1 = [sigma[i] for i in range(len(delay)) if 15 <= delay[i]]
    logY1 = np.log(y1)
    slopeZ, interceptZ = np.polyfit(x1, logY1, 1)
    yExp2 = [np.exp(slopeZ * t + interceptZ) for t in x1]
    plt.plot(x1, yExp2, ":", color="tab:red", label=f"m = {slopeZ:.4f}", lw=3, zorder=5)

def hist(img, imgName):
    hist = cv.calcHist([img],[0], None,[256],[0,256])
    plt.figure()
    plt.plot(hist)
    plt.title(imgName)
    plt.xlabel("bins")
    plt.ylabel("# of pixels")
    plt.xlim(0,40)
    plt.show()