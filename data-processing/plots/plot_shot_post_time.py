# generate plots for THz post and shot signal vs time for all shots in the processed data folder

import matplotlib.pyplot as plt
import numpy as np
import os

file_root = '/Users/yuewang/SLAC/goldthz/Au_data/800nm/processed_data/single_shot/'

def create_plot(path, wp, pos, num):
    file_path = file_root + path
    post_bal_raw = np.loadtxt(file_path + 'THz_post_bal.dat')
    post_bal = np.mean(post_bal_raw, axis=1)  # average over the columns
    shot_bal = np.loadtxt(file_path + 'THz_shot_bal.dat')
    time_axis = np.loadtxt(file_path + 'time_axis.dat')

    x, y, z = post_bal.shape, shot_bal.shape, time_axis.shape
    print("Post_bal :", x, "Shot_bal :", y, "Time_axis :", z)

    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, post_bal, "b-",label="THz post") # plot only the 1st column of post_bal
    plt.plot(time_axis, shot_bal, "r-",label="THz shot")
    plt.xlabel("Time (ps)")
    plt.xlim(0,7)
    plt.ylabel("THz signal (unit)")
    plt.title(f"THz signal comparison for {wp}, {pos}, shot {num}")
    plt.legend()
    plt.show()

    # save figure
    if not os.path.exists(file_root + "post_shot_time/"):
        os.makedirs(file_root + "post_shot_time/")
    # plt.savefig(file_root + "post_shot_time/" + f"{wp}_{pos}_shot_{num}.png", format='png', dpi=400)
    # plt.show()

def cycle_through(position: str, p_range, wp='wp43'):
    for digit in p_range:
        pos = position + str(digit)

        shot_num = 3
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
                create_plot(path, wp, pos, num)

# neg cycle
#cycle_through("neg", [i for i in range(3)])

# pos cycle
#cycle_through("pos", [1, 2, 5, '7p5', 10, '12p5', 15, 20, 25, '30_3', 30, 50, '50_3'])
cycle_through("neg", [1])





