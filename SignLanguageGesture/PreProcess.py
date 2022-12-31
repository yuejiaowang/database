import os

from matplotlib import colors
from scipy import signal
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


def audio2img(fname, save_name=None):
    
    # sample rate
    sr = 44100

    # load the source audio
    y, sr = librosa.load(fname, sr=sr)

    # ButterWorth filter
    bandpass = signal.butter(10, [19000, 21000], btype='bandpass', fs=sr, output='sos')
    filtered = signal.sosfilt(bandpass, y)
    bandstop = signal.butter(10, [19985, 20015], btype='bandstop', fs=sr, output='sos')
    filtered = signal.sosfilt(bandstop, filtered)

    # stft
    stft_signal = librosa.stft(filtered, win_length=2048)

    amplitude = np.abs(stft_signal)
    amplitude = ndimage.gaussian_filter(amplitude, sigma=(1, 1), order=0)

    # binary
    # amplitude = np.where(amplitude < 1, 0, 8)

    # visualize
    plt.figure(figsize=(8, 6))

    D = librosa.amplitude_to_db(amplitude, ref=np.max)

    jet_160 = plt.cm.get_cmap('jet', 160)
    jet_500 = plt.cm.get_cmap('jet', 3000)
    jet_80 = plt.cm.get_cmap('jet', 60)

    hsv = plt.cm.get_cmap('hsv', 200)
    jet_160 = jet_160(np.linspace(0, 1, 160))
    jet_500 = jet_500(np.linspace(0, 1, 3000))
    jet_80 = jet_80(np.linspace(0, 1, 60))
    hsv = hsv(np.linspace(0, 1, 200))

    # jet_160[105:110, :] = np.flipud(hsv[40:45, :])
    jet_160[:110, :] = jet_500[60:170, :]
    jet_160[110:, :] = jet_80[10:, :]

    my_jet = colors.ListedColormap(jet_160)

    # librosa.display.specshow(D, y_axis='fft', x_axis='time', sr=sr, cmap='plasma')
    librosa.display.specshow(D, y_axis='fft', x_axis='time', sr=sr, cmap=my_jet)



    # plt.specgram(D)
    plt.axis('off')
    plt.ylim([19000, 21000])
    # plt.ylim([17000, 22000])
    plt.xlim([0.3, 8.1])
    plt.ylabel('Frequency(kHz)', fontsize=20, )
    plt.xlabel('Time(s)', fontsize=20, )
    # plt.savefig("1.png", bbox_inches='tight', dpi=500)
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', pad_inches = 0)
    else:
        plt.savefig("temp.jpg", bbox_inches='tight', pad_inches = 0)
    # plt.show()


if __name__ == '__main__':
    # single image
    # file_name = 'wavs/2022.08.18.212031.wav'
    # audio2img(file_name)

    path = 'wavs'
    save_dir = 'images'
    for i, f in enumerate(os.listdir(path)):
        if (i+1) % 20 == 0:
            print(i)
        file_front = f[:-4]
        audio2img(os.path.join(path, f), save_name = os.path.join(save_dir, file_front + '.jpg'))





