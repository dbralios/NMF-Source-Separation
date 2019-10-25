#!/usr/bin/python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def costEuclid(V, W, H):
    WH = np.matmul(W, H)
    return np.sum((V - WH)**2)

def costKLD(V, W, H):
    WH = np.matmul(W, H)
    return np.sum(V * (np.log10(V / WH) - 1) + WH)

def costTemporal(H):
    T = H.shape[1]
    sigma = np.sum(H**2 / T, axis=1, keepdims=True)
    return np.sum(np.sum(np.diff(H)**2, axis=1, keepdims=True) / sigma**2)

def costSparsity(H):
    T = H.shape[1]
    sigma = np.sum(H**2 / T, axis=1, keepdims=True)
    return np.sum(np.abs(H / sigma))

def updateB(V, W, H):
    WH = np.matmul(W, H)
    HT = np.transpose(H)
    mul = np.matmul((V / WH), HT) / np.matmul(np.ones(WH.shape), HT)
    return W * mul

def updateG(V, W, H, a = 1, b = 1):
    K = H.shape[0]
    T = H.shape[1]
    F = V.shape[0]

    WT = np.transpose(W)
    WH = np.matmul(W, H)
    sigma = np.sum(H**2 / T, axis=1, keepdims=True)
    
    deltaPlus = np.matmul(WT, np.ones((F, T)))
    deltaPlus += a * ((4 * H) / sigma) 
    deltaPlus += b * np.sqrt(1 / sigma) 

    deltaMinus = np.matmul(WT, V / WH)
    Hp = np.roll(H, -1, axis=1)
    Hp[:, -1] = np.zeros(K)
    Hn = np.roll(H, 1, axis=1)
    Hp[:, 1] = np.zeros(K)
    deltaMinus += a * ((2 / sigma) * (Hp+Hn) + (2 * T * H * (np.sum(np.diff(H)**2, axis=1, keepdims=True) / np.sum(H**2, axis=1, keepdims=True)**2)) )
    deltaMinus += b * (H * T**0.5 * (np.sum(H, axis=1, keepdims=True)/(np.sum(H**2, axis=1, keepdims=True))**1.5))

    return H * (deltaMinus / deltaPlus)

def separate(fileName = 'audio.wav', numComp = 5, numIter = 250, a = 1, b = 1):
    # Read file
    fs, data = wavfile.read(fileName)
    x = data[:,0] if len(data.shape) == 2 else data 

    # Get Spectrogram  -  40ms frames, 50% overlap
    winLen = int(40e-3 * fs)
    noverlap = winLen // 2
    win = signal.windows.hamming(winLen, sym=False) 
    f, t, X = signal.stft(x, fs, win, winLen, noverlap, winLen, detrend=False, return_onesided=True, boundary=None)

    plt.pcolormesh(t, f, 20*np.log10(np.abs(X)))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of the Original Signal')
    plt.show()

    T = len(t) # Number of time frames
    F = len(f) # Frequency ticks
    K = numComp  # Number of components

    V = np.abs(X)
    W = np.abs(np.random.normal(size=(F, K)))
    H = np.abs(np.random.normal(size=(K, T)))

    # Train - numIter, a, b
    for i in range(numIter):
        W = updateB(V, W, H)
        H = updateG(V, W, H, a, b)
        cost = costKLD(V, W, H) + a * costTemporal(H) + b * costSparsity(H) 
        print(cost)

    # Synthesis
    WH = np.matmul(W, H)
    plt.pcolormesh(t, f, 20*np.log10(np.abs(WH)))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of the Reconstructed Signal')
    plt.show()

    # Reconstructed audio
    ang = np.angle(X)
    complexBG = WH * np.exp(1j * ang)
    t, y = signal.istft(complexBG, fs, win, winLen, noverlap, winLen, True, False)
    wavfile.write('recons.wav', fs, np.int16(y))

    # Save each component
    for j in range(K):
        curH = H[None, j, :]    # Slice while maintaining dims
        curW = W[:, j, None] 
        comp = (curW * curH) * np.exp(1j * ang)
        t, y = signal.istft(comp, fs, win, winLen, noverlap, winLen, True, False)
        wavfile.write('comp{}.wav'.format(j), fs, np.int16(y))

if __name__ == '__main__':
    #File Name, Num of Components, Num of Iterations, a, b
    argc = len(sys.argv) - 1
    if argc == 1:
        separate(sys.argv[1])
    elif argc == 2:
        separate(sys.argv[1], int(sys.argv[2]))
    elif argc == 3:
        separate(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    elif argc == 5:
        separate(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
    else:
        separate()