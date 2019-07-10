#!/usr/bin/python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def costEuclid(X, B, G):
    BG = np.matmul(B, G)
    return np.sum((X - BG)**2)

def costKLD(X, B, G):
    BG = np.matmul(B, G)
    return np.sum(X * (np.log10(X / BG) - 1) + BG)

def costTemporal(G):
    T = G.shape[1]
    sigma = np.sum(G**2 / T, axis=1, keepdims=True)
    return np.sum(np.sum(np.diff(G)**2, axis=1, keepdims=True) / sigma**2)

def costSparsity(G):
    T = G.shape[1]
    sigma = np.sum(G**2 / T, axis=1, keepdims=True)
    return np.sum(np.abs(G / sigma))

def updateB(X, B, G):
    BG = np.matmul(B, G)
    GT = np.transpose(G)
    mul = np.matmul((X / BG), GT) / np.matmul(np.ones(BG.shape), GT)
    return B * mul

def updateG(X, B, G, a = 1, b = 1):
    J = G.shape[0]
    T = G.shape[1]
    K = X.shape[0]

    BT = np.transpose(B)
    BG = np.matmul(B, G)
    sigma = np.sum(G**2 / T, axis=1, keepdims=True)
    
    deltaPlus = np.matmul(BT, np.ones((K, T)))
    deltaPlus += a * ((4 * G) / sigma) 
    deltaPlus += b * np.sqrt(1 / sigma) 

    deltaMinus = np.matmul(BT, X / BG)
    Gp = np.roll(G, -1, axis=1)
    Gp[:, -1] = np.zeros(J)
    Gn = np.roll(G, 1, axis=1)
    Gp[:, 1] = np.zeros(J)
    deltaMinus += a * ((2 / sigma) * (Gp+Gn) + (2 * T * G * (np.sum(np.diff(G)**2, axis=1, keepdims=True) / np.sum(G**2, axis=1, keepdims=True)**2)) )
    deltaMinus += b * (G * T**0.5 * (np.sum(G, axis=1, keepdims=True)/(np.sum(G**2, axis=1, keepdims=True))**1.5))

    return G * (deltaMinus / deltaPlus)

def separate(fileName = 'audio.wav', numComp = 5, numIter = 100, a = 1, b = 1):
    # Read file
    fs, data = wavfile.read(fileName)
    x = data[:,0] if len(data.shape) == 2 else data 

    # Get Spectrogram  -  40ms frames, 50% overlap
    winLen = int(40e-3 * fs)
    noverlap = winLen // 2
    win = signal.windows.hamming(winLen, sym=False) 
    #f, t, S = signal.spectrogram(x, fs, win, winLen, noverlap, winLen, False, mode='magnitude')
    f, t, S = signal.stft(x, fs, win, winLen, noverlap, winLen, detrend=False, return_onesided=True, boundary=None)

    plt.pcolormesh(t, f, 20*np.log10(np.abs(S)))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.show()

    T = len(t) # Number of time frames
    K = len(f) # Frequency ticks
    J = numComp  # Number of components

    X = np.abs(S)
    B = np.abs(np.random.normal(size=(K, J)))
    G = np.abs(np.random.normal(size=(J, T)))

    # Train - numIter, a, b
    for i in range(numIter):
        B = updateB(X, B, G)
        G = updateG(X, B, G, a, b)
        cost = costKLD(X, B, G) + a * costTemporal(G) + b * costSparsity(G) 
        print(cost)

    # Synthesis
    BG = np.matmul(B, G)
    plt.pcolormesh(t, f, 20*np.log10(np.abs(BG)))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.show()

    ang = np.angle(S)
    complexBG = BG * np.exp(1j * ang)
    t, y = signal.istft(complexBG, fs, win, winLen, noverlap, winLen, True, False)
    wavfile.write('res.wav', fs, np.int16(y))

    #plt.imshow(B[:,1, None], extent=[0,1,0,20000], aspect='auto', interpolation='nearest')
    #plt.show()
    # Get Components
    for j in range(J):
        curG = G[None, j, :]    # Slice while maintaining dims
        curB = B[:, j, None] 
        comp = (curB * curG) * np.exp(1j * ang)
        t, y = signal.istft(comp, fs, win, winLen, noverlap, winLen, True, False)
        wavfile.write('comp{}.wav'.format(j), fs, np.int16(y))

if __name__ == '__main__':
    #File Name, Num of Components, Num of Iterations, a, b
    argc = len(sys.argv) - 1
    if argc == 1:
        separate(sys.argv[1])
    elif argc == 2:
        separate(sys.argv[1], sys.argv[2])
    elif argc == 3:
        separate(sys.argv[1], sys.argv[2], sys.argv[3])
    elif argc == 5:
        separate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        separate()