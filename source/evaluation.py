#!/usr/bin/python3
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
from scipy import signal


def D1(Y, Yhat):
    NormYhat = (Yhat / np.sqrt(np.sum(Yhat**2)))
    NormY = (Y / np.sqrt(np.sum(Y**2)))
    return np.sum((NormYhat - NormY)**2) 

def D2(Y, Yhat):
    return np.sum(Y**2) / np.sum((Yhat - Y)**2)

def D3(Y, Yhat):
    return np.sum((Y * (np.log10(Y / Yhat) - 1) + Yhat)**2)

def evaluate(path, W, H):
    T = H.shape[1] # Number of time frames
    F = W.shape[0] # Frequency ticks
    K = W.shape[1] # Number of components

    compNMF = np.empty((K, F, T))
    for k in range(K):
        curW = W[:, k, None]
        curH = H[None, k, :]    # Slice while maintaining dims
        compNMF[k, :, :] = (curW * curH) 

    files = []
    comp = np.empty((K, F, T))
    for k, file in enumerate(os.listdir(path)):
        if file.endswith('.wav'):
            fs, data = wavfile.read(os.path.join(path, file))
            x = data[:,0] if len(data.shape) == 2 else data 

            winLen = int(40e-3 * fs)
            noverlap = winLen // 2
            win = signal.windows.hamming(winLen, sym=False) 

            f, t, X = signal.stft(x, fs, win, winLen, noverlap, winLen, detrend=False, return_onesided=True, boundary=None)          
            comp[k, :, :] = np.abs(X)
            
            files = files + [file]
    
    print('Componets x Files x Eval')

    for i, Yhat in enumerate(compNMF):
        print('\nComponent {}'.format(i))

        for j, Y in enumerate(comp):
            print('File:', files[j])
            print('D1:', D1(Y, Yhat))
            print('D2:', D2(Y, Yhat))
            print('D3:', D3(Y, Yhat))

if __name__ == '__main__':
    # Path to the Audio files, Path to W.npy, H.npy
    argc = len(sys.argv) - 1
        
    if argc == 2:
        W = np.load(os.path.join(sys.argv[2], 'W.npy'))
        H = np.load(os.path.join(sys.argv[2], 'H.npy'))
        evaluate(sys.argv[1], W, H)       