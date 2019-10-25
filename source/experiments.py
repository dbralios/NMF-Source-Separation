#!/usr/bin/python3
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def separate(exName = '', fileName = 'audio.wav', numComp = 5, numIter = 100, a = 1, b = 1):
    # Read file
    fs, data = wavfile.read(fileName)
    x = data[:,0] if len(data.shape) == 2 else data 

    # Get Spectrogram  -  40ms frames, 50% overlap
    winLen = int(40e-3 * fs)
    noverlap = winLen // 2
    win = signal.windows.hamming(winLen, sym=False) 
    f, t, X = signal.stft(x, fs, win, winLen, noverlap, winLen, detrend=False, return_onesided=True, boundary=None)

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

    ang = np.angle(X)
    complexWH = WH * np.exp(1j * ang)
    _, y = signal.istft(complexWH, fs, win, winLen, noverlap, winLen, True, False)
    if exName != '':
        path = './' + exName + '/' 
        wavfile.write(path + 'recons.wav', fs, np.int16(y))

    comp = np.empty((K, F, T), dtype=np.complex_)
     # Get Components 
     # Vectorized ???
    for k in range(K):
        curW = W[:, k, None]
        curH = H[None, k, :]    # Slice while maintaining dims
        comp[k] = (curW * curH) * np.exp(1j * ang)
        _, y = signal.istft(comp[k], fs, win, winLen, noverlap, winLen, True, False)
        if exName != '':
            path = './' + exName + '/'
            wavfile.write(path + 'comp{}.wav'.format(k), fs, np.int16(y))  
    
    # Save data
    if exName != '':
        np.save('./' + exName + '/W', W)
        np.save('./' + exName + '/H', H)
    
    # Plots
    fsize = (10, 4.5)
    vmin = 0
    vmax = np.max(20*np.log10(np.abs(X)))

    # Original and Reconstructed
    fig = plt.figure(constrained_layout=False, figsize=fsize)
    gs = fig.add_gridspec(1, 2, wspace=0.1)

    ax = fig.add_subplot(gs[:, 0])
    ax.pcolormesh(t, f, 20*np.log10(V), vmin=0, vmax=vmax)
    ax.set_title('Original')
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Frequency [Hz]')
    ax = fig.add_subplot(gs[:, 1])
    ax.pcolormesh(t, f, 20*np.log10(WH), vmin=0, vmax=vmax)
    ax.set_title('Reconstructed')
    ax.set_xlabel('Time [sec]')
    ax.set_yticks([])
    plt.show()
    if exName != '':
        fig.savefig('./' + exName + '/Figure_' + exName + '1.png')

    # Spectrograms of the Components
    vmax = np.max(20*np.log10(np.abs(comp)))
    fig = plt.figure(constrained_layout=False, figsize=fsize)
    fig.suptitle('Spectrograms of the Components')
    gs = fig.add_gridspec(1, K, wspace=0.05)

    for k in range(K):
        ax = fig.add_subplot(gs[:, k])
        ax.set_title('Component {}'.format(k + 1))
        ax.pcolormesh(t, f, 20*np.log10(np.abs(comp[k, :, :])), vmin=0, vmax=vmax)
        ax.set_ylabel('Frequency [Hz]') if k == 0 else ax.set_yticks([])
        ax.set_xlabel('Time [sec]')

    plt.show()
    if exName != '':
        fig.savefig('./' + exName + '/Figure_' + exName + '2.png')

    # Activtion and Basis Vectors
    vmax = np.max(20*np.log10(np.abs(W)))
    fig = plt.figure(constrained_layout=False, figsize=fsize)
    fig.suptitle('Activation and Basis Vectors')
    gs = fig.add_gridspec(K, 3*K + 1, wspace=0.2)

    for k in range(K):
        ax = fig.add_subplot(gs[:, k - K])
        ax.imshow(20*np.log10(np.abs(W[:, k, None][::-1])), aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_yticks([])
        ax.set_xticks([])  
        ax.set_xlabel('$\mathbf{W_{' + str(k + 1) + '}}$') 

        ax = fig.add_subplot(gs[k, :-K])    
        ax.plot(t, H[k, :])
        ax.set_ylabel('$\mathbf{H_{' + str(k + 1) + '}}$')
        if k + 1 < K:
            ax.set_xticks([])
    
    ax.set_xlabel('Time [sec]')
    plt.show()
    if exName != '':
        fig.savefig('./' + exName + '/Figure_' + exName + '3.png')

if __name__ == '__main__':
    # Results Folder(Can be omitted), Audio File Name, Num of Components, Num of Iterations, a, b
    argc = len(sys.argv) - 1
    c = 0
    exName = ''
    if argc > 0 and sys.argv[1].find('.wav') == -1:
        c = 1
        exName = sys.argv[1]
        if not os.path.exists('./' + exName + '/'):
            os.makedirs('./' + exName + '/')
        
    if argc == 1 + c:
        separate(exName, sys.argv[1 + c])
    elif argc == 2 + c:
        separate(exName, sys.argv[1 + c], int(sys.argv[2 + c]))
    elif argc == 3 + c:
        separate(exName, sys.argv[1 + c], int(sys.argv[2 + c]), int(sys.argv[3 + c]))
    elif argc == 5 + c:
        separate(exName, sys.argv[1 + c], int(sys.argv[2 + c]), int(sys.argv[3 + c]), float(sys.argv[4 + c]), float(sys.argv[5 + c]))
    else:
        separate()