# NMFvirtanen.py
# Performs NMF 
# Outputs the resulting componets (.wav) and the reconstructed mixture (.wav)

# Audio Mixture File Name, Num of Components, Num of Iterations, a, b
python3 NMFvirtanen.py
python3 NMFvirtanen.py A-piano-G3-C4-G6.wav
python3 NMFvirtanen.py A-piano-G3-C4-G6.wav 5
python3 NMFvirtanen.py A-piano-G3-C4-G6.wav 5 250
python3 NMFvirtanen.py A-piano-G3-C4-G6.wav 5 250 0 0

# experiments.py
# Performs the experiment as seen on the report
# Outputs the componets and the reconstructed signal
# Outupts the figures and W.npy and H.npy files

# Results Folder(Can be omitted), Audio File Name, Num of Components, Num of Iterations, a, b
python3 experiments.py A-piano-G3-C4-G6.wav 3 250 100 10
python3 experiments.py A_results A-piano-G3-C4-G6.wav 3 250 0 0

# evaluation.py
# Prints the evaluation metrics (D1, D2, D3) given the aligned initial audio components and the W.npy, H.npy files

# Path to the Audio files, Path to W.npy, H.npy
python3 evaluation.py A_audiofiles/ A_results/