import numpy as np


data_dir = "/itet-stor/wolflu/net_scratch/projects/General_DETRtime/data/processed/sleep-edf-153"
file_names =['val.npz']
seq_len = 32
def cut_test(data_dir, filename, seq_len):
    data = np.load(data_dir + '/' + filename)
    X = data['EEG']
    y = data['labels']
    X = X[:seq_len, :, :]
    y = y[:seq_len, :]
    np.savez(data_dir+'/cut_'+filename, EEG=X, labels=y)

for f in file_names:
    cut_test(data_dir, f, seq_len)