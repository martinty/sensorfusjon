from os.path import abspath, dirname, join, pardir

import scipy.io as sio

data_path = abspath(dirname(__file__))

def get_joyride_data():
    # %% load data and plot
    filename_to_load = join(data_path, "data_joyride.mat")
    loaded_data = sio.loadmat(filename_to_load)
    T = loaded_data["time"].squeeze()
    K = loaded_data["K"].item()
    Ts = loaded_data["Ts"].squeeze()
    Xgt = loaded_data["Xgt"].T
    Z = [zk.T for zk in loaded_data["Z"].ravel()]
    return K, Ts, T, Xgt, Z