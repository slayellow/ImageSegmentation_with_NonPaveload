import numpy as np

class NPYLoader:

    def __init__(self):
        self.data_dic = None

    def load_npy(self, npy_path=None):
        if npy_path is None:
            return

        print("NPY Path : " + str(npy_path))

        self.data_dic = np.load(npy_path, encoding='latin1', allow_pickle=True).item()

    def get_keys_list(self):
        return list(self.data_dic.keys())

    def get_values(self, key, index):
        return self.data_dic[key][index]
