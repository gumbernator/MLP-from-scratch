# TODO: add constraints, raise errors
import numpy as np

class DataController:

    def __init__(self, data_type):
        self.data_type = data_type

    def set(self, x, y, batch_size, shuffle = False):
        self.x = x.astype(self.data_type)
        self.y = y.astype(self.data_type)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle_in_unison(self.x, self.y)
    
    def get_batches(self):
        for i in range(0, self.x.shape[0], self.batch_size):
            batch_x = self.x[i : min(self.x.shape[0], i + self.batch_size)]
            batch_y = self.y[i : min(self.y.shape[0], i + self.batch_size)]
            yield batch_x, batch_y

    def shuffle_in_unison(self, a, b):
        curr_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(curr_state)
        np.random.shuffle(b)