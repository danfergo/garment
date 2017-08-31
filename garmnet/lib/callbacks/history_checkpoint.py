import numpy as np
import pickle

from keras.callbacks import Callback
import os
import sys

assert sys.version_info >= (3, 0)

parent_path = os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))) + '/history/'


def load_losses(train_name):
    past_history = {}
    full_path = parent_path + train_name + '_losses.pkl'

    try:
        pkl_file = open(full_path, 'rb')
        past_history = pickle.load(pkl_file)
        pkl_file.close()
    except Exception:
        # raise Exception
        print('Failed to load losses (' + full_path + ')')

    return past_history


class HistoryCheckpoint(Callback):
    losses = {}

    def __init__(self, train_name):
        super(HistoryCheckpoint, self).__init__()
        self.train_name = train_name

    def on_epoch_end(self, batch, logs={}):
        for k in logs:
            if k in self.losses:
                self.losses[k].append(logs[k])
            else:
                self.losses[k] = [logs[k]]

    def re_save(self):

        past_history = load_losses(self.train_name)

        full_history = {}

        for k in self.losses:
            if k in past_history:
                full_history[k] = np.concatenate((past_history[k], np.array(self.losses[k])), axis=0)
            else:
                full_history[k] = self.losses[k]

        pkl_file = open(parent_path + self.train_name + '_losses.pkl', 'wb')
        pickle.dump(full_history, pkl_file)
        pkl_file.close()

        return full_history

    def n_past_epochs(self):
        losses = load_losses(self.train_name)
        n = 0 if len(losses) == 0 else len(losses['loss'])
        print(n)
        return n
