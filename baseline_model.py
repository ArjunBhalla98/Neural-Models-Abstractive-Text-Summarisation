import pickle
import torch.nn as nn
import torchvision as torch
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Input, GRU
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def load_data(ds):
    """
    Returns a tuple of lists of (headline, article, id) tuples (all in integer encoded format).
    The dataset subsection is based on the input arg: anything except 0, 1, or 2 will return 0 
    Then, it also returns the validation then test sets.
    """
    ds_return = None
    validation = None
    test = None

    if ds == 1:
        with open("train_1.pickle", "rb") as f:
            ds_return = pickle.load(f)
    elif ds == 2:
        with open("train_2.pickle", "rb") as f:
            ds_return = pickle.load(f)
    else:
        with open("train_0.pickle", "rb") as f:
            ds_return = pickle.load(f)

    with open("validation.pickle", "rb") as f:
        validation = pickle.load(f)

    with open("test.pickle", "rb") as f:
        test = pickle.load(f)

    return ds_return, validation, test


def get_xy(data, seq_length):
    """
    Will split into x, y set of fixed x length (article).
    """
    fin_x = []
    fin_y = []

    # Teacher force this bitch (for now)
    for ex in data:
        headline = ex[0]
        article = ex[1]
        for i in range(0, len(headline) - seq_length, 1):
            seq_in = headline[i : i + seq_length]
            seq_out = [headline[i + seq_length]]
            fin_x.append(seq_in)
            fin_y.append(seq_out)
        ############# COULD BE DUBIOUS HERE, DOING THIS ONLY FOR A BASELINE CHECK ################
        # if len(headline) < n_feats:
        #     while len(headline) < n_feats:
        #         headline.append(
        #             -1
        #         )  # Unknown characters. Will definitely fuck up accuracy.

    fin_x = np.asarray(fin_x)
    fin_x = fin_x / np.linalg.norm(fin_x)  # Normalise data for softmax
    y = np.asarray(fin_y)
    y = y / np.linalg.norm(y)

    X = np.reshape(fin_x, (len(fin_x), seq_length, 1))
    return X, y


def train(seq_length):
    """
    Trains using a basic NN (may be updated later to reflect easy functionality)
    """
    train_set, valid_set, test_set = load_data(0)
    train_X, train_y = get_xy(train_set, seq_length)

    print(np.shape(train_X))
    print(np.shape(train_y))
    # LSTM model
    model = Sequential()
    model.add(GRU(256, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(train_y.shape[1], activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    # define the checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=1, save_best_only=True, mode="min"
    )
    callbacks_list = [checkpoint]

    model.fit(train_X, train_y, epochs=20, batch_size=128, callbacks=callbacks_list)


if __name__ == "__main__":
    train(5)
