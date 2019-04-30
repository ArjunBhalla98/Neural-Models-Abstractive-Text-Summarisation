import pickle
import torch.nn as nn
import torchvision as torch
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, GRU
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.layers.embeddings import Embedding
import random


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

    # Teacher forced training
    for ex in data[:5000]:
        headline = ex[0]
        article = ex[1]
        for i in range(0, len(headline) - seq_length, 1):
            seq_in = headline[i : i + seq_length]
            seq_out = [headline[i + seq_length]]
            fin_x.append(seq_in)
            fin_y.append(seq_out)
    X = np.asarray(fin_x)
    print(np.shape(X))
    X = np.reshape(X, (len(X), seq_length))
    y = np_utils.to_categorical(fin_y)  # One-hot encoding
    return X, y


def train(seq_length, do_output, num_words):
    """
    Trains using a basic NN (may be updated later to reflect easy functionality)
    """
    train_set, valid_set, test_set = load_data(0)
    train_X, train_y = get_xy(train_set, seq_length)

    adam = Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    )

    # LSTM model
    model = Sequential()
    model.add(Embedding(num_words, 100, input_length=seq_length))
    model.add(
        GRU(
            512, input_shape=(train_X.shape[0], train_X.shape[1]), return_sequences=True
        )
    )
    model.add(Dropout(0.2))
    model.add(GRU(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(512))
    model.add(Dropout(0.1))
    model.add(Dense(train_y.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    # define the checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=1, save_best_only=True, mode="min"
    )
    callbacks_list = [checkpoint]

    if not do_output:
        model.fit(train_X, train_y, epochs=30, batch_size=64, callbacks=callbacks_list)
    else:
        filename = "weights-improvement-22-0.8882.hdf5"
        predict(filename, model, seq_length, num_words)


def article_to_net(article, seq_length):
    x = []
    y = []
    for i in range(0, len(article) - seq_length, 1):
        seq_in = article[i : i + seq_length]
        seq_out = [article[i + seq_length]]
        x.append(seq_in)
        y.append(seq_out)

    x = np.asarray(x)
    x = np.reshape(x, (len(x), seq_length, 1))

    return x


def predict(filename, model, seq_length, num_words):
    test_set = None
    inv_idx = None
    with open("train_0.pickle", "rb") as f:
        test_set = pickle.load(f)

    with open("inv_idx.pickle", "rb") as f:
        inv_idx = pickle.load(f)

    test_X, test_y = get_xy(test_set, seq_length)
    test_set = test_set[: int(len(test_set) * 0.2)]
    model.load_weights(filename)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    num_gens = 5
    for i in range(num_gens):
        ex_test = random.randint(0, len(test_set) - 1)
        while len(test_set[ex_test][0]) < seq_length:
            ex_test = random.randint(0, len(test_set) - 1)
        x = test_set[ex_test][0][:seq_length]
        final_out = []
        act_out = []

        for wrd_idx in test_set[ex_test][0]:
            if wrd_idx != num_words:  # I know magic numbers are bad code leave me alone
                act_out.append(inv_idx[wrd_idx])

        print("Article headline actual: " + " ".join(act_out))
        for i in range(15):
            x = np.reshape(x, (1, len(x)))
            prediction = model.predict(x, verbose=0)
            idx = np.argmax(prediction)
            result = "DNK" if idx == (num_words) else inv_idx[idx]
            final_out.append(result)
            x = np.append(x, idx)
            x = x[1:]
        print("Generated headline: " + " ".join(final_out), end="\n")
        print()


def continue_train(model, epch, seq_length):
    """
    Loads a pretrained model and will continue to train it on some data.
    """
    n_model = load_model(model)
    train_set, valid_set, test_set = load_data(0)
    train_X, train_y = get_xy(train_set, seq_length)

    # define the checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=1, save_best_only=True, mode="min"
    )
    callbacks_list = [checkpoint]
    n_model.fit(train_X, train_y, epochs=epch, callbacks=callbacks_list)


if __name__ == "__main__":
    train(5, True, 20000)
    # continue_train("weights-improvement-22-0.8882.hdf5", 30, 5)
