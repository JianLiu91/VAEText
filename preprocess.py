import pickle
import numpy as np

def data_iter():
    x = pickle.load(open("./data/bookcorpus_sample.p", "rb"))
    train, val, test = x[0], x[1], x[2]
    word2idx, idx2word = x[3], x[4]

    print(idx2word[0])


if __name__ == '__main__':
    data_iter()