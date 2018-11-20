import pickle
import numpy as np

def prepare_bookcorpus():
    x = pickle.load(open("./data/bookcorpus_sample.p", "rb"))
    train, val, test = x[0], x[1], x[2]
    word2idx, idx2word = x[3], x[4]

    return train, val, test, word2idx, idx2word


def prepare_ptb():
    def _worddict_for_file(filename, wd):
        with open(filename) as filein:
            for line in filein:
                for w in line.split():
                    if w != '<unk>' and w not in wd:
                        wd[w] = len(wd)

    wd = {'<PAD>': 0, '<UNK>': 1, '<BOS>':2, '<END>': 3}
    _worddict_for_file('data/data_ptb/ptb.train.txt', wd)
    _worddict_for_file('data/data_ptb/ptb.valid.txt', wd)
    _worddict_for_file('data/data_ptb/ptb.test.txt', wd)

    word2idx = wd
    idx2word = {v: k for k, v in word2idx.items()}

    def _trans_file(filename, wd):
        res = []
        with open(filename) as filein:
            for line in filein:
                tmp = []
                for w in line.split():
                    w = '<UNK>' if w == '<unk>' else w
                    tmp.append(wd[w])
                res.append(tmp)
        return res

    train = _trans_file('data/data_ptb/ptb.train.txt', wd)
    val = _trans_file('data/data_ptb/ptb.valid.txt', wd)
    test = _trans_file('data/data_ptb/ptb.test.txt', wd)

    return train, val, test, word2idx, idx2word



if __name__ == '__main__':
    train, val, test, wd, iw = prepare_ptb()
    print(train[0])