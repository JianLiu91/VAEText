import pickle
import sys
import torch
import numpy.random as random

from tqdm import tqdm
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

class Dataset(object):
    def __init__(self, dataset, batch_size, word2idx, device):
        super(Dataset, self).__init__()

        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

        self.batch_size = batch_size
        self.word2idx = word2idx
        self.device = device

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, shuffle=True):
        return tqdm(self.reader(shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch)
        if shuffle:
            self.shuffle()

    def batchify(self, batch):
        cur_batch_size = len(batch)

        encode_sequence_ipt = []
        decode_sequence_ipt = []

        for instance_ind in range(cur_batch_size):
            instance = batch[instance_ind]
            encode_sequence_ipt.append(instance[:] + [self.word2idx['<END>']])
            decode_sequence_ipt.append([self.word2idx['<BOS>']] + instance[:])

        lens = [len(tup) for tup in encode_sequence_ipt]
        max_len = max(lens)

        encode_sequence_ipt = list(map(lambda x: pad_sequence_to_length(x, max_len), encode_sequence_ipt))
        decode_sequence_ipt = list(map(lambda x: pad_sequence_to_length(x, max_len), decode_sequence_ipt))
        mask = get_mask_from_sequence_lengths(torch.LongTensor(lens), max_len)

        encode_sequence_ipt = torch.LongTensor(encode_sequence_ipt).to(self.device)
        decode_sequence_ipt = torch.LongTensor(decode_sequence_ipt).to(self.device)
        mask = mask.to(self.device)

        return [encode_sequence_ipt, decode_sequence_ipt, mask]


if __name__ == '__main__':
    x = pickle.load(open("./data/bookcorpus_sample.p", "rb"))
    train, val, test = x[0], x[1], x[2]
    word2idx, idx2word = x[3], x[4]

    ds = Dataset(train, 10, word2idx, 'cpu')

    for elem in ds.get_tqdm(shuffle=True):
        #print(elem)
        pass