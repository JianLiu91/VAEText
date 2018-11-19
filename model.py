import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn import LSTM
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import sequence_cross_entropy_with_logits, masked_softmax

class PlainVAE(nn.Module):
    def __init__(self, args):
        super(PlainVAE, self).__init__()

        self.device = args['device']

        self.x_dim = args['x_dim']
        self.h_dim = args['h_dim']
        self.z_dim = args['z_dim']
        self.mb_size = args['mb_size']

        self.x2h = nn.Linear(self.x_dim, self.h_dim)
        self.h2mu = nn.Linear(self.h_dim, self.z_dim)
        self.h2var = nn.Linear(self.h_dim, self.z_dim)

        self.z2h = nn.Linear(self.z_dim, self.h_dim)
        self.h2x = nn.Linear(self.h_dim, self.x_dim)

    def q(self, x):
        h = F.relu(self.x2h(x))
        z_mu = self.h2mu(h)
        z_var = self.h2var(h)
        return z_mu, z_var

    def sample_z(self, mu, log_var):
        eps = torch.randn(self.mb_size, self.z_dim).to(self.device)
        return mu + torch.exp(log_var / 2) * eps

    def p(self, z):
        h = F.relu(self.z2h(z))
        o = self.h2x(h)
        o = F.sigmoid(o)  # for binary code
        return o

    def forward(self, x):
        z_mu, z_var = self.q(x)
        z_example = self.sample_z(z_mu, z_var)
        x_reconstructed = self.p(z_example)

        recon_loss = F.binary_cross_entropy(x_reconstructed, x, size_average=False) / self.mb_size
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))

        return recon_loss, kl_loss


class RNNVAE(nn.Module):
    def __init__(self, args):
        super(RNNVAE, self).__init__()

        self.device = args['device']
        self.mb_size = args['mb_size']

        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']

        self.encoder_input_size = args['encoder_input_size']
        self.encoder_hidden_size = args['encoder_hidden_size']
        self.encoder_num_layers = args['encoder_num_layers']

        self.z_dim = args['z_dim']

        self.decoder_num_layers = args['decoder_num_layers']
        self.decoder_hidden_size = args['decoder_hidden_size']

        self.word_embedding = nn.Embedding(self.vocab_size, self.emb_dim)

        lstm_in = nn.GRU(bidirectional=False,
                    input_size=self.encoder_input_size,
                    num_layers=self.encoder_num_layers,
                    hidden_size=self.encoder_hidden_size,
                    batch_first=True)
        self.encoder = PytorchSeq2VecWrapper(lstm_in)

        self.h2mu = nn.Linear(self.encoder_hidden_size, self.z_dim)
        self.h2var = nn.Linear(self.encoder_hidden_size, self.z_dim)

        self.z2h = nn.Linear(self.z_dim, self.decoder_hidden_size)
        lstm_out = nn.GRU(bidirectional=False,
                        input_size=self.encoder_input_size,
                        num_layers=self.decoder_num_layers,
                        hidden_size=self.decoder_hidden_size,
                        batch_first=True)
        self.decoder = PytorchSeq2SeqWrapper(lstm_out)
        self.word_classifier = nn.Linear(self.decoder_hidden_size, self.vocab_size)


    def q(self, x, mask):
        embed = self.word_embedding(x)
        h = self.encoder(embed, mask)
        z_mu = self.h2mu(h)
        z_var = self.h2var(h)
        return z_mu, z_var

    def sample_z(self, mu, log_var, dim=None):
        if dim is None:
            dim = self.mb_size
        eps = torch.randn(dim, self.z_dim).to(self.device)
        return mu + torch.exp(log_var / 2) * eps

    def p(self, ipt, mask, z):
        init_h0 = self.z2h(z).unsqueeze(0)
        hidden = self.decoder(ipt, mask, init_h0)
        x_reconstructed = self.word_classifier(hidden)
        return x_reconstructed

    def forward(self, x, x_dropout, mask):
        z_mu, z_var = self.q(x, mask)
        z_example = self.sample_z(z_mu, z_var)

        x_dropout_emb = self.word_embedding(x_dropout)
        x_reconstructed = self.p(x_dropout_emb, mask, z_example)

        recon_loss = sequence_cross_entropy_with_logits(x_reconstructed, x, mask, batch_average=False)

        # transfer the output of sequence_cross_entropy_with_logits to sentence average NLL
        recon_loss = recon_loss * (mask.sum(1).float() + 1e-13)

        recon_loss = torch.mean(recon_loss)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))

        return recon_loss, kl_loss


    def sample(self, num, ipt, z=None):
        if z is None:
            z_rand = torch.randn(num, self.z_dim).to(self.device)
        else:
            z_rand = z
        init_h0 = self.z2h(z_rand).unsqueeze(0)
        res = []
        for c in range(20):
            ipt = self.word_embedding(ipt)
            hidden = self.decoder(ipt, None, init_h0)
            out = torch.nn.functional.softmax(self.word_classifier(hidden.squeeze(1)), dim=-1)
            _, sample = torch.topk(out, 1, dim=-1)
            ipt = sample
            res.append(sample.cpu().numpy().reshape(num, 1))
        res = np.stack(res, axis=1)
        res = res.reshape(res.shape[0], res.shape[1])
        return res


if __name__ == '__main__':

    use_cuda = True
    device = torch.device("cuda:1" if use_cuda else "cpu")

    args = {
        'device': device,

        'vocab_size': 1024,
        'emb_dim': 300,
        'mb_size': 200,

        'encoder_input_size': 300,
        'encoder_hidden_size': 256,
        'encoder_num_layers': 1,

        'z_dim': 16,

        'decoder_num_layers': 1,
        'decoder_hidden_size': 256
    }

    model = RNNVAE(args).to(device)
    print(model.sample(5, torch.ones(5, 1).long().to(device)))
