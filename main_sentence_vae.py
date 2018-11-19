import pickle
import torch
import torch.optim as optim
import numpy as np

from dataset import Dataset
from model import RNNVAE

def logistic(x, x0=5000, k=0.0050):
    return float(1 / (1 + np.exp(-k * (x - x0))))

if __name__ == '__main__':

    x = pickle.load(open("./data/bookcorpus_sample.p", "rb"))
    train, val, test = x[0], x[1], x[2]
    word2idx, idx2word = x[3], x[4]

    use_cuda = True
    device = torch.device("cuda:1" if use_cuda else "cpu")

    batch_size = 500
    learning_rate = 0.01
    lr_decay =  0.05
    word_keep_rate = 0.5 # word dropout, 1.0 == no word dropout

    dataset = Dataset(train, batch_size, word2idx, device)
    batch_per_epoch = int(dataset.index_length / batch_size)

    args = {
        'device': device,

        'vocab_size': len(word2idx),
        'emb_dim': 300,
        'mb_size': batch_size,

        'encoder_input_size': 300, # should be equal to emb_dim
        'encoder_hidden_size': 512,
        'encoder_num_layers': 1,

        'z_dim': 256,

        'decoder_num_layers': 1,
        'decoder_hidden_size': 512
    }


    model = RNNVAE(args).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    step = 0
    for indexs in range(1000):
        current_lr = learning_rate / (1 + (indexs + 1) * lr_decay)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.train()

        rec_loss_tatal = 0
        kl_loss_total = 0

        for encoder_ipt, decoder_ipt, mask in dataset.reader(shuffle=True):
            step += 1
            model.train()
            optimizer.zero_grad()
            # word dropout
            word_dropout_mask = (torch.FloatTensor(decoder_ipt.size()).uniform_() < word_keep_rate).long().to(device)
            word_dropout_mask_flip = 1 - word_dropout_mask
            temp = word2idx['<UNK>'] * word_dropout_mask_flip
            decoder_ipt = decoder_ipt * word_dropout_mask + temp
            decoder_ipt[:, 0] = word2idx['<BOS>'] # we keep the <BOS> symbol

            recon_loss, kl_loss = model(encoder_ipt, decoder_ipt, mask)


            x0 = batch_per_epoch * 2  # logistic(x0) == 0.5 there
            kl_weight = logistic(step, x0)

            loss = recon_loss  + kl_loss * kl_weight

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            print('Iter-{}|{}/{}; Rec Loss: {:.4}, KL Loss {:.4}, KL weight {:.8}, Total Loss: {:.4}'.format(
                indexs, step % batch_per_epoch, batch_per_epoch,
                recon_loss.item(), kl_loss.item(), kl_weight, loss.item()))

            if step % 100 == 0:
                def _truncate_list(word_list, ch='<END>'):
                    end_idx = len(word_list)
                    try:
                        end_idx = word_list.index(ch)
                    except:
                        pass
                    return word_list[:end_idx]

                with torch.no_grad():
                    model.eval()
                    sample_count = 5
                    bos_symbol = word2idx['<BOS>']

                    print()
                    print('_____random sampling..... z ~ N(0,1)')
                    res = model.sample(sample_count, bos_symbol * torch.ones(sample_count, 1).long().to(device))
                    sen = [ [idx2word[t] for t in s] for s in res]
                    print()
                    for s in sen:
                        s = _truncate_list(s)
                        print(' '.join(s))
                    print()

                    print('_____encoder decoder .....')
                    x, mask = encoder_ipt[:sample_count], mask[:sample_count]
                    z_mu, z_var = model.q(x, mask)

                    print('__mean')
                    res = model.sample(sample_count,
                                       bos_symbol * torch.ones(sample_count, 1).long().to(device),
                                       z_mu)
                    sen = [[idx2word[t] for t in s] for s in res]
                    sen_o = [[idx2word[t] for t in s] for s in x.cpu().numpy()]
                    for o, s in zip(sen_o, sen):
                        o = _truncate_list(o)
                        s = _truncate_list(s)
                        print(' '.join(o), '|||', ' '.join(s))
                    print()

                    print('___sample1')
                    z_example1 = model.sample_z(z_mu, z_var, sample_count)
                    res = model.sample(sample_count,
                                       bos_symbol * torch.ones(sample_count, 1).long().to(device),
                                       z_example1)
                    sen = [[idx2word[t] for t in s] for s in res]
                    sen_o = [[idx2word[t] for t in s] for s in x.cpu().numpy()]
                    for o, s in zip(sen_o, sen):
                        o = _truncate_list(o)
                        s = _truncate_list(s)
                        print(' '.join(o), '|||', ' '.join(s))
                    print()

                    print('___sample2')
                    z_example2 = model.sample_z(z_mu, z_var, sample_count)
                    res = model.sample(sample_count,
                                       bos_symbol * torch.ones(sample_count, 1).long().to(device),
                                       z_example2)
                    sen = [[idx2word[t] for t in s] for s in res]
                    sen_o = [[idx2word[t] for t in s] for s in x.cpu().numpy()]
                    for o, s in zip(sen_o, sen):
                        o = _truncate_list(o)
                        s = _truncate_list(s)
                        print(' '.join(o), '|||', ' '.join(s))
                    print()


                    print('_____interpolating')
                    x, mask = encoder_ipt[:2], mask[:2]
                    z_mu, z_var = model.q(x, mask)
                    res = model.sample(2,
                                       bos_symbol * torch.ones(2, 1).long().to(device),
                                       z_mu)

                    sen1, sen2 = res
                    sen1 = [idx2word[t] for t in sen1]
                    sen2 = [idx2word[t] for t in sen2]
                    sen1 = _truncate_list(sen1)
                    sen2 = _truncate_list(sen2)

                    sen_o1 = _truncate_list([idx2word[t] for t in x.cpu().numpy()[0]])
                    sen_o2 = _truncate_list([idx2word[t] for t in x.cpu().numpy()[1]])

                    print(' '.join(sen1), '________________SOURCE___________[origin sentence:', ' '.join(sen_o1), ']')
                    beg, end = z_mu.cpu().numpy()
                    inter = []
                    for i in range(1, 10):
                        inter_point = [i/10 * (e-b) + b for (b, e) in zip(beg, end)]
                        inter.append(inter_point)

                    res = model.sample(9,
                                       bos_symbol * torch.ones(9, 1).long().to(device),
                                       torch.Tensor(inter).float().to(device))
                    sen = [[idx2word[t] for t in s] for s in res]
                    for s in sen:
                        s = _truncate_list(s)
                        print(' '.join(s))
                    print(' '.join(sen2), '________________TARGET___________[origin sentence:', ' '.join(sen_o2), ']')
                    print()

