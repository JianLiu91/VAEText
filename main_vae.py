from tensorflow.examples.tutorials.mnist import input_data

import os
import numpy as np

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import VAEText

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

if __name__ == '__main__':

    use_cuda = True
    device = torch.device("cuda:1" if use_cuda else "cpu")

    lr = 1e-3

    args = {
        'device': device,
        'x_dim': mnist.train.images.shape[1],
        'z_dim': 100,
        'h_dim': 128,
        'mb_size': 64
    }

    model = VAEText(args).to(device)
    solver = optim.Adam(model.parameters(), lr=lr)

    c = 0
    for it in range(100000):
        model.train()
        solver.zero_grad()

        X, _ = mnist.train.next_batch(args['mb_size'])
        X = torch.from_numpy(X).to(device)

        recon_loss, kl_loss = model(X)
        loss = recon_loss + kl_loss

        loss.backward()
        solver.step()

        # for test
        if it % 1000 == 0:
            with torch.no_grad():
                model.eval()

                print('Iter-{}; Reconstruct Loss: {:.4}, KL Loss {:.4}, Loss: {:.4}'.format(it, recon_loss.item(), kl_loss.item(), loss.item()))

                # ____true
                samples = X.cpu().numpy()[:16]
                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(4, 4)
                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}_true.png'.format(str(c).zfill(3)), bbox_inches='tight')
                plt.close(fig)

                # ___________________________________
                z_mu, z_var = model.q(X)

                # ____samples_z_mean
                samples = model.p(z_mu).cpu().numpy()[:16]
                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(4, 4)
                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}_mean.png'.format(str(c).zfill(3)), bbox_inches='tight')
                plt.close(fig)


                # ____samples1
                z_sample = model.sample_z(z_mu, z_var)
                samples = model.p(z_sample).cpu().numpy()[:16]
                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(4, 4)
                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}_rec1.png'.format(str(c).zfill(3)), bbox_inches='tight')
                plt.close(fig)

                # ____samples2
                z_sample = model.sample_z(z_mu, z_var)
                samples = model.p(z_sample).cpu().numpy()[:16]
                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(4, 4)
                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}_rec2.png'.format(str(c).zfill(3)), bbox_inches='tight')
                plt.close(fig)


                # ____random
                z_rand = torch.randn(args['mb_size'], args['z_dim']).to(device)
                samples = model.p(z_rand).cpu().numpy()[:16]
                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(4, 4)
                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}_rand.png'.format(str(c).zfill(3)), bbox_inches='tight')
                plt.close(fig)

                c = c + 1