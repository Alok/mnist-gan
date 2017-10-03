import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn
import torch.nn.functional as nn
import torch.optim as optim
from tensorflow.examples.tutorials.mnist import input_data
from torch.autograd import Variable
from  argparse import ArgumentParser

parser = ArgumentParser
parser.add_argument()

# TODO adapt for car image data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
BATCH_SIZE = 1024
z_dim = 10
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
d_step = 3  # discriminator takes 3 steps for every 1 the generator does.
lr = 1e-3

G = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid(),
).cuda()

D = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
).cuda()


def reset_grads():
    G.zero_grad()
    D.zero_grad()


G_optim = optim.Adam(G.parameters(), lr=lr)
D_optim = optim.Adam(D.parameters(), lr=lr)

for iter in range(1000000):
    #
    for _ in range(d_step):
        # Sample data
        z = Variable(torch.randn(BATCH_SIZE, z_dim)).cuda()
        X, _ = mnist.train.next_batch(BATCH_SIZE)
        X = Variable(torch.from_numpy(X)).cuda()

        # Discriminator
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss = 0.5 * (torch.mean((D_real - 1) ** 2) + torch.mean(D_fake ** 2))

        D_loss.backward()
        D_optim.step()
        reset_grads()

    # Generator
    z = Variable(torch.randn(BATCH_SIZE, z_dim)).cuda()

    G_sample = G(z)
    D_fake = D(G_sample)

    G_loss = 0.5 * torch.mean((D_fake - 1) ** 2)

    G_loss.backward()
    G_optim.step()
    reset_grads()

    # Print and plot every now and then
    if iter % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'.format(iter, D_loss.data[0], G_loss.data[0]))

        samples = G(z).cpu().data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
