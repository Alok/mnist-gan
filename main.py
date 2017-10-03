#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
from argparse import ArgumentParser

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
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential, Sigmoid
from torch.optim import Adam

ITERS = 1000000  # number of training steps (for the generator)

parser = ArgumentParser()
parser.add_argument('--render', action='store_true')
parser.add_argument('--iterations', '-n', type=int, default=ITERS)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# TODO adapt for car image data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
BATCH_SIZE = args.batch_size
z_dim = 10
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
hidden_dim = 128
d_step = 3  # discriminator takes 3 steps for every 1 the generator does.
lr = 1e-3

G = Sequential(
    Linear(z_dim, hidden_dim),
    ReLU(),
    Linear(hidden_dim, X_dim),
    Sigmoid(),
).cuda()

D = Sequential(
    Linear(X_dim, hidden_dim),
    ReLU(),
    Linear(hidden_dim, 1),
).cuda()


def reset_grads():
    G.zero_grad()
    D.zero_grad()


G_optim, D_optim = Adam(G.parameters(), lr=lr), Adam(D.parameters(), lr=lr)

if __name__ == '__main__':
    for iter in range(args.iterations):
        #
        for _ in range(d_step):
            # Sample data
            z = Variable(torch.randn(BATCH_SIZE, z_dim)).cuda()
            X = Variable(torch.from_numpy(mnist.train.next_batch(BATCH_SIZE)[0]).pin_memory()
                         ).cuda()

            # Discriminator

            D_loss = (torch.mean((D(X) - 1) ** 2) + torch.mean(D(G(z)) ** 2)) / 2

            D_loss.backward()
            D_optim.step()
            reset_grads()

        # Generator
        z = Variable(torch.randn(BATCH_SIZE, z_dim)).cuda()

        G_loss = torch.mean((D(G(z)) - 1) ** 2) / 2

        G_loss.backward()
        G_optim.step()
        reset_grads()

        # Print and plot every now and then
        if iter % 1000 == 0:
            print(
                'i: {}'.format(iter),
                'D: {:.4}'.format(D_loss.data[0]),
                'G: {:.4}'.format(G_loss.data[0]),
                72 * '-',  # Separator for each print summary
                sep='\n',
            )

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

            if not os.path.exists('imgs/'):
                os.makedirs('imgs/')

            plt.savefig('imgs/{}.png'.format(str(iter)), bbox_inches='tight')
            plt.close(fig)
