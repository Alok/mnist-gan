#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from argparse import ArgumentParser

import torch
from tensorflow.examples.tutorials.mnist import input_data
from torch.autograd import Variable
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Linear,
    Sequential,
    Sigmoid,
    Tanh,
)
from torch.optim import Adam


# Need this or else plotting won't work
import matplotlib  # isort:skip

matplotlib.use("Agg")  # isort:skip

import matplotlib.gridspec as gridspec  # isort:skip
import matplotlib.pyplot as plt  # isort:skip

parser = ArgumentParser()
parser.add_argument("--render", action="store_true")
parser.add_argument("--iterations", "-n", type=int, default=1_000_000)
parser.add_argument("--batch_size", "-b", type=int, default=128)
parser.add_argument("--log_every", "-l", type=int, default=1000)
parser.add_argument("--new", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("-z", "--z_dim", type=int, default=16)
parser.add_argument("-i", "--h_dim", type=int, default=64)
parser.add_argument("-d", "--discrim_steps", type=int, default=3)
parser.add_argument("--save_every", type=int, default=5000)
args = parser.parse_args()

mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)
dataset = mnist.train

BATCH_SIZE = args.batch_size

Z_DIM, X_DIM = args.z_dim, dataset.images.shape[1]

# discriminator takes 3 steps for every 1 the generator does.
HIDDEN_DIM, DISCRIM_STEPS = args.h_dim, args.discrim_steps

ACTIVATION = Tanh


def create_generator() -> Sequential:
    G = Sequential(
        Linear(Z_DIM, HIDDEN_DIM),
        ACTIVATION(),
        Linear(HIDDEN_DIM, HIDDEN_DIM),
        ACTIVATION(),
        BatchNorm1d(HIDDEN_DIM),
        Linear(HIDDEN_DIM, HIDDEN_DIM),
        ACTIVATION(),
        BatchNorm1d(HIDDEN_DIM),
        Dropout(),
        Linear(HIDDEN_DIM, HIDDEN_DIM),
        ACTIVATION(),
        BatchNorm1d(HIDDEN_DIM),
        Dropout(),
        Linear(HIDDEN_DIM, X_DIM),
        Sigmoid(),
    ).cuda()

    return G


def create_discriminator() -> Sequential:
    D = Sequential(
        Linear(X_DIM, HIDDEN_DIM),
        ACTIVATION(),
        Dropout(),
        Linear(HIDDEN_DIM, HIDDEN_DIM),
        ACTIVATION(),
        Dropout(),
        Linear(HIDDEN_DIM, HIDDEN_DIM),
        ACTIVATION(),
        Dropout(),
        Linear(HIDDEN_DIM, 1),
    ).cuda()

    return D


G = (
    torch.load("data/models/generator")
    if os.path.exists("data/models/generator") and not args.new
    else create_generator()
)

D = (
    torch.load("data/models/discriminator")
    if os.path.exists("data/models/discriminator") and not args.new
    else create_discriminator()
)


def reset_grads() -> None:
    G.zero_grad()
    D.zero_grad()


# reducing discriminator learning rate seems to help, at least based on one experiment
G_optim, D_optim = Adam(G.parameters()), Adam(D.parameters(), lr=0.0002)

if __name__ == "__main__":

    plt.ioff()  # avoid error when running with nohup

    for iteration in range(args.iterations):
        #
        for _ in range(DISCRIM_STEPS):
            # Train discriminator extra
            z = Variable(torch.randn(BATCH_SIZE, Z_DIM).pin_memory()).cuda()
            X = Variable(
                torch.from_numpy(dataset.next_batch(BATCH_SIZE)[0]).pin_memory()
            ).cuda()

            D_loss = (torch.mean((D(X) - 1) ** 2 + D(G(z)) ** 2)) / 2

            D_loss.backward()
            D_optim.step()
            reset_grads()

        # train generator
        z = Variable(torch.randn(BATCH_SIZE, Z_DIM).pin_memory()).cuda()

        G_loss = torch.mean((D(G(z)) - 1) ** 2) / 2

        G_loss.backward()
        G_optim.step()
        reset_grads()

        # Print and plot every now and then
        if iteration % args.log_every == 0 and iteration != 0:
            print(
                f"i: {iteration}",
                f"D: {D_loss.data[0] : .4}",
                f"G: {G_loss.data[0] : .4}",
                72 * "-",  # Separator for each print summary
                sep="\n",
            )

            samples = G(z).cpu().data.numpy()[:16]

            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis("off")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect("equal")
                # Note to self: Need `imshow` or else plot is just white background.
                plt.imshow(sample.reshape(28, 28), cmap="Greys_r")

            if not os.path.exists("imgs/"):
                os.makedirs("imgs/")

            plt.savefig(f"imgs/{iteration}.png", bbox_inches="tight")
            plt.close(fig)

        if args.save and iteration % args.save_every == 0 and iteration != 0:
            torch.save(G, f"data/models/generator-h{HIDDEN_DIM}-z{Z_DIM}")
            torch.save(D, f"data/models/discriminator-h{HIDDEN_DIM}-z{Z_DIM}")
