import math
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle

import pandas as pd

from gfo import GradientFreeOptimization


class Block:
    def __init__(
            self,
            scheme,
            gfo,
            dims,
            block_file=None,
            save_dir=None,
            arch=None,
            dataset=None,
            **kwargs,
    ):

        self.scheme = scheme
        self.gfo = gfo
        self.dir = save_dir
        self.file = block_file
        self.block_file = block_file
        self.dims = dims
        self.blocked_dims = None
        self.arch = arch
        self.dataset = dataset
        self.blocker = None
        self.unblocker = None
        self.generator = None
        if "random" in scheme:
            self.blocker = self.randomized_blocker
            self.unblocker = self.randomized_unblocker

        elif "1bin" in scheme:
            self.blocker = self.optimized_blocker
            self.unblocker = self.optimized_unblocker

        else:
            raise ValueError("The block scheme is not recognized!")
        self.scheme = scheme
        # block_mask = self.load_mask()
        self.blocked_dims = len(self.load_mask())

    def load_mask(self, path=None):
        if path == None:
            path = self.file
        blocks_mask = None
        with open(path, "rb") as f:
            blocks_mask = pickle.load(f)
        return blocks_mask

    def optimized_unblocker(self, pop_blocked, blocks_mask=None):
        if blocks_mask == None:
            blocks_mask = self.load_mask()
        blocked_dims = len(blocks_mask)
        pop_unblocked = np.ones((pop_blocked.shape[0], self.dims))
        for i_p in range(pop_blocked.shape[0]):
            for i in range(blocked_dims):
                # print(blocks_mask[i].shape)
                pop_unblocked[i_p, blocks_mask[i]] *= pop_blocked[i_p, i]
        return pop_unblocked

    def randomized_unblocker(self, pop_blocked):
        blocks_mask = self.load_mask()
        block_size = int(math.ceil(self.dims / self.blocked_dims))
        pop_unblocked = np.ones(
            (
                len(pop_blocked),
                self.dims + ((block_size - (self.dims % block_size))),
            )
        )

        for i in range(self.blocked_dims):
            pop_unblocked[:, blocks_mask[i, :]] *= pop_blocked[:, i]

        return pop_unblocked[:, : self.dims]

    def randomized_blocker(self, pop):
        blocks_mask = self.load_mask()
        dim_to_block = 0
        return pop[:, blocks_mask[:, dim_to_block]].copy()

    def optimized_blocker(self, pop=None, blocks_mask=None):
        if blocks_mask == None:
            blocks_mask = self.load_mask()
        blocked_dims = len(blocks_mask)
        params_blocked = np.zeros((pop.shape[0], blocked_dims))
        for i_p in range(pop.shape[0]):
            for i in range(blocked_dims):
                block_params = pop[i_p, blocks_mask[i]]
                if len(block_params) != 0:
                    params_blocked[i_p, i] = np.mean(block_params)

        return params_blocked