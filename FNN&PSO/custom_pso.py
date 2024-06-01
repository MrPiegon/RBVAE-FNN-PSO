#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987
# @Custom  : Pigeon 2023/6/3
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import tensorboardX
from sko.base import SkoBase
from sko.tools import func_transformer
from base_classes.trainer import TrainerBase
from sklearn.preprocessing import normalize, StandardScaler


class PSO(SkoBase):
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations

    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration


    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    def __init__(self, args, func, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5,
                logging=False, y_hist_record=False, valscore=False):
        # self.func = func_transformer(func)
        self.args = args
        self.func = func
        self.valscore = valscore
        self.w = w  # inertia
        self.max_w = w
        self.min_w = 0.9
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
        self.Y, self.num_log, valid_num_epoch_sum, self.epo_cation_smiles, self.epo_anion_smiles = self.cal_y()
        self.aval_mat = self.num_log
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.pbest_x_cs, self.pbest_x_as = self.epo_cation_smiles.copy(), self.epo_anion_smiles.copy()
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.y_hist = (np.arange(self.dim-2)).reshape(-1, 1)  # gbest_y of every iteration
        self.pbest_y_hist = self.pbest_y
        self.update_gbest()
        self.logging = logging
        # record verbose values
        self.record_mode = y_hist_record
        self.w_list = []
        # self.record_value = {'X': [], 'V': [], 'Y': []}
        self.Y_min_hist = []
        if self.logging:
            loggerbase = TrainerBase(args.logdir, '')
            self.logger, self.tensorboard = loggerbase.logger, loggerbase.tensorboard

    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V
        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.zl, self.zr = self.X[:, 2:int(len(self.X[2:][0]) / 2)+1], self.X[:, int(len(self.X[2:][0]) / 2)+1:]
        self.zl, self.zr = self.normalize(self.zl), self.normalize(self.zr)
        self.t, self.p = self.X[:, 0], self.X[:, 1]
        self.X = np.hstack((self.t.reshape(-1, 1), self.p.reshape(-1, 1), self.zl, self.zr))
        self.Y, self.num_log, valid_epoch_num, cation_smiles, anion_smiles = self.func(self.X)
        return self.Y, self.num_log, valid_epoch_num, cation_smiles, anion_smiles

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_x_cs = np.where(self.pbest_y > self.Y, self.epo_cation_smiles, self.pbest_x_cs)
        self.pbest_x_as = np.where(self.pbest_y > self.Y, self.epo_anion_smiles, self.pbest_x_as)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        min = self.Y.min()
        if self.gbest_y > min:
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def update_w(self, iter):
        self.w = self.max_w - (self.max_w - self.min_w) * (iter / self.max_iter)
        self.w_list.append(self.w)

    def recorder(self):
        if self.record_mode:
            si = np.argsort(self.Y, axis=0)
            sort_y = np.take_along_axis(self.Y, si, axis=0)
            sort_y = -(sort_y + 3000)
            self.y_hist = np.hstack((self.y_hist, sort_y))
        # self.record_value['X'].append(self.X)
        # self.record_value['V'].append(self.V)
        # self.Y_min_hist.append(-(self.Y.min()+3000))

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(1, self.max_iter+1):
            self.update_V()
            self.update_X()
            self.Y, self.num_log, valid_num, self.epo_cation_smiles, self.epo_anion_smiles = self.cal_y()
            self.aval_mat = np.vstack((self.aval_mat, self.num_log))
            self.recorder()
            # self.update_w(iter_num)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
            if self.logging:
                if self.args.psovalscore:
                    self.logger.info("iter: {} gby: {:.4f} valid smiles number: {}".format(iter_num, -(self.gbest_y + 3000), valid_num))
                else:
                    self.logger.info("iter: {} gby: {:.4f} valid smiles number: {}".format(iter_num, -self.gbest_y, valid_num))
                self.tensorboard.add_scalar('valid_num', valid_num, iter_num)
                self.tensorboard.add_scalar('global_best_y', -(self.gbest_y + 3000), iter_num)
            else:
                if self.args.psovalscore:
                    print("iter: {} gby: {:.4f} valid smiles number: {}".format(iter_num, -(self.gbest_y + 3000), valid_num))
                else:
                    print("iter: {} gby: {:.4f} valid smiles number: {}".format(iter_num, -self.gbest_y, valid_num))
            # self.gbest_x_hist = np.hstack((self.gbest_x_hist, self.gbest_x.reshape(1, -1)))
            # self.pbest_y_hist = np.hstack((self.pbest_y_hist, self.pbest_y))

        # sort
        si = np.argsort(self.pbest_y, axis=0)
        sort_x = np.take_along_axis(self.pbest_x, si, axis=0)
        sort_y = np.take_along_axis(self.pbest_y, si, axis=0)
        self.sort_cs = np.take_along_axis(self.pbest_x_cs, si, axis=0)
        self.sort_as = np.take_along_axis(self.pbest_x_as, si, axis=0)
        self.sort_y = sort_y
        self.sort_x = sort_x[:, 2:]
        self.best10x = sort_x[:10, :]
        self.best10y = sort_y[:10, :]
        """
        self.logger.info("similarity num: {}, gby for epoch {}: {:.3f}, time:{:.3f}".format(sim_num, epoch, 
        -(pso.gbest_y + 3000), (time_end - time_start) / 60))
        """
        if self.record_mode:
            time_str = time.strftime("%m-%d_%H-%M-%S", time.localtime())
            df = pd.DataFrame(self.y_hist)
            df.to_excel('y_hist_' + time_str + '.xlsx', index=False)
        return self

    def normalize(self, z):
        sc = StandardScaler()
        for i in range(len(z)):
            z[i] = sc.fit_transform(z[i].reshape(-1, 1)).squeeze(1)
        return z

    fit = run

class GBPSO(SkoBase):
    """
    Do GBPSO (Gradient based Particle swarm optimization) algorithm.
    """

    def __init__(self, args, func, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5,
                c3=0.001, logging=False, y_hist_record=False, valscore=False):
        # self.func = func_transformer(func)
        self.args = args
        self.func = func
        self.valscore = valscore
        self.w = w  # inertia
        self.max_w = w
        self.min_w = 0.9
        self.cp, self.cg, self.c3 = c1, c2, c3  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
        self.Y, self.grads_epoch, valid_num_epoch_sum, self.epo_cation_smiles, self.epo_anion_smiles = self.cal_y()
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.pbest_x_cs, self.pbest_x_as = self.epo_cation_smiles.copy(), self.epo_anion_smiles.copy()
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.y_hist = (np.arange(self.dim-2)).reshape(-1, 1)  # gbest_y of every iteration
        self.pbest_y_hist = self.pbest_y
        self.update_gbest()
        self.logging = logging
        # record verbose values
        self.record_mode = y_hist_record
        self.w_list = []
        # self.record_value = {'X': [], 'V': [], 'Y': []}
        self.Y_min_hist = []
        if self.logging:
            loggerbase = TrainerBase(args.logdir, '')
            self.logger, self.tensorboard = loggerbase.logger, loggerbase.tensorboard

    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X) + \
                 self.c3 * self.grads_epoch

    def update_X(self):
        self.X = self.X + self.V
        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.zl, self.zr = self.X[:, 2:int(len(self.X[2:][0]) / 2)+1], self.X[:, int(len(self.X[2:][0]) / 2)+1:]
        self.zl, self.zr = self.normalize(self.zl), self.normalize(self.zr)
        self.t, self.p = self.X[:, 0], self.X[:, 1]
        self.X = np.hstack((self.t.reshape(-1, 1), self.p.reshape(-1, 1), self.zl, self.zr))
        self.Y, self.grads_epoch, valid_epoch_num, cation_smiles, anion_smiles = self.func(self.X, grad_return=self.args.gbpso)
        return self.Y, self.grads_epoch, valid_epoch_num, cation_smiles, anion_smiles

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_x_cs = np.where(self.pbest_y > self.Y, self.epo_cation_smiles, self.pbest_x_cs)
        self.pbest_x_as = np.where(self.pbest_y > self.Y, self.epo_anion_smiles, self.pbest_x_as)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        min = self.Y.min()
        if self.gbest_y > min:
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def update_w(self, iter):
        self.w = self.max_w - (self.max_w - self.min_w) * (iter / self.max_iter)
        self.w_list.append(self.w)

    def recorder(self):
        if self.record_mode:
            si = np.argsort(self.Y, axis=0)
            sort_y = np.take_along_axis(self.Y, si, axis=0)
            sort_y = -(sort_y + 3000)
            self.y_hist = np.hstack((self.y_hist, sort_y))
        # self.record_value['X'].append(self.X)
        # self.record_value['V'].append(self.V)
        # self.Y_min_hist.append(-(self.Y.min()+3000))

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(1, self.max_iter+1):
            self.update_V()
            self.update_X()
            self.Y, self.grads_epoch, valid_num, self.epo_cation_smiles, self.epo_anion_smiles = self.cal_y()
            self.recorder()
            # self.update_w(iter_num)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
            if self.logging:
                if self.args.psovalscore:
                    self.logger.info("iter: {} gby: {:.4f} valid smiles number: {}".format(iter_num, -(self.gbest_y + 3000), valid_num))
                else:
                    self.logger.info("iter: {} gby: {:.4f} valid smiles number: {}".format(iter_num, -self.gbest_y, valid_num))
                self.tensorboard.add_scalar('valid_num', valid_num, iter_num)
                self.tensorboard.add_scalar('global_best_y', -(self.gbest_y + 3000), iter_num)
            else:
                if self.args.psovalscore:
                    print("iter: {} gby: {:.4f} valid smiles number: {}".format(iter_num, -(self.gbest_y + 3000), valid_num))
                else:
                    print("iter: {} gby: {:.4f} valid smiles number: {}".format(iter_num, -self.gbest_y, valid_num))
            # self.gbest_x_hist = np.hstack((self.gbest_x_hist, self.gbest_x.reshape(1, -1)))
            # self.pbest_y_hist = np.hstack((self.pbest_y_hist, self.pbest_y))

        # sort
        si = np.argsort(self.pbest_y, axis=0)
        sort_x = np.take_along_axis(self.pbest_x, si, axis=0)
        sort_y = np.take_along_axis(self.pbest_y, si, axis=0)
        self.sort_cs = np.take_along_axis(self.pbest_x_cs, si, axis=0)
        self.sort_as = np.take_along_axis(self.pbest_x_as, si, axis=0)
        self.sort_y = sort_y
        self.sort_x = sort_x[:, 2:]
        self.best10x = sort_x[:10, :]
        self.best10y = sort_y[:10, :]
        """
        self.logger.info("similarity num: {}, gby for epoch {}: {:.3f}, time:{:.3f}".format(sim_num, epoch, 
        -(pso.gbest_y + 3000), (time_end - time_start) / 60))
        """
        if self.record_mode:
            time_str = time.strftime("%m-%d_%H-%M-%S", time.localtime())
            df = pd.DataFrame(self.y_hist)
            df.to_excel('y_hist_' + time_str + '.xlsx', index=False)
        return self

    def normalize(self, z):
        sc = StandardScaler()
        for i in range(len(z)):
            z[i] = sc.fit_transform(z[i].reshape(-1, 1)).squeeze(1)
        return z

    fit = run