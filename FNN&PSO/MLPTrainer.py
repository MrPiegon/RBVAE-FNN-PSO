import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from vae import vae_models
from Filter import filter_pt
from torchmetrics.functional import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from data.molecule_iterator import SmileBucketIterator
from data.DataProcessing import build_xyloader, build_loader_X
from torch.optim.lr_scheduler import ReduceLROnPlateau
from base_classes.trainer import TrainerBase, TrainerArgParser


class MLPTrainer(TrainerBase):
    def __init__(self, args, latent_dim=256, device="cuda:0", start_logging=True):
        super(MLPTrainer, self).__init__(args.logdir, '', start_logging)
        self.args = args
        self.latent_size = latent_dim
        self.device = device

        self.epoch = 0
        self.num_epoch = args.num_epochs
        self.batch_size = args.mlp_batch_size

        # self.loss_hist = []
        self.MSE = nn.MSELoss(reduction='mean')
        self.MAE = nn.L1Loss(reduction='mean')
        self.start_logging = start_logging
        """
        if self.start_logging:
            self.logger.info(latent_dim)
            self.logger.info(network)
        """

    def fit_val(self, model, train_loader, test_loader, pred_return=False, model_return=False, logging=True, mae_return=False):
        # results_fmt = ("{} :: {} :: train_loss {:.3f} test_loss {:.3f} test_r2 {:.3f} " + " " * 30).format
        results_fmt = ("{} :: {} :: train_loss {:.3f} test_loss {:.3f} " + " " * 30).format
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=logging)
        for self.epoch in range(self.num_epoch):
            model.train()
            train_total_loss = [0]
            train_batch_num = 0
            for batch_x, batch_y in train_loader:  # batch_x:tensor(bs, la+2)
                z, t, p = batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]
                optimizer.zero_grad()
                y_pred = model.forward(z, t, p)
                # loss = self.MSE(y_pred, batch_y)
                train_loss = self.MAE(y_pred, batch_y)
                train_total_loss[-1] += train_loss.item()
                train_loss.backward()
                optimizer.step()
                train_batch_num += 1
            epoch_train_loss = train_total_loss[0] / train_batch_num
            model.eval()
            test_total_loss = [0]
            test_batch_num = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    z, t, p = batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]
                    y_pred = model.forward(z, t, p)
                    # test_loss = self.MSE(y_pred, batch_y).item()
                    test_loss = self.MAE(y_pred, batch_y).item()
                    test_total_loss[-1] += test_loss
                    test_batch_num += 1
                    # test_r2 = r2_score(y_pred, batch_y).item()
            epoch_test_loss = test_total_loss[0] / test_batch_num
            if logging:
                self.tensorboard.add_scalar('train/MAE_loss', epoch_train_loss, self.epoch)
                self.tensorboard.add_scalar('test/MAE_loss', epoch_test_loss, self.epoch)
                # self.tensorboard.add_scalar('test/r2', test_r2, self.epoch)
                # self.logger.info(results_fmt(time.strftime("%H:%M:%S"), self.epoch, train_loss[-1], test_loss, test_r2))
                self.logger.info(results_fmt(time.strftime("%H:%M:%S"), self.epoch, epoch_train_loss, epoch_test_loss))
            scheduler.step(epoch_test_loss)
        if pred_return:
            y_pred = np.zeros((0, 1))
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in train_loader:
                    z, t, p = batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]
                    train_y_pred = model.forward(z, t, p)
                    y_pred = np.vstack((y_pred, train_y_pred.cpu().numpy().reshape(-1, 1)))
                for batch_x, batch_y in test_loader:
                    z, t, p = batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]
                    test_y_pred = model.forward(z, t, p)
                    y_pred = np.vstack((y_pred, test_y_pred.cpu().numpy().reshape(-1, 1)))
                if mae_return:
                    return y_pred, train_loss[-1], test_loss
                else:
                    if model_return :
                        return y_pred, model
                    return y_pred
        else:
            if model_return:
                return model


    def fold_fit_val(self, index, model, train_loader, test_loader):
        "No significant changes except for logging&optimizer/scheduler initializing."
        results_fmt = ("Fold {} :: {} :: {} :: train_loss {:.3f} test_loss {:.3f} test_r2 {:.3f} " + " " * 30).format
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
        for self.epoch in range(self.num_epoch):
            model.train()
            for batch_x, batch_y in train_loader:  # batch_x:tensor(bs, la+2)
                train_loss = [0]
                z, t, p = batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]
                optimizer.zero_grad()
                y_pred = model.forward(z, t, p)
                loss = self.criterion(y_pred, batch_y)
                train_loss[-1] += loss.item()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    z, t, p = batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]
                    y_pred = model.forward(z, t, p)
                    test_loss = self.criterion(y_pred, batch_y).item()
                    test_r2 = r2_score(y_pred, batch_y).item()
            if self.start_logging:
                self.tensorboard.add_scalar('Fold_' + str(index) + '_train/MSE_loss', train_loss[-1], self.epoch)
                self.tensorboard.add_scalar('Fold_' + str(index) + '_test/MSE_loss', test_loss, self.epoch)
                self.tensorboard.add_scalar('Fold_' + str(index) + '_test/r2', test_r2, self.epoch)
                self.logger.info(
            results_fmt(time.strftime("%H:%M:%S"), index, self.epoch, train_loss[-1], test_loss, test_r2))
            scheduler.step(test_loss)
        return test_r2

    def save_fit(self, model, train_loader):
        results_fmt = ("{} :: {} :: train_loss {:.3f}" + " " * 30).format
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
        for self.epoch in range(self.num_epoch):
            model.train()
            for batch_x, batch_y in train_loader:  # batch_x:tensor(bs, la+2)
                train_loss = [0]
                z, t, p = batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]
                optimizer.zero_grad()
                y_pred = model.forward(z, t, p)
                loss = self.criterion(y_pred, batch_y)
                train_loss[-1] += loss.item()
                loss.backward()
                optimizer.step()
            self.tensorboard.add_scalar('train/MSE_loss', train_loss[-1], self.epoch)
            self.logger.info(results_fmt(time.strftime("%H:%M:%S"), self.epoch, train_loss[-1]))
            scheduler.step(train_loss[-1])
        return model

    def val(self, model, valid_data_loader, logging):
        results_fmt = ("{} :: valid_loss {:.3f} valid_r2 {:.3f} " + " " * 30).format
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in valid_data_loader:
                z, t, p = batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]
                y_pred = model.forward(z, t, p)
                valid_loss = self.MAE(y_pred, batch_y).item()
                valid_r2 = r2_score(y_pred, batch_y).item()
        if logging:
            self.logger.info(results_fmt(time.strftime("%H:%M:%S"), valid_loss, valid_r2))

    def custom_val(self, model, train_loader, test_loader):
        """ Validate Process """
        train_whole_y = np.concatenate([batch_y.cpu().numpy() for batch_x, batch_y in train_loader])
        test_whole_y = np.concatenate([batch_y.cpu().numpy() for batch_x, batch_y in test_loader])
        train_pred_y = np.concatenate(
            [model.forward(batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]).cpu().numpy()
             for batch_x, batch_y in train_loader])
        test_pred_y = np.concatenate(
            [model.forward(batch_x[:, :self.latent_size * 2], batch_x[:, -2], batch_x[:, -1]).cpu().numpy()
             for batch_x, batch_y in test_loader])
        train_error = train_whole_y - train_pred_y
        test_error = test_whole_y - test_pred_y
        return train_whole_y.flatten(), test_whole_y.flatten(), train_pred_y.flatten(), \
               test_pred_y.flatten(), train_error.flatten(), test_error.flatten()


class VAE_MLP_Trainer(TrainerBase):
    def __init__(self, args, vae, network, latent_dim, device):
        super(VAE_MLP_Trainer, self).__init__(args.logdir, '')
        self.args = args
        self.latent_size = latent_dim
        self.device = device

        self.epoch = 0
        self.num_epoch = args.num_epochs
        self.batch_size = args.mlp_batch_size

        # self.loss_hist = []
        self.MLP_criterion = nn.MSELoss(reduction='mean')
        # self.logger.info(latent_dim)
        # self.logger.info(network)

    def load_vae(self):
        Cation_SMILES = '../data/Smiles&Vocab/Solu/SoluCation.txt'  # Not important for loading
        Anion_SMILES = '../data/Smiles&Vocab/Solu/SoluAnion.txt'  # Not important for loading
        cation_vocab_file = '../data/Smiles&Vocab/gz/split/cation_vocab'
        anion_vocab_file = '../data/Smiles&Vocab/gz/split/new_anion_vocab'
        """Creating smi_iterator just for vocab,padding/sos/unk idx. Iterator isn't important."""
        cation_smi_iterator = SmileBucketIterator(Cation_SMILES, cation_vocab_file, self.args.smi_load_batch_size,
                                                  file_read=True,
                                                  load_vocab=True, training=self.args.vae_train)
        anion_smi_iterator = SmileBucketIterator(Anion_SMILES, anion_vocab_file, self.args.smi_load_batch_size,
                                                 file_read=True,
                                                 load_vocab=True, training=self.args.vae_train)

        cation_vocab_size, anion_vocab_size = cation_smi_iterator.vocab_size, anion_smi_iterator.vocab_size
        cation_padding_idx, anion_padding_idx = cation_smi_iterator.padding_idx, anion_smi_iterator.padding_idx
        cation_sos_idx, anion_sos_idx = cation_smi_iterator.sos_idx, anion_smi_iterator.sos_idx
        cation_unk_idx, anion_unk_idx = cation_smi_iterator.unk_idx, anion_smi_iterator.unk_idx
        cation_vocab, anion_vocab = cation_smi_iterator.get_vocab(), anion_smi_iterator.get_vocab()

        state = torch.load(self.args.cation_vae_dir)
        cation_vae = vae_models.Vae(cation_vocab, cation_vocab_size, self.args.vae_embedding_size,
                                    self.args.vae_dropout,
                                    cation_padding_idx, cation_sos_idx, cation_unk_idx, self.args.vae_max_len,
                                    self.args.vae_n_layers,
                                    self.args.vae_layer_size, bidirectional=self.args.enc_bidir,
                                    latent_size=self.latent_size,
                                    partialsmiles=self.args.partialsmiles).cuda()
        cation_vae.load_state_dict(state['vae'])

        state = torch.load(self.args.anion_vae_dir)
        anion_vae = vae_models.Vae(anion_vocab, anion_vocab_size, self.args.vae_embedding_size, self.args.vae_dropout,
                                   anion_padding_idx, anion_sos_idx, anion_unk_idx, self.args.vae_max_len,
                                   self.args.vae_n_layers,
                                   self.args.vae_layer_size, bidirectional=self.args.enc_bidir,
                                   latent_size=self.latent_size,
                                   partialsmiles=self.args.partialsmiles).cuda()
        anion_vae.load_state_dict(state['vae'])
        return anion_vae, cation_vae

    def train(self):
        cation_vae, anion_vae = self.load_vae()
        MD = pd.read_excel("../data/ModelData/CO2ModelData.xlsx", sheet_name='ModelData')
        ModelData = MD.values
        t, p, y, label = ModelData[:, -4], ModelData[:, -3], ModelData[:, -2], ModelData[:, -1]


class MLPTrainer_es(TrainerBase, BaseEstimator, RegressorMixin):
    def __init__(self, args=None, latent_dim=256, model=None, start_logging=False):
        # 多父类继承调用https://blog.csdn.net/m0_37220818/article/details/108969947
        TrainerBase.__init__(self, args.logdir, '', start_logging)
        BaseEstimator.__init__(self)
        RegressorMixin.__init__(self)

        self.args = args
        self.latent_dim = latent_dim
        self.model = model
        self.start_logging = start_logging

        self.epoch = 0
        self.num_epoch = args.num_epochs
        self.batch_size = args.mlp_batch_size

        self.criterion = nn.L1Loss(reduction='mean')

    def fit(self, X, y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
        train_loader = build_loader(X, y)
        # train_total_loss = 0
        for self.epoch in range(self.num_epoch):
            self.model.train()
            for batch_x, batch_y in train_loader:  # batch_x:tensor(bs, la+2)
                train_epoch_loss = [0]
                z, t, p = batch_x[:, :self.latent_dim * 2], batch_x[:, -2], batch_x[:, -1]
                optimizer.zero_grad()
                y_pred = self.model.forward(z, t, p)
                loss = self.criterion(y_pred, batch_y)
                train_epoch_loss[-1] += loss.item()
                loss.backward()
                optimizer.step()
            scheduler.step(train_epoch_loss[-1]) #too big?
        """
            train_total_loss += train_epoch_loss[-1]
        
            train_average_loss = train_total_loss / self.num_epoch
        if self.start_logging:
            self.logger.info("train loss: {:.4f}".format(train_average_loss))
        """
        return self

    def predict(self, X):
        # results_fmt = ("{} :: {} :: train_loss {:.3f} test_loss {:.3f} test_r2 {:.3f} " + " " * 30).format
        self.model.eval()
        # valid_loader = build_loader_X(X)
        X = X.astype("float32")
        X = torch.tensor(X, dtype=torch.float32, device="cuda:0")
        with torch.no_grad():
            z, t, p = X[:, :self.latent_dim * 2], X[:, -2], X[:, -1]
            y_pred = self.model.forward(z, t, p).cpu().numpy()
        return y_pred

    def score(self, X, y, sample_weight=None):
        Y = self.predict(X)
        valid_loss = self.criterion(Y, y).item()
        # valid_r2 = r2_score(Y, y).item()
        if self.start_logging:
            self.tensorboard.add_scalar('test/MAE_loss', valid_loss)
            # self.tensorboard.add_scalar('test/r2', valid_r2)
        return valid_loss

    def max_sco(self, y_true, y_pred):
        _, sort_y = filter_pt(self.args, t=self.args.tt, pr=self.args.prt, mlp=self.model)
        max_solu = np.max(sort_y)
        return -max_solu