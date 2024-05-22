from base_classes.trainer import TrainerBase, TrainerArgParser
from vae.SAscore import calculateScore
from rdkit.rdBase import BlockLogs
import matplotlib.pyplot as plt
import methods
from rdkit import Chem
import seaborn as sb
import numpy as np
import shutil
import torch
import time


class VAETrainer(TrainerBase):
    def __init__(self, args, vocab, vae, enc_optimizer, dec_optimizer, scheduler, train_bucket_iter, test_bucket_iter):
        super(VAETrainer, self).__init__(args.logdir, args.tag)
        # log
        # self.logger.info(args)
        # Properties for using
        self.vocab = vocab
        self.vae = vae
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self.scheduler = scheduler
        self.train_bucket_iter = train_bucket_iter
        self.test_bucket_iter = test_bucket_iter
        self.args = args
        if args.training:
            print('Train_data_batches length:', len(list(train_bucket_iter)))
        self.latent_size = args.latent_size
        self.max_len = args.max_len
        self.num_epochs = args.num_epochs
        self.grad_clip = args.grad_clip
        self.total_cnt = 0
        self.epoch = 0
        # loss相关
        self.loss_weight = None
        """
        if args.weighted_loss:
            print('Train with weighted loss!!!')
            self.loss_weight = []
            for i in range(self.vae.vocab_size):
                token = self.vocab.itos[i]
                if token in weight:
                    self.loss_weight.append(weight[token])
                else:
                    self.loss_weight.append(1.)
        """
        self.loss_function = methods.VAELoss(loss_weight=self.loss_weight)

    def run_epoch(self, epsilon_std=1.0):
        cnt = total_xent_loss = total_kl_loss = total_loss = 0
        self.vae.train()
        for batch in self.train_bucket_iter:
            kl_weight = methods.kl_anneal_function('linear', step=self.total_cnt, k1=0.1, k2=0.2, max_value=0.1, x0=100000)
            self.tensorboard.add_scalar('train/kl_weight', kl_weight, self.total_cnt)
            self.total_cnt += 1
            x = batch.smile[0]
            lens = batch.smile[1]
            # for target, remove the first <sos> token
            x_target = x[:, 1:].cuda().detach().long()

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            x_hat, z_mean, z_log_var, z = self.vae(x, epsilon_std=epsilon_std)

            xent_loss, kl_loss = self.loss_function.forward(x_target, x_hat, z_mean, z_log_var)
            loss = (xent_loss + kl_weight * kl_loss)
            total_loss += loss.cpu().detach().numpy()
            loss = loss / x.size(0)
            loss.backward()

            # Preventing grad explosion(放在backward和step之间)
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip)

            self.enc_optimizer.step()
            self.dec_optimizer.step()

            total_xent_loss += xent_loss.cpu().detach().numpy()
            total_kl_loss += kl_loss.cpu().detach().numpy()
            cnt += x.size(0)

        return total_loss / cnt, total_xent_loss / cnt, total_kl_loss / cnt

    def run_epoch_train(self):
        total_xent_loss = 0
        cnt = 0
        self.vae.train()
        for batch in self.test_bucket_iter:
            x = batch.smile[0]
            # for target, remove the first <sos> token
            x_target = x[:, 1:].cuda().detach().long()

            x_hat, z_mean, z_log_var, _ = self.vae(x, epsilon_std=1.0)
            xent_loss, _ = self.loss_function.forward(x_target, x_hat, z_mean, z_log_var)
            total_xent_loss += xent_loss.cpu().detach().numpy()
            cnt += x.size(0)

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

        self.tensorboard.add_scalar('train/loss_test_data', total_xent_loss / cnt, self.epoch)

    def encode(self, latent_path, epsilon_std=1.0):
        self.vae.eval()
        latent = np.zeros([0, self.args.latent_size])
        for batch in self.test_bucket_iter:
            x = batch.smile[0]
            _, _, _, z = self.vae.encoder_sample(x, epsilon_std=epsilon_std)[-1].cpu().detach().numpy()  # list(bs, latent_size)
            latent = np.vstack((latent, z))
        np.savetxt(latent_path, latent)

    def validate(self, epsilon_std=1.0):
        total_data = reconstructed = reconstructed_valid = 0
        total_bits = reconstructed_bits = 0
        total_mutual_info = 0
        total_loss = 0
        cnt_total = 0
        miss_pred = []
        self.vae.eval()
        for batch in self.test_bucket_iter:
            x = batch.smile[0]
            lens = batch.smile[1]
            # calculate mutual information I(x, z)
            mutual_info = self.vae.calc_mi(x)
            total_mutual_info += mutual_info * x.size(0)

            z = self.vae.encoder_sample(x, epsilon_std=epsilon_std)[-1]
            smiles_pred, index_pred = self.vae.inference(latent=z, max_len=self.max_len)

            # remove start token
            x = x[:, 1:]
            index_pred = index_pred[:, :x.size(1)]
            if index_pred.size(1) < x.size(1):
                padding = torch.LongTensor(np.ones((x.size(0), x.size(1) - index_pred.size(1))))
                index_pred = torch.cat((index_pred, padding), 1)
            # calculate the bit-level accuracy
            reconstructed_bits += torch.sum(x == index_pred).cpu().detach().item()
            total_bits += x.numel()

            total_data += x.size(0)
            for i in range(x.size(0)):
                x_k = x[i].cpu().numpy().tolist()
                x_k = x_k[:(lens[i] - 1)]
                smi_k = [self.vocab.itos[p] for p in x_k]
                smi_k = "".join(smi_k).strip()
                reconstructed += (smi_k == smiles_pred[i])
                reconstructed_valid += len(smiles_pred[i]) > 0
                if smi_k != smiles_pred[i]:
                    miss_pred.append([smi_k, smiles_pred[i]])

            # logits_t = logits_t[:, :x.size(1), :]
            # xent_loss, _ = self.loss_function.forward(x, logits_t, z, z)
            # total_loss += xent_loss.detach().cpu().numpy()
            cnt_total += x.size(0)

        reconstructed_valid = reconstructed_valid / cnt_total
        with open('miss_pred.txt', 'w') as f:
            for line in miss_pred:
                f.write(line[0] + '\n')
                f.write(line[1] + '\n')

        self.tensorboard.add_scalar('test/mutual_info', total_mutual_info / cnt_total, self.epoch)
        # self.tensorboard.add_scalar('test/loss', total_loss / cnt_total, self.epoch)

        # calculate validity
        n_samples = 2500
        cnt_valid = 0
        cnt_total = 0
        block = BlockLogs()
        for _ in range(40):
            cnt_total += n_samples
            smiles_pred = self.sample_prior(n_samples=n_samples, index_return=False, max_len=self.max_len)
            for smi in smiles_pred:
                if len(smi) > 1 and Chem.MolFromSmiles(smi) is not None:
                    cnt_valid += 1
        del block
        self.tensorboard.add_scalar('test/bits_recon_acc', reconstructed_bits / total_bits, self.epoch)
        self.logger.info('reconstructed_bits_acc: {:.4f}'.format(reconstructed_bits / total_bits))
        self.logger.info('reconstructed_valid: {:.4f}'.format(reconstructed_valid))
        self.tensorboard.add_scalar('test/validity', cnt_valid / cnt_total, self.epoch)
        self.logger.info('validity: {:.4f}'.format(cnt_valid / cnt_total))
        return reconstructed / total_data

    def validate_rec(self, epsilon_std=1.0):
        total_data = reconstructed = reconstructed_valid = 0
        total_bits = reconstructed_bits = 0
        # total_mutual_info = 0
        cnt_total = 0
        self.vae.eval()
        for batch in self.test_bucket_iter:
            x = batch.smile[0]
            lens = batch.smile[1]
            # calculate mutual information I(x, z)
            # mutual_info = self.vae.calc_mi(x)
            # total_mutual_info += mutual_info * x.size(0)

            z = self.vae.encoder_sample(x, epsilon_std=epsilon_std)[-1]
            smiles_pred, index_pred = self.vae.inference(latent=z, max_len=self.max_len)

            # remove start token
            x = x[:, 1:]
            index_pred = index_pred[:, :x.size(1)]
            if index_pred.size(1) < x.size(1):
                padding = torch.LongTensor(np.ones((x.size(0), x.size(1) - index_pred.size(1))))
                index_pred = torch.cat((index_pred, padding), 1)
            # calculate the bit-level accuracy
            reconstructed_bits += torch.sum(x == index_pred).cpu().detach().item()
            total_bits += x.numel()

            total_data += x.size(0)
            for i in range(x.size(0)):
                x_k = x[i].cpu().numpy().tolist()
                x_k = x_k[:(lens[i] - 1)]
                smi_k = [self.vocab.itos[p] for p in x_k]
                smi_k = "".join(smi_k).strip()
                reconstructed += (smi_k == smiles_pred[i])

            # logits_t = logits_t[:, :x.size(1), :]
            # xent_loss, _ = self.loss_function.forward(x, logits_t, z, z)
            # total_loss += xent_loss.detach().cpu().numpy()
            cnt_total += x.size(0)

        reconstructed_valid = reconstructed_valid / cnt_total

        # self.tensorboard.add_scalar('test/mutual_info', total_mutual_info / cnt_total, self.epoch)
        # self.tensorboard.add_scalar('test/loss', total_loss / cnt_total, self.epoch)

        # self.tensorboard.add_scalar('test/bits_recon_acc', reconstructed_bits / total_bits, self.epoch)
        self.logger.info('reconstructed_bits_acc: {:.4f}'.format(reconstructed_bits / total_bits))
        self.logger.info('reconstructed_acc: {:.4f}'.format(reconstructed / total_data))
        # self.logger.info('reconstructed_valid: {:.4f}'.format(reconstructed_valid))

    def vuns_test(self, file_path):
        """ 0. Sample """
        block = BlockLogs()
        n_samples = 2500
        cnt_valid = 0
        cnt_total = 0
        cnt_kind = 0
        cnt_simsmi = 0
        sas_list = []
        with open(file_path, 'r') as f:
            mol_strs = f.read().strip().split('\n')
            mol_strs = [mol.replace(' ', '') for mol in mol_strs]
        train_set = set(mol_strs)
        for step in range(40):
            cnt_total += n_samples
            smiles_pred, index_pred = self.sample_prior(n_samples=n_samples, index_return=True, max_len=self.max_len)
            for smi in smiles_pred:
                """ 1. Validity """
                mol = Chem.MolFromSmiles(smi)
                if len(smi) > 1 and mol is not None:
                    cnt_valid += 1
            """ 2. Uniqueness """
            unique_index = np.unique(index_pred, axis=0)
            cnt_kind += unique_index.shape[0]
            """ 3. Novelty """
            test_set = set(smiles_pred)
            cnt_simsmi += len(train_set & test_set)
            """ 4. Synthetic Accessibility Score"""
            print('Calculating SAScore in Step {} '.format(step + 1))
            # sas_step = 0
            for smiles in smiles_pred:
                # sas_step += 1
                # print('SAScore {}'.format(sas_step))
                m = Chem.MolFromSmiles(smiles)
                try:
                    s = calculateScore(m)
                except:
                    continue
                sas_list.append(s)

        del block
        self.logger.info('Validity: {:.4f}'.format(cnt_valid / cnt_total))
        self.logger.info('Uniqueness: {:.4f}'.format(cnt_kind / cnt_total))
        self.logger.info('Novelty: {:.4f}'.format(1 - (cnt_simsmi / cnt_total)))
        sa_score_list = np.array(sas_list)
        plt.figure()
        a = sb.displot(sa_score_list)
        plt.show()

    def save(self, is_best, name):
        state = {'vae': self.vae.state_dict(),
                 'enc_optimizer': self.enc_optimizer.state_dict(),
                 'dec_optimizer': self.dec_optimizer.state_dict()}
        path = self.checkpoint_path(name)
        torch.save(state, path)
        if is_best:
            shutil.copyfile(path, self.checkpoint_path('top'))

    def load(self, step):
        self.load_raw(self.checkpoint_path(step))

    def load_raw(self, path):
        state = torch.load(path)
        self.vae.load_state_dict(state['vae'])

    def train(self):
        results_fmt = ("{} :: {} :: loss {:.3f} xcent {:.3f} kl {:.3f}" + " " * 30).format
        for self.epoch in range(self.num_epochs):
            epsilon_std = 1.0
            loss, xcent_loss, kl_loss = self.run_epoch(epsilon_std=epsilon_std)

            if (self.epoch % 10 == 0 and self.epoch >= 10) or (self.epoch == self.num_epochs - 1):
                self.save(False, self.epoch)

            self.run_epoch_train()
            self.tensorboard.add_scalar('train/loss', loss, self.epoch)
            self.tensorboard.add_scalar('train/xcent_loss', xcent_loss, self.epoch)
            self.tensorboard.add_scalar('train/kl_loss', kl_loss, self.epoch)

            if self.epoch % 10 == 0 and self.epoch >= 10:
            # if self.epoch >= 0:
                recon_acc = self.validate(epsilon_std=1e-6)
                self.tensorboard.add_scalar('test/recon_acc', recon_acc, self.epoch)
                self.logger.info('recon_acc: {:.4f}'.format(recon_acc))

            self.logger.info(results_fmt(time.strftime("%H:%M:%S"), self.epoch, loss, xcent_loss, kl_loss))

    def sample_prior(self, n_samples, index_return=False, z_samples=None, max_len=None):
        if z_samples is None:
            z_samples = torch.FloatTensor(n_samples, self.latent_size).normal_(0, 1)
        smiles_pred, index_pred = self.vae.inference(latent=z_samples.cuda(), max_len=max_len)
        if index_return:
            return smiles_pred, index_pred
        else:
            return smiles_pred



class VAEArgParser(TrainerArgParser):
    def __init__(self):
        super(VAEArgParser, self).__init__()
        self.add_argument('--logdir', type=str, default='../log/simvo_vae')
        self.add_argument('--test_mode', action='store_true')
        self.add_argument('--encode_mode', type=bool, default=False)
        self.add_argument('--vae_dropout', type=float, default=0.5)
        self.add_argument('--grad_clip', type=float, default=5.0)
        self.add_argument('--wd', type=float, default=1e-4)
        self.add_argument('--batch_size', type=int, default=250)
        self.add_argument('--generate_samples', action='store_true')
        self.add_argument('--weighted_loss', action='store_true')
        self.add_argument('--enc_bidir', action='store_false')
        self.add_argument('--partialsmiles', action='store_true')
        self.add_argument('--vae_n_layers', type=int, default=2)
        self.add_argument('--hidden_size', type=int, default=512)
        self.add_argument('--latent_size', type=int, default=256)
        self.add_argument('--embedding_size', type=int, default=48)
        self.add_argument('--vae_max_len', type=int, default=120)
        self.add_argument('--num_epochs', type=int, default=100)
        self.add_argument('--smi_load_batch_size', type=int, default=500)
        self.add_argument('--smi_test_size', type=int, default=10000)
