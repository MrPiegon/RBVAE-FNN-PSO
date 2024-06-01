import sys
import torch
import random
import numpy as np
import pandas as pd
from vae import vae_models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data.molecule_iterator import SmileBucketIterator


class Soludata(Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label

    def __getitem__(self, index):
        b = self.label[index, :]
        return self.data[index, :], b

    def __len__(self):
        return self.data.shape[0]


class SoludataX(Dataset):
    def __init__(self, data):
        self.X = data

    def __getitem__(self, index):
        return self.X[index, :]

    def __len__(self):
        return self.X.shape[0]


def encode_stack(args, train_mode, latent_dim=256, data_save=False, standarder_return=False):
    # 1. Read Original Model Data, SMILES & Vocab
    MD = pd.read_excel("../data/ModelData/CO2ModelData.xlsx", sheet_name='ModelData')
    ModelData = MD.values
    t, p, y, label = ModelData[:, -4], ModelData[:, -3], ModelData[:, -2], ModelData[:, -1]
    Cation_SMILES = MD['Canonical Cation SMILES']
    Anion_SMILES = MD['Canonical Anion SMILES']
    cation_smi = ModelData[:, 1]
    anion_smi = ModelData[:, 2]
    cation_vocab_file = '../data/Smiles&Vocab/gz/split/cation_vocab'
    anion_vocab_file = '../data/Smiles&Vocab/gz/split/new_anion_vocab'

    # 2. Build SMILES iterator for VAE(vocab_size, padding_idx, sos_idx, unk_idx)
    cation_smi_iterator = SmileBucketIterator(vocab_file=cation_vocab_file, batch_size=512,
                                              train_data_file=None, valid_data_file=Cation_SMILES,
                                              file_read=False, load_vocab=True, training=args.vae_train)
    cation_test_bucket_iter = cation_smi_iterator.valid_bucket_iter_nosort()

    anion_smi_iterator = SmileBucketIterator(vocab_file=anion_vocab_file, batch_size=512,
                                             train_data_file=None, valid_data_file=Anion_SMILES,
                                             file_read=False, load_vocab=True, training=args.vae_train)
    anion_test_bucket_iter = anion_smi_iterator.valid_bucket_iter_nosort()

    # print("Build SMILES iterator complete.")
    cation_vocab_size, anion_vocab_size = cation_smi_iterator.vocab_size, anion_smi_iterator.vocab_size
    cation_padding_idx, anion_padding_idx = cation_smi_iterator.padding_idx, anion_smi_iterator.padding_idx
    cation_sos_idx, anion_sos_idx = cation_smi_iterator.sos_idx, anion_smi_iterator.sos_idx
    cation_unk_idx, anion_unk_idx = cation_smi_iterator.unk_idx, anion_smi_iterator.unk_idx
    cation_vocab, anion_vocab = cation_smi_iterator.get_vocab(), anion_smi_iterator.get_vocab()

    cation_state = torch.load(args.cation_vae_dir)
    cation_vae = vae_models.Vae(cation_vocab, cation_vocab_size, args.embedding_size, args.vae_dropout,
                                cation_padding_idx, cation_sos_idx, cation_unk_idx, args.vae_max_len, args.vae_n_layers,
                                args.hidden_size, bidirectional=args.enc_bidir, latent_size=latent_dim,
                                partialsmiles=args.partialsmiles).cuda()
    cation_vae.load_state_dict(cation_state['vae'])

    anion_state = torch.load(args.anion_vae_dir)
    anion_vae = vae_models.Vae(anion_vocab, anion_vocab_size, args.embedding_size, args.vae_dropout,
                               anion_padding_idx, anion_sos_idx, anion_unk_idx, args.vae_max_len, args.vae_n_layers,
                               args.hidden_size, bidirectional=args.enc_bidir, latent_size=latent_dim,
                               partialsmiles=args.partialsmiles).cuda()
    anion_vae.load_state_dict(anion_state['vae'])
    print("SMILES iterator & VAEs has been built, Encoding SMILES...")

    # 3. Encode SMILES
    def encode(vae, latent_size, bucket_iter, epsilon_std=1.0):
        vae.eval()
        latent = np.zeros([0, latent_size])
        for batch in bucket_iter:
            x = batch.smile[0]
            _, _, _, z = vae.encoder_sample(x, epsilon_std=epsilon_std)  # list(bs, latent_size)
            latent = np.vstack((latent, z.cpu().detach().numpy()))
        return latent

    cation_latent = encode(cation_vae, latent_dim, cation_test_bucket_iter)  # ndarray(dpn, lat_dim)
    anion_latent = encode(anion_vae, latent_dim, anion_test_bucket_iter)
    # 4. Data Transform
    if train_mode == 'DataExtracting':
        print("Encoding Complete.Extracting Data without scaling...")
        data = np.hstack(
            (cation_latent, anion_latent, t.reshape(-1, 1), p.reshape(-1, 1), y.reshape(-1, 1), label.reshape(-1, 1)))
        torch.save(data, 'train_data_' + str(latent_dim) + '_non-scaled.pkl')
        sys.exit()
    print("Encoding Complete.Scaling Data...")
    scaler_T = StandardScaler()
    scaler_P = StandardScaler()
    if standarder_return:
        scaler_T.fit(t.reshape(-1, 1))
        scaler_P.fit(p.reshape(-1, 1))
        return scaler_T, scaler_P
    T = scaler_T.fit_transform(t.reshape(-1, 1))
    P = scaler_P.fit_transform(p.reshape(-1, 1))
    # y = np.log(y)
    data = np.hstack((cation_smi.reshape(-1, 1), anion_smi.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1),
                      cation_latent, anion_latent, T, P, y.reshape(-1, 1), label.reshape(-1, 1))) #(dpn, la*2+3)
    if data_save:
        torch.save(data, args.data_save_dir)
    return data


def bulid_loader(data, mode, save=False, data_return=False):
    if mode == 'Random':
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=114514, shuffle=True)
        train_data_loader, test_data_loader = build_split_loader(train_data, test_data)
        if save:
            torch.save(train_data_loader, 'mlp_train_loader_256' + mode + '_3.pkl')
            torch.save(test_data_loader, 'mlp_test_loader_256' + mode + '_3.pkl')
        else:
            if data_return:
                return train_data, test_data, train_data_loader, test_data_loader
            else:
                return train_data_loader, test_data_loader
    if mode == 'CrossValidation':
        train_data_loader, test_data_loader = build_split_loader(train_data, test_data)
        return train_data_loader, test_data_loader
    if mode == 'IonCrossValidation':
        loader_dict = ion_cross_fold(data, k=10, seed=1919)
        torch.save(loader_dict, 'mlp_DataLoader_256_' + mode + '.pkl')
    if mode == 'WholeValidation':
        valid_data_loader = build_full_loader(data)
        if save:
            torch.save(valid_data_loader, 'mlp_train_loader_256_' + mode + '.pkl')
        else:
            if data_return:
                return data, valid_data_loader
            else:
                return valid_data_loader
    if mode == 'whole_non_loader':
        torch.save(data, 'mlp_256_' + mode + '.pkl')


def ion_cross_fold(data, k, seed, write=False):
    # Build DataFrame for filter
    name_columns = ['T', 'P', 'y', 'label']
    columns = []
    for index in range(1, 513):
        columns.append(str(index))
    columns.extend(name_columns)
    df = pd.DataFrame(data, columns=columns)
    random.seed(seed)
    # getting index for cv
    label_pool = label_list = np.unique(df['label'].values)
    fold_size = len(label_list) // k
    if write:
        writer = pd.ExcelWriter('valid_Seed1919.xlsx', mode="a", engine="openpyxl")
    loader_dict = {}
    for i in range(k):
        if i == k - 1:  # uncut for Last Fold
            fold_label = label_pool
        else:
            fold_label = random.sample(list(label_pool), fold_size)
        valid_data = df.loc[df['label'].isin(fold_label), :]
        if write:
            sheet_name = 'Sheet' + str(i)
            valid_data.to_excel(writer, sheet_name=sheet_name)
            writer.save()

        def cut(uncut_list, list_to_cut):
            # return cut list
            temp_list = []
            for element in uncut_list:
                if element not in list_to_cut:
                    temp_list.append(element)
            return temp_list

        # create train label list & data
        train_label_list = cut(label_list, fold_label)
        train_data = df.loc[df['label'].isin(train_label_list), :]
        # changing label pool for extracting next fold label
        label_pool = cut(label_pool, fold_label)
        train_data, valid_data = train_data.values, valid_data.values
        train_data_loader, test_data_loader = build_split_loader(train_data, valid_data)
        loader_dict[i] = [train_data_loader, test_data_loader]
    return loader_dict


def build_split_loader(train_data, test_data, batch_size=150):
    train_X, test_X = train_data[:, 4:-2], test_data[:, 4:-2]
    train_Y, test_Y = train_data[:, -2], test_data[:, -2]
    train_X, test_X = train_X.astype("float32"), test_X.astype("float32")
    train_Y, test_Y = train_Y.astype("float32"), test_Y.astype("float32")
    train_dataset = Soludata(torch.tensor(train_X, dtype=torch.float32, device="cuda:0"),
                             torch.tensor(train_Y.reshape(-1, 1), dtype=torch.float32, device="cuda:0"))
    test_dataset = Soludata(torch.tensor(test_X, dtype=torch.float32, device="cuda:0"),
                            torch.tensor(test_Y.reshape(-1, 1), dtype=torch.float32, device="cuda:0"))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader


def build_full_loader(data):
    bs = data.shape[0]
    X, Y = data[:, 4:-2], data[:, -2]
    X, Y = X.astype("float32"), Y.astype("float32")
    train_dataset = Soludata(torch.tensor(X, dtype=torch.float32, device="cuda:0"),
                             torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, device="cuda:0"))
    train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    return train_data_loader


def build_xyloader(X, y, batch_size=150):
    train_X, train_Y = X.astype("float32"), y.astype("float32")
    train_dataset = Soludata(torch.tensor(train_X, dtype=torch.float32, device="cuda:0"),
                             torch.tensor(train_Y.reshape(-1, 1), dtype=torch.float32, device="cuda:0"))
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def build_loader_X(X, batch_size=150):
    X = X.astype("float32")
    dataset = SoludataX(torch.tensor(X, dtype=torch.float32, device="cuda:0"))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


