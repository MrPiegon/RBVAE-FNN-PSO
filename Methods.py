import os
import sys
import time
import rdkit
import torch
import numpy as np
import pandas as pd
import seaborn as sb
from rdkit import Chem
from tqdm import trange
from numpy import random
from vae import vae_models
import matplotlib.cm as cm
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit.rdBase import BlockLogs
from torch.autograd import Variable
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from vae.SAscore import calculateScore
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LinearSegmentedColormap
from data.molecule_iterator import SmileBucketIterator

def encode(args, smi, model, model_name, epsilon_std=1.0, file_read=False):
    if model_name == 'cation_vae':
        vocab_file = '../data/Smiles&Vocab/gz/split/cation_vocab'
    elif model_name == 'anion_vae':
        vocab_file = '../data/Smiles&Vocab/gz/split/new_anion_vocab'
    smi_iterator = SmileBucketIterator(vocab_file, args.smi_load_batch_size,
                                       train_data_file=None, valid_data_file=smi,
                                       random_test=False, file_read=file_read, load_vocab=True, training=False)
    bucket_iter = smi_iterator.valid_bucket_iter_nosort()
    model.eval()
    latent = np.zeros([0, args.latent_size])
    for batch in bucket_iter:
        x = batch.smile[0]
        _, mean, _, _ = model.encoder_sample(x, epsilon_std=epsilon_std)  # list(bs, latent_size)
        latent = np.vstack((latent, mean.cpu().detach().numpy()))
    return latent


def MolToImgFile(z, model, filename_, s_filename_, smiles_return=True):
    try:
        # the shape of z must be [bs, latent]
        z = torch.tensor(z, dtype=torch.float32, device='cuda:0')
        smiles = model.inference(z, max_len=120)[0][0]
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, filename_)
        d = Draw.MolDraw2DSVG(400, 400)
        d.ClearDrawing()
        d.DrawMolecule(mol)
        d.FinishDrawing()
        with open(s_filename_, 'w+') as f:
            f.write(d.GetDrawingText())
        if smiles_return:
            return smiles
    except (AssertionError, ValueError):
        MolToImgFile(z, model, filename_, s_filename_)


def available_test(cation_list, anion_list):
    block = BlockLogs()  # Disable RDKit's logging
    normative_score_list = []
    aval_num = 0
    cation_aval_num = 0
    anion_aval_num = 0
    cation_ion_num = 0
    anion_ion_num = 0
    IL_miss_num = 0
    rd_aval_num = 0
    rdion_aval_num = 0
    for cation_smiles, anion_smiles in zip(cation_list, anion_list):
        cation_aval_bool = 0
        anion_aval_bool = 0
        cation_ion_bool = 0
        anion_ion_bool = 0
        # checks if the two ions have opposite charges and if they are valid.
        # If both conditions are met, it sets the variable normative_score_list to -3000 and aval_bool is set to 1.
        Cation = Chem.MolFromSmiles(cation_smiles)
        Anion = Chem.MolFromSmiles(anion_smiles)
        if type(Cation) == rdkit.Chem.rdchem.Mol:
            cation_aval_num += 1
            cation_aval_bool = 1
        if type(Anion) == rdkit.Chem.rdchem.Mol:
            anion_aval_num += 1
            anion_aval_bool = 1
        if cation_aval_bool * anion_aval_bool:
            rd_aval_num += 1
            if '+' in cation_smiles:
                cation_ion_num += 1
                cation_ion_bool = 1
            if '-' in anion_smiles:
                anion_ion_num += 1
                anion_ion_bool = 1
            if cation_ion_bool * anion_ion_bool:
                rdion_aval_num += 1
            else:
                normative_score_list.append(0)
                continue
            IL_smiles = cation_smiles + '.' + anion_smiles
            IL = Chem.MolFromSmiles(IL_smiles)
            if IL == None:
                normative_score_list.append(0)
                IL_miss_num += 1
            else:
                normative_score_list.append(-3000)
                aval_num += 1
        else:
            normative_score_list.append(0)
    del block

    normative_score_list = np.array(normative_score_list).reshape(-1, 1)
    num_list = [cation_aval_num, anion_aval_num, rd_aval_num, cation_ion_num, anion_ion_num, rdion_aval_num, aval_num]
    num_list = np.array(num_list)
    return normative_score_list, num_list, aval_num


def load_vae(args):
    Cation_SMILES = '../data/Smiles&Vocab/Solu/SoluCation.txt'  # Not important for loading
    Anion_SMILES = '../data/Smiles&Vocab/Solu/SoluAnion.txt'  # Not important for loading
    cation_vocab_file = '../data/Smiles&Vocab/gz/split/cation_vocab'
    anion_vocab_file = '../data/Smiles&Vocab/gz/split/new_anion_vocab'
    """Creating smi_iterator just for vocab,padding/sos/unk idx. Iterator isn't important."""
    cation_smi_iterator = SmileBucketIterator(cation_vocab_file, batch_size=512,
                                              train_data_file=None, valid_data_file=Cation_SMILES,
                                              random_test=False, file_read=True, load_vocab=True, training=False)
    anion_smi_iterator = SmileBucketIterator(anion_vocab_file, batch_size=512,
                                             train_data_file=None, valid_data_file=Anion_SMILES,
                                             random_test=False, file_read=True, load_vocab=True, training=False)

    cation_vocab_size, anion_vocab_size = cation_smi_iterator.vocab_size, anion_smi_iterator.vocab_size
    cation_padding_idx, anion_padding_idx = cation_smi_iterator.padding_idx, anion_smi_iterator.padding_idx
    cation_sos_idx, anion_sos_idx = cation_smi_iterator.sos_idx, anion_smi_iterator.sos_idx
    cation_unk_idx, anion_unk_idx = cation_smi_iterator.unk_idx, anion_smi_iterator.unk_idx
    cation_vocab, anion_vocab = cation_smi_iterator.get_vocab(), anion_smi_iterator.get_vocab()

    state = torch.load(args.cation_vae_dir)
    cation_vae = vae_models.Vae(cation_vocab, cation_vocab_size, args.embedding_size, args.vae_dropout,
                                cation_padding_idx, cation_sos_idx, cation_unk_idx, args.vae_max_len, args.vae_n_layers,
                                args.hidden_size, bidirectional=args.enc_bidir, latent_size=args.latent_size,
                                partialsmiles=args.partialsmiles).cuda()
    cation_vae.load_state_dict(state['vae'])

    state = torch.load(args.anion_vae_dir)
    anion_vae = vae_models.Vae(anion_vocab, anion_vocab_size, args.embedding_size, args.vae_dropout,
                               anion_padding_idx, anion_sos_idx, anion_unk_idx, args.vae_max_len, args.vae_n_layers,
                               args.hidden_size, bidirectional=args.enc_bidir, latent_size=args.latent_size,
                               partialsmiles=args.partialsmiles).cuda()
    anion_vae.load_state_dict(state['vae'])
    return cation_vae, anion_vae


def sample_prior(args, vae, n_samples, index_return=False, z_samples=None, max_len=None):
    if z_samples is None:
        z_samples = torch.FloatTensor(n_samples, args.latent_size).normal_(0, 1)
    smiles_pred, index_pred = vae.inference(latent=z_samples.cuda(), max_len=max_len)
    return smiles_pred, index_pred if index_return else smiles_pred


def sampling_figure(model, model_name, given_start, samples=None, log_path=None, seed=1919, latent_size=256, distance=None,
                    smiles_return=True):
    random.seed(seed)
    if samples is None:
        samples = [5]
    if log_path is None:
        log_path = time.strftime("%d_%H-%M-%S", time.localtime())
    if distance is None:
        distance = [35, 45, 55, 65, 75]
    if given_start:
        start = model.simple_encode(given_start)
    else:
        start = np.random.randn(1, latent_size)

    os.makedirs('Graph_Mol/' + log_path + '/' + model_name)
    start_filename = 'Graph_Mol/' + log_path + '/' + model_name + '/start_mol.png'
    start_s_filename = 'Graph_Mol/' + log_path + '/' + model_name + '/start_mol.svg'
    MolToImgFile(start, model, start_filename, start_s_filename, smiles_return=False)
    z_sample = []
    smiles_sample = []
    block = BlockLogs()
    if len(samples) == 1:
        for i_ in range(len(distance)):
            sample_matrix = np.random.randn(samples, latent_size)
            for j_ in range(sample_matrix.shape[0]):
                if given_start:
                    sample_matrix[j_] = ((sample_matrix[j_] / np.sum(sample_matrix[j_] ** 2) * distance[i_]).reshape(
                        -1, 1) + start.reshape(-1, 1)).reshape(-1)
                else:
                    sample_matrix[j_] = sample_matrix[j_] / np.sum(sample_matrix[j_] ** 2) * distance[i_]
            z_sample.append(sample_matrix)
    elif len(samples) > 1 and len(samples) == len(distance):
        s_i = [0]
        for index in range(len(samples)-1):
            s_i.append(s_i[index]+samples[index]) #start_index
        sample_matrix = np.random.randn(sum(samples), latent_size)
        for i_ in range(len(distance)):
            index_slice = range(s_i[i_], s_i[i_]+samples[i_])
            for j_ in index_slice:
                if given_start:
                    sample_matrix[j_] = ((sample_matrix[j_] / np.sum(sample_matrix[j_] ** 2) * distance[i_]).reshape(
                        -1, 1) + start.reshape(-1, 1)).reshape(-1)
                else:
                    sample_matrix[j_] = sample_matrix[j_] / np.sum(sample_matrix[j_] ** 2) * distance[i_]
            z_sample.append(sample_matrix[index_slice])
    else:
        print("Please set correct Samples. Current Samples:", samples)
        sys.exit()

    for i_ in trange(len(z_sample)):
        smiles_sample_item = []
        for j_ in range(z_sample[i_].shape[0]):
            z = z_sample[i_][j_].reshape(1, -1)
            filename = 'Graph_Mol/' + log_path + '/' + model_name + '/' + str(i_) + '_' + str(j_) + '.png'
            s_filename = 'Graph_Mol/' + log_path + '/' + model_name + '/s' + str(i_) + '_' + str(j_) + '.svg'
            # s_filename_png = 'Graph_Mol/' + log_path + '/' + model_name + '/s' + str(i_) + '_' + str(j_) + '.png'
            smiles = MolToImgFile(z, model, filename, s_filename)
            # cairosvg.svg2png(url=s_filename, write_to=s_filename_png)
            smiles_sample_item.append(smiles)
        smiles_sample.append(smiles_sample_item)
    del block
    if smiles_return:
        for smi in smiles_sample:
            print(smi)


def il_possibility_test(args, cation_vae, anion_vae):
    num_list = [0, 0, 0, 0, 0, 0, 0]
    for _ in range(40):
        cation_smiles_pred = sample_prior(args, cation_vae, n_samples=2500, index_return=False, max_len=120)
        anion_smiles_pred = sample_prior(args, anion_vae, n_samples=2500, index_return=False, max_len=120)
        cation_smiles_list = cation_smiles_pred[0]
        anion_smiles_list = anion_smiles_pred[0]
        _, fold_num_list, _ = available_test(cation_smiles_list, anion_smiles_list)
        num_list += fold_num_list
    # cation_aval_num, anion_aval_num, rd_aval_num, cation_ion_num, anion_ion_num, rdion_aval_num, aval_num
    print("cation_aval_num", num_list[0])
    print("anion_aval_num", num_list[1])
    print("rd_aval_num", num_list[2])
    print("cation_ion_num", num_list[3])
    print("anion_ion_num", num_list[4])
    print("rdion_aval_num", num_list[5])
    print("aval_num", num_list[6])


def scatter_draw(points, labels, figsize=(10, 8), bbox_to_anchor=(1.21, 1), right_dis=0.85, fig_path=False):
    # 不同类的标签
    unique_labels = list(set(labels))
    label_len = len(unique_labels)
    colors = []
    for i in range(label_len):
        colors.append('C' + str(i))
    color_mapping = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    # 调整图纸大小
    fig, ax = plt.subplots(figsize=figsize)

    # 根据color_mapping为random_points画散点图
    for i, point in enumerate(points):
        label = labels[i]
        color = color_mapping[label]
        ax.scatter(point[0], point[1], color=color)

    ax.tick_params(axis='both', which='major', labelsize=18)

    # 添加图例
    legend_elements = []
    for label, color in color_mapping.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=label_len))
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=bbox_to_anchor,
              prop={'family': 'Times New Roman'})

    # 调整散点图与图纸边界的边距
    plt.subplots_adjust(right=right_dis)

    # 显示散点图
    plt.show()

    if fig_path:
        fig.savefig(fig_path)


def principle_component_analysis(args, smi, labels, model, model_name, draw_mode):
    pca = PCA(n_components=2)
    if draw_mode == 'frag_draw':
        latent_set = encode(args, smi, model, model_name)
        points = pca.fit_transform(latent_set)
        scatter_draw(points, labels)
    elif draw_mode == 'whole_draw':
        cation_latent_set = encode(args, smi[0], model[0], model_name[0])
        anion_latent_set = encode(args, smi[1], model[1], model_name[1])
        latent_set = np.hstack((cation_latent_set, anion_latent_set))
        points = pca.fit_transform(latent_set)
        scatter_draw(points, labels[0])
        scatter_draw(points, labels[1])


def tsne(args, smi, labels, model, model_name, draw_mode, fig_path=False):
    pca = PCA(n_components=100)
    tsne = TSNE(perplexity=37, n_iter=1000)
    if draw_mode == 'frag_draw':
        latent_set = encode(args, smi, model, model_name)
        d_latent_set = pca.fit_transform(latent_set)
        points = tsne.fit_transform(d_latent_set)
        scatter_draw(points, labels, fig_path=fig_path)
    elif draw_mode == 'whole_draw':
        cation_latent_set = encode(args, smi[0], model[0], model_name[0])
        anion_latent_set = encode(args, smi[1], model[1], model_name[1])
        latent_set = np.hstack((cation_latent_set, anion_latent_set))
        d_latent_set = pca.fit_transform(latent_set)
        points = tsne.fit_transform(d_latent_set)
        scatter_draw(points, labels[0])
        scatter_draw(points, labels[1])


def vuns_test(train_file_path):
    """ 0. Sample """
    block = BlockLogs()
    n_samples = 2500
    cnt_valid = 0
    cnt_total = 0
    cnt_kind = 0
    cnt_simsmi = 0
    sas_list = []
    with open(train_file_path, 'r') as f:
        mol_strs = f.read().strip().split('\n')
        mol_strs = [mol.replace(' ', '') for mol in mol_strs]
    train_set = set(mol_strs)
    for step in range(40):
        cnt_total += n_samples
        smiles_pred, index_pred = sample_prior(n_samples=n_samples, index_return=True, max_len=120)
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
        print('Calculating SAScore in Step {} '.format(step + 1))
        """ 4. Synthetic Accessibility Score"""
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
    # logger.info('Validity: {:.4f}'.format(cnt_valid / cnt_total))
    # logger.info('Uniqueness: {:.4f}'.format(cnt_kind / cnt_total))
    # logger.info('Novelty: {:.4f}'.format(1 - (cnt_simsmi / cnt_total)))
    sa_score_list = np.array(sas_list)
    plt.figure()
    a = sb.displot(sa_score_list)
    plt.show()


def vdp(data, model):
    data_z, data_t, data_p, data_y = data[:, :256*2], data[:, 256*2], data[:, -3], data[:, -2]
    # data_t = T_scaler.fit_transform(data_t.reshape(-1, 1))
    # data_p = P_scaler.fit_transform(data_p.reshape(-1, 1))
    data_z = data_z.astype(np.float32)
    data_t = data_t.astype(np.float32)
    data_p = data_p.astype(np.float32)
    var_data_t = Variable(torch.from_numpy(data_t)).cuda()
    var_data_p = Variable(torch.from_numpy(data_p)).cuda()
    var_data_z = Variable(torch.from_numpy(data_z)).cuda()
    y_pred = model(var_data_z, var_data_t, var_data_p)
    y_pred = y_pred.cpu().numpy().reshape(-1, 1)
    y_true = data_y.reshape(-1, 1)
    return y_true, y_pred


def plot_predictions(true_values, predicted_values):
    plt.scatter(true_values, predicted_values, color='blue', label='Predicted Values')
    plt.plot(true_values, true_values, color='red', label='y=x')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()


def his_draw(data=None, data_name=None, figure_save=True):

    if data_name == 'P0.8':
        xticks = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        xlabel = 'Pressure(kPa)'
        figure_name = 'Pressure_v2_lb.svg'
        data = pd.read_excel("../data/ModelData/CO2ModelData.xlsx", sheet_name='Statistic')
        data = data[data_name].values
    elif data_name == 'T':
        xticks = [250, 275, 300, 325, 350, 375, 400, 425, 450]
        xlabel = 'Temperature(K)'
        figure_name = 'Temperature_v3_lb.svg'
        data = pd.read_excel("../data/ModelData/CO2ModelData.xlsx", sheet_name='Statistic')
        data = data[data_name].values
    elif data_name == 'Mole':
        xticks = [0, 0.1, 0.19, 0.29, 0.38, 0.48, 0.57, 0.67, 0.76, 0.86, 0.95]
        xlabel = 'Experimental solubility(Mole Fraction)'
        figure_name = 'Mole Fraction_v2_lb.svg'
        data = pd.read_excel("../data/ModelData/CO2ModelData.xlsx", sheet_name='Statistic')
        data = data[data_name].values
    elif data_name == 'sorty':
        xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        xlabel = 'Predicted solubility of combined ILs(Mole Fraction)'
        figure_name = 'Mole Fraction_3_lb.svg'
        data = pd.read_csv("../data/ModelData/sort_y_10p.csv", header=0, index_col=False)
        data = data.values
    elif data_name == 'optsorty':
        xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        xlabel = 'Predicted solubility of optimized ILs(Mole Fraction)'
        figure_name = 'PSO Mole Fraction_3_lb.svg'
        data = pd.read_csv("../data/ModelData/t_sort_y.csv", header=0, index_col=False)
        data = data.values
    elif data_name == None:
        # option value
        # xticks = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]
        xlabel = 'Predicted solubility of combined ILs(Mole Fraction)'
        figure_name = 'Mole Fraction_filter.svg'

    data = data[~np.isnan(data)]

    # 计算频率和累积百分比
    values, counts = np.unique(data, return_counts=True)
    cumulative_percent = np.cumsum(counts) / data.size * 100

    # 设置箱数
    num_bins = 15

    # 计算直方图
    hist_values, bin_edges = np.histogram(data, bins=num_bins)

    # 创建带有两个子图的画布
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # 使用浅蓝色
    color = 'lightblue'

    # 绘制每个条形图，并设置颜色
    for left, bottom, height in zip(bin_edges[:-1], np.zeros_like(hist_values), hist_values):
        width = bin_edges[1] - bin_edges[0]
        # color = cmap(height/np.max(heights))
        ax1.bar(left, height, width=width, bottom=bottom, color=color, edgecolor='black', align='edge')

    # 绘制直方图
    # 使用ax1.bar创建透明的条形图（并保留边框）
    # ax1.bar(bin_edges[:-1], hist_values, width=(bin_edges[1] - bin_edges[0]), color='none', edgecolor='black', align='edge')
    ax1.set_ylabel('Frequency', fontdict={'family': 'Times New Roman', 'size':24})

    # 设置x轴刻度为直方图的区间边界及标题
    ax1.set_xticks(bin_edges[:-1])
    # ax1.set_xticks([0, 0.1, 0.19, 0.29, 0.38, 0.48, 0.57, 0.67, 0.76, 0.86, 0.95])
    # ax1.set_xticks(xticks)
    ax1.set_xlabel(xlabel, fontdict={'family': 'Times New Roman', 'size':24})
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # 在同一张图上添加第二个子图
    ax2 = ax1.twinx()
    ax2.plot(values, cumulative_percent, 'r-')
    ax2.set_ylabel('Cumulative percentage', fontdict={'family': 'Times New Roman', 'color':'r','size':24})
    ax2.tick_params(axis='y', colors='r', labelsize=18)
    # 添加虚线
    # ax2.axhline(y=90, color='gray', linestyle='dashed', linewidth=1)

    plt.show()
    if figure_save:
        fig.savefig(figure_name, format='svg', dpi=150)


def comparison_plt(train_y_true, train_y_pred, test_y_true, test_y_pred, fig_save=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    train_r2 = r2_score(train_y_true, train_y_pred)
    test_r2 = r2_score(test_y_true, test_y_pred)
    train_mse = mean_squared_error(train_y_true, train_y_pred)
    test_mse = mean_squared_error(test_y_true, test_y_pred)
    ax.scatter(train_y_true, train_y_pred, color='blue')
    ax.scatter(test_y_true, test_y_pred, color='#00CC00')
    ax.plot(xticks, yticks, color='black', label='y=x')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.legend(['train_set', 'test_set'], prop={'family':'Arial', 'size':18})
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel("Experimental Solubility", fontdict={'family':'Arial', 'size':28})
    ax.set_ylabel("Predicted Solubility", fontdict={'family':'Arial', 'size':28})
    print("train_r2: ", train_r2, "test_r2: ", test_r2)
    print("train_mse: ", train_mse, "test_mse: ", test_mse)
    plt.show()
    if fig_save:
        fig.savefig('./picture/comparison(rev).svg', dpi = 300)


def error_compar_plt(train_y_true, train_y_pred, test_y_true, test_y_pred, fig_save=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    error_train = train_y_pred - train_y_true
    error_test = test_y_pred - test_y_true
    ax.scatter(train_y_true, error_train, color='blue')
    ax.scatter(test_y_true, error_test, color='#00CC00')
    ax.plot([0, 1], [0, 0], color='black', label='y=0')
    ax.axhline(y=0.1, color='gray', linestyle='dashed', linewidth=1)
    ax.axhline(y=-0.1, color='gray', linestyle='dashed', linewidth=1)
    ax.axhline(y=0.2, color='gray', linestyle='dashed', linewidth=1)
    ax.axhline(y=-0.2, color='gray', linestyle='dashed', linewidth=1)
    plt.ylim(-0.25, 0.25)
    plt.xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(['train_set', 'test_set'], prop={'family':'Arial', 'size':18})
    ax.set_xlabel("Experimental Solubility", fontdict={'family':'Arial', 'size':28})
    ax.set_ylabel("Error of Model", fontdict={'family':'Arial', 'size':28})
    plt.show()
    if fig_save:
        fig.savefig('./picture/ErrorComparison(rev).svg', dpi = 300)


def err_distri_plt(train_y_true, train_y_pred, test_y_true, test_y_pred, fig_save=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    error_train = train_y_pred - train_y_true
    error_test = test_y_pred - test_y_true
    ax.hist(error_train, bins=60, alpha=0.5, label='train_set', color='blue', range=(-0.2, 0.2), edgecolor='black')
    ax.hist(error_test, bins=60, alpha=0.5, label='test_set', color='#00CC00', range=(-0.2, 0.2), edgecolor='black')
    ax.legend(['train_set', 'test_set'], prop={'family':'Arial', 'size':18})
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel("Experimental Solubility", fontdict={'family':'Arial', 'size':28})
    ax.set_ylabel("Frequency of data points in each error range", fontdict={'family':'Arial', 'size':28})
    plt.xlim(-0.2, 0.2)
    plt.show()
    if fig_save:
        fig.savefig('./picture/ErrorDistribution(rev).svg', dpi = 300)

def combination(ani_smi, cat_smi):
    with open(cat_smi, 'r') as f:
        cat_mol_strs = f.read().strip().split('\n')
        cat_mol_strs = [mol.replace(' ', '') for mol in cat_mol_strs]
    with open(ani_smi, 'r') as f:
        ani_mol_strs = f.read().strip().split('\n')
        ani_mol_strs = [mol.replace(' ', '') for mol in ani_mol_strs]
    cps = []  # coupled smiles
    for i in range(len(cat_mol_strs)):
        cs = cat_mol_strs[i]
        for j in range(len(ani_mol_strs)):
            ans = ani_mol_strs[j]
            tcps = cs + '.' + ans
            cps.append(tcps)
    cps = np.array(cps)
    return cps