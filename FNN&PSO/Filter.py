import sys
import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from base_classes.trainer import TrainerBase
from .. import Methods
from Methods import encode, load_vae
from sklearn.preprocessing import StandardScaler
from base_classes.argparser import JoinArgParser


def filter(args, t, pr, mlp=None, vae_load=True, batct_input=False, data_return=True):
    if mlp is None:
        try:
            mlp = torch.load('../Solu/mlp_256.pkl')
            for p in mlp.parameters():
                p.requires_grad_(False)
        except FileNotFoundError:
            print('Error: No mlp model found, please train one.')
            sys.exit()

    if vae_load:
        try:
            anion_vae, cation_vae = load_vae(args)
        except FileNotFoundError:
            print('Error: No vae model found, please train.')
            sys.exit()

    MD = pd.read_excel("../data/ModelData/CO2ModelData.xlsx", sheet_name='ModelData')
    ModelData = MD.values
    ani_smi = '../data/Smiles&Vocab/Solu/213anioncan.txt'
    ani_lat = encode(args, ani_smi, anion_vae, "anion_vae", file_read=True) #[45, 256]
    cat_smi = '../data/Smiles&Vocab/Solu/213cationcan.txt'
    cat_lat = encode(args, cat_smi, cation_vae, "cation_vae", file_read=True)
    with open(cat_smi, 'r') as f:
        cat_mol_strs = f.read().strip().split('\n')
        cat_mol_strs = [mol.replace(' ', '') for mol in cat_mol_strs]
    with open(ani_smi, 'r') as f:
        ani_mol_strs = f.read().strip().split('\n')
        ani_mol_strs = [mol.replace(' ', '') for mol in ani_mol_strs]

    # scale
    data_t, data_p = ModelData[:, -4], ModelData[:, -3]
    T_scaler, P_scaler = StandardScaler(), StandardScaler()
    T_scaler.fit(data_t.reshape(-1, 1))
    P_scaler.fit(data_p.reshape(-1, 1))
    leng = len(cat_lat)*len(ani_lat)
    if not batct_input:
        tv = np.array([t]*leng)
        prv = np.array([pr]*leng)
    else:
        tv = np.array(t)
        prv = np.array(pr)
    t = T_scaler.transform(np.array(tv).reshape(-1, 1))   #(3735,1)
    pr = P_scaler.transform(np.array(prv).reshape(-1, 1)) #(3735,1)

    """1. Combination"""
    cpl = np.zeros(512) # coupled latent
    for i in range(len(cat_lat)):
        cl = cat_lat[i, :]
        for j in range(len(ani_lat)):
            al = ani_lat[j, :]
            tcpl = np.hstack((cl, al))
            cpl = np.vstack((cpl, tcpl))
    cpl = np.delete(cpl, obj=0, axis=0)

    cps = [] # coupled smiles
    for i in range(len(cat_mol_strs)):
        cs = cat_mol_strs[i]
        for j in range(len(ani_mol_strs)):
            ans = ani_mol_strs[j]
            tcps = cs + '.' + ans
            cps.append(tcps)
    cps = np.array(cps)
    cps = cps.reshape(-1, 1)

    """2. Prediction&Filter"""
    cpl = cpl.astype(np.float32)
    t = t.astype(np.float32)
    pr = pr.astype(np.float32)
    var_cpl = Variable(torch.from_numpy(cpl)).cuda()
    var_t = Variable(torch.from_numpy(t)).cuda()
    var_pr = Variable(torch.from_numpy(pr)).cuda()
    with torch.no_grad():
        y_pred = mlp(var_cpl, var_t, var_pr)
    y_pred = y_pred.cpu().numpy()

    # sort
    si = np.argsort(y_pred, axis=0)[::-1]
    sort_smi = np.take_along_axis(cps, si, axis=0)
    sort_y = np.take_along_axis(y_pred, si, axis=0)

    if data_return:
        return sort_smi, sort_y
    # sort_smi_y = np.hstack((sort_smi, sort_y))
    # best10x = sort_smi[:10, :]
    # best10y = sort_y[:10, :]
    """
    for i in range(10):
        if logging:
            logger.info("top_smiles_{}: {}, value:{:.3f}".format(i + 1, sort_smi[i, 0],  best10y[i, 0]))
        else:
            print("top_smiles_{}: {}, value:{:.3f}".format(i + 1, sort_smi[i, 0],  best10y[i, 0]))
    """


def filter_pt(args, t, pr, mlp=None, vae_load=True, data_return=True):
    """ filter but with pre_transformed input temperature and pressure. """
    if mlp is None:
        try:
            mlp = torch.load('../Solu/mlp_256.pkl')
            for p in mlp.parameters():
                p.requires_grad_(False)
        except FileNotFoundError:
            print('Error: No mlp model found, please train one.')
            sys.exit()

    if vae_load:
        try:
            anion_vae, cation_vae = load_vae(args)
        except FileNotFoundError:
            print('Error: No vae model found, please train.')
            sys.exit()

    ani_smi = '../data/Smiles&Vocab/Solu/213anioncan.txt'
    ani_lat = encode(args, ani_smi, anion_vae, "anion_vae", file_read=True) #[45, 256]
    cat_smi = '../data/Smiles&Vocab/Solu/213cationcan.txt'
    cat_lat = encode(args, cat_smi, cation_vae, "cation_vae", file_read=True)
    with open(cat_smi, 'r') as f:
        cat_mol_strs = f.read().strip().split('\n')
        cat_mol_strs = [mol.replace(' ', '') for mol in cat_mol_strs]
    with open(ani_smi, 'r') as f:
        ani_mol_strs = f.read().strip().split('\n')
        ani_mol_strs = [mol.replace(' ', '') for mol in ani_mol_strs]
    leng = len(cat_lat) * len(ani_lat)
    tv = np.array([t] * leng)
    prv = np.array([pr] * leng)
    t = tv.reshape(-1, 1)   #(3735,1)
    pr = prv.reshape(-1, 1) #(3735,1)

    """1. Combination"""
    cpl = np.zeros(512)
    for i in range(len(cat_lat)):
        cl = cat_lat[i, :]
        for j in range(len(ani_lat)):
            al = ani_lat[j, :]
            tcpl = np.hstack((cl, al))
            cpl = np.vstack((cpl, tcpl))
    cpl = np.delete(cpl, obj=0, axis=0)

    cps = []
    for i in range(len(cat_mol_strs)):
        cs = cat_mol_strs[i]
        for j in range(len(ani_mol_strs)):
            ans = ani_mol_strs[j]
            tcps = cs + '.' + ans
            cps.append(tcps)
    cps = np.array(cps)
    cps = cps.reshape(-1, 1)

    """2. Prediction&Filter"""
    cpl = cpl.astype(np.float32)
    t = t.astype(np.float32)
    pr = pr.astype(np.float32)
    var_cpl = Variable(torch.from_numpy(cpl)).cuda()
    var_t = Variable(torch.from_numpy(t)).cuda()
    var_pr = Variable(torch.from_numpy(pr)).cuda()
    with torch.no_grad():
        y_pred = mlp(var_cpl, var_t, var_pr)
    y_pred = y_pred.cpu().numpy()

    # sort
    si = np.argsort(y_pred, axis=0)[::-1]
    sort_smi = np.take_along_axis(cps, si, axis=0)
    sort_y = np.take_along_axis(y_pred, si, axis=0)

    if data_return:
        return sort_smi, sort_y


if __name__ == '__main__':
    # args
    t = 298.15
    pr = 100
    args = JoinArgParser().parse_args()
    args.logdir = './log/Filter'
    args.logging = True
    # logger
    if args.logging:
        loggerbase = TrainerBase(args.logdir, '')
        logger, tensorboard = loggerbase.logger, loggerbase.tensorboard
    data_dir = 'train_data_256_non-scaled.pkl'
    data = torch.load(data_dir)
    filter(args, data, t, pr)
