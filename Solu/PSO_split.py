import sys
import time
import rdkit
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
import pubchempy as pcp
from custom_pso import PSO, GBPSO
from Methods import load_vae
from rdkit.rdBase import BlockLogs
from torch.autograd import Variable, grad
from base_classes.trainer import TrainerBase
from base_classes.argparser import JoinArgParser
from sklearn.preprocessing import StandardScaler
from rdkit.ML.Descriptors import MoleculeDescriptors
from data.molecule_iterator import SmileBucketIterator


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


def pso_opt(args, t_min, t_max, p_min, p_max, model=None, data=None,
            latent_dim=256, store=False, repeat_num=0, repeat_sample=False, pre_scaled=True):
    """ 0. load """
    time_str = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    if model is None:
        try:
            mlp_name = 'mlp_' + str(latent_dim) + '.pkl'
            # mlp = JDnn(args.mlp_cat_index, args.mlp_network, latent_dim*2, 1, 0).cuda()
            mlp = torch.load(mlp_name)
            for p in mlp.parameters():
                p.requires_grad_(False)
        except FileNotFoundError:
            print('Error: No mlp model found, please train one.')
            sys.exit()
    else:
        mlp = model
    try:
        cation_vae, anion_vae = load_vae(args)
    except FileNotFoundError:
        print('Error: No vae model found, please train.')
        sys.exit()
    if data is None:
        data_name = 'train_data_' + str(latent_dim) + '_non-scaled.pkl'
        data = torch.load(data_name)
        data_z, data_t, data_p = data[:, :latent_dim * 2], data[:, latent_dim * 2], data[:, -3]
    else:
        data = data
        data_z, data_t, data_p = data[:, 4:latent_dim*2+4], data[:, -4], data[:, -3]

    """ 2. Preparation """
    def optim_func(pop_latent_code, grad_return=False):
        latent_code = Variable(torch.FloatTensor(pop_latent_code)).cuda()# latent_code:[T, P, latent] T,P stay constant
        t, p, latent_vector = latent_code[:, 0], latent_code[:, 1], latent_code[:, 2:]
        if grad_return:
            latent_vector.requires_grad_(True)
            t.requires_grad_(True)
            p.requires_grad_(True)
        property_ = -mlp(latent_vector, t, p)
        if grad_return:
            ones_tensor = torch.ones(512, 1).cuda()
            torch.autograd.backward(property_, grad_tensors=ones_tensor)
            # property_.requires_grad_(True)
            # anion_tensor, cation_tensor = latent_vector[:, :latent_dim], latent_vector[:, latent_dim:]
            latent_grad = latent_vector.grad.cpu().numpy()
            t_grad = t.grad.cpu().numpy().reshape(-1, 1)
            p_grad = p.grad.cpu().numpy().reshape(-1, 1)
            grads = np.hstack((latent_grad, t_grad, p_grad))
            property_ = property_.detach().cpu().numpy()
        else:
            property_ = property_.cpu().numpy()
        cation_latent, anion_latent = torch.split(latent_vector, latent_dim, dim=1)
        cation_smiles, _ = cation_vae.inference(cation_latent)
        anion_smiles, _ = anion_vae.inference(anion_latent)
        normative_, num_list, aval_num = available_test(cation_smiles, anion_smiles)
        cation_smiles, anion_smiles = np.array(cation_smiles), np.array(anion_smiles)
        if args.psovalscore:
            obj = property_ + normative_
            return obj, num_list, aval_num, cation_smiles.reshape(-1, 1), anion_smiles.reshape(-1, 1)
        else:
            return property_, grads, aval_num, cation_smiles.reshape(-1, 1), anion_smiles.reshape(-1, 1)

    if store:
        best10x = np.load("best10x.npy")
        best10y = np.load("best10y.npy")
        for i in range(10):
            cation_latent, anion_latent = best10x[i, 2:latent_dim+2], best10x[i, latent_dim+2:]
            cation_latent, anion_latent = cation_latent.reshape(1, -1), anion_latent.reshape(1, -1)
            cation_latent_code = Variable(torch.FloatTensor(cation_latent)).cuda()
            anion_latent_code = Variable(torch.FloatTensor(anion_latent)).cuda()
            cation_smiles, _ = cation_vae.inference(cation_latent_code)
            anion_smiles, _ = anion_vae.inference(anion_latent_code)
            # logger.info("top_smiles_{}: {}.{}, value:{:.3f}".format(i+1, cation_smiles[0], anion_smiles[0], -(best10y[i, 0]+3000)))
        sys.exit()

    # scale and get boundary(Z temporarily not scaled)
    # data_z, data_t, data_p = data[:, :latent_dim*2], data[:, -3], data[:, -2]
    if not pre_scaled:
        T_scaler, P_scaler = StandardScaler(), StandardScaler()
        T_scaler.fit(data_t.reshape(-1, 1))
        P_scaler.fit(data_p.reshape(-1, 1))
        # max = 1
        # min = -1
        t_min = T_scaler.transform(np.array(t_min).reshape(-1, 1))[0][0]
        t_max = T_scaler.transform(np.array(t_max).reshape(-1, 1))[0][0]
        p_min = P_scaler.transform(np.array(p_min).reshape(-1, 1))[0][0]
        p_max = P_scaler.transform(np.array(p_max).reshape(-1, 1))[0][0]
    data_cz, data_az = data_z[:, :latent_dim], data_z[:, latent_dim:]
    cl_max, cl_min = np.max(data_cz), np.min(data_cz)              # max/min of latent vectors
    al_max, al_min = np.max(data_az), np.min(data_az)              # max/min of latent vectors
    low_boundary = np.ones(latent_dim * 2 + 2) * (cl_min)
    low_boundary[latent_dim:] = al_min
    up_boundary = np.ones(latent_dim * 2 + 2) * (cl_max)
    up_boundary[latent_dim:] = al_max
    low_boundary[0], low_boundary[1] = t_min, p_min
    up_boundary[0], up_boundary[1] = t_max, p_max
    low_boundary = low_boundary.tolist()
    up_boundary = up_boundary.tolist()
    # low: [t_min, p_min, min_z, min_z, ...]
    # max: [t_max, p_max, max_z, max_z, ...]
    if repeat_num > 1:
        # t_sort_y = np.zeros((1, 1))
        t_sort_data = np.zeros((1, 3))
        for epoch in range(repeat_num):
            if args.gbpso:
                pso = GBPSO(args=args, func=optim_func, dim=latent_dim * 2 + 2, pop=512, max_iter=args.max_iter,
                          lb=low_boundary, ub=up_boundary, w=1.2, c1=0.5, c2=0.5, c3=0.01, logging=False, y_hist_record=False)
            else:
                pso = PSO(args=args, func=optim_func, dim=latent_dim * 2 + 2, pop=512, max_iter=args.max_iter,
                          lb=low_boundary, ub=up_boundary, w=1.2, c1=0.5, c2=0.5, logging=False, y_hist_record=False)
            pso.run()
            if repeat_sample:
                sim_num = 0
                for i in range(10):
                    # decode
                    cation_latent, anion_latent = (pso.best10x[i, 2:latent_dim+2]).reshape(1, -1), (pso.best10x[i, latent_dim+2:]).reshape(1, -1)
                    cation_smiles, _ = cation_vae.inference(Variable(torch.FloatTensor(cation_latent)).cuda())
                    anion_smiles, _ = anion_vae.inference(Variable(torch.FloatTensor(anion_latent)).cuda())
                    # search
                    cation_smi, anion_smi = cation_smiles[0], anion_smiles[0]
                    ca_sim_num, an_sim_num = 0, 0
                    try:
                        ca_sim_num = len(pcp.get_compounds(cation_smi, 'smiles', searchtype='similarity', MaxRecords=5))
                        sim_num += 1
                    except:
                        pass
                    try:
                        an_sim_num = len(pcp.get_compounds(anion_smi, 'smiles', searchtype='similarity', MaxRecords=5))
                        sim_num += 1
                    except:
                        pass
                    # logger.info("similarity: {}, {} for smi {}: {}.{}"
                    #             .format(ca_sim_num, an_sim_num, i, cation_smi, anion_smi))
            else:
                """
                e_sort_x = pso.sort_x
                latent_code = Variable(torch.FloatTensor(e_sort_x)).cuda()
                cation_latent, anion_latent = torch.split(latent_code, latent_dim, dim=1)
                cation_smiles, _ = cation_vae.inference(cation_latent)
                anion_smiles, _ = anion_vae.inference(anion_latent)
                cs, ans = np.array(cation_smiles), np.array(anion_smiles)
                t_sort_sy = np.hstack((cs.reshape(-1, 1), ans.reshape(-1, 1), -pso.sort_y-3000))
                """
                t_sort_sy = np.hstack((pso.sort_cs, pso.sort_as, -pso.sort_y))
                t_sort_data = np.vstack((t_sort_data, t_sort_sy))
        sort_data = np.delete(t_sort_data, obj=0, axis=0)
        df = pd.DataFrame(sort_data)
        df.to_excel('pso_sort_data_' + time_str + '.xlsx', index=False)
    else:
        if args.gbpso:
            pso = GBPSO(args=args, func=optim_func, dim=latent_dim * 2 + 2, pop=512, max_iter=args.max_iter,
                          lb=low_boundary, ub=up_boundary, w=1.2, c1=0.5, c2=0.5, c3=0.001, logging=False,
                          y_hist_record=False)
        else:
            pso = PSO(args=args, func=optim_func, dim=latent_dim * 2 + 2, pop=512, max_iter=args.max_iter,
                      lb=low_boundary, ub=up_boundary, w=1.1, c1=0.5, c2=0.5, logging=False)
        pso.run()
        latent_code = Variable(torch.FloatTensor(pso.sort_x)).cuda()
        cation_latent, anion_latent = torch.split(latent_code, latent_dim, dim=1)
        cation_smiles, _ = cation_vae.inference(cation_latent)
        anion_smiles, _ = anion_vae.inference(anion_latent)
        cs, ans = np.array(cation_smiles), np.array(anion_smiles)
        sort_sy = np.hstack((cs.reshape(-1, 1), ans.reshape(-1, 1), -pso.sort_y))
        df = pd.DataFrame(sort_sy)
        df.to_excel('pso_sort_data_' + time_str + '.xlsx', index=False)
    if args.PSO_draw:
        """ draw best """
        # cation_latent, anion_latent = pso.gbest_x_hist[:, 2:latent_dim+2], pso.gbest_x_hist[:, latent_dim+2:]
        cation_latent, anion_latent = pso.gbest_x[2:latent_dim+2], pso.gbest_x[latent_dim+2:]
        cation_latent, anion_latent = cation_latent.reshape(1, -1), anion_latent.reshape(1, -1)
        cation_latent_code = Variable(torch.FloatTensor(cation_latent)).cuda()
        anion_latent_code = Variable(torch.FloatTensor(anion_latent)).cuda()
        cation_smiles, _ = cation_vae.inference(cation_latent_code)
        anion_smiles, _ = anion_vae.inference(anion_latent_code)
        # print('best_smiles: ', cation_smiles[0], '.', anion_smiles[0])
        pso.logger.info("best_smiles: {}.{} ".format(cation_smiles[0], anion_smiles[0]))
        # np.save("best10x.npy", pso.best10x)
        # np.save("best10y.npy", pso.best10y)
        """ draw best 10 """
        for i in range(10):
            cation_latent, anion_latent = pso.best10x[i, 2:latent_dim+2], pso.best10x[i, latent_dim+2:]
            cation_latent, anion_latent = cation_latent.reshape(1, -1), anion_latent.reshape(1, -1)
            cation_latent_code = Variable(torch.FloatTensor(cation_latent)).cuda()
            anion_latent_code = Variable(torch.FloatTensor(anion_latent)).cuda()
            cation_smiles, _ = cation_vae.inference(cation_latent_code)
            anion_smiles, _ = anion_vae.inference(anion_latent_code)
            pso.logger.info("top_smiles_{}: {}.{}, value:{:.3f}".format(i+1, cation_smiles[0], anion_smiles[0], -(pso.best10y[i, 0]+3000)))


if __name__ == '__main__':
    # args
    args = JoinArgParser().parse_args()
    # t_max = 453.15 t_min = 243.2 p_max = 50000 p_min = 0
    T_min = 296
    T_max = 300.15
    P_min = 100 #kPa
    P_max = 100.1
    latent_dim = 256
    pso_opt(args, latent_dim, T_min, T_max, P_min, P_max, store=False, repeat_num=10, repeat_sample=True)

