import sys
import time
import torch
import numpy as np
import pandas as pd
from solu_models import JDnn
from MLPTrainer import MLPTrainer
from sklearn.model_selection import KFold
from PSO_split import pso_opt
from data.DataProcessing import encode_stack, bulid_loader, build_xyloader
from sklearn.model_selection import train_test_split, cross_validate
from base_classes.argparser import JoinArgParser
from Filter import filter, filter_pt
# 1. Specific Arg
args = JoinArgParser().parse_args()
args.network = [256, 128, 64, 32, 16, 8, 4, 8, 8, 8]
args.cat_index = 6
args.data_save_dir = 'encoded_data.pkl'
time_str = time.strftime("%m-%d_%H-%M-%S", time.localtime())
Data_Process = False
debug = False
mode = 'Random'  # Mode Select:'Random', 'Custom', 'CrossValidation'(Default: 10fold)',
args.filter = False
args.filter_save = True
args.valid_mode = False
args.psovalscore = False
if debug:
    args.logging = False
    args.num_epochs = 1
    args.pred_save = False
    args.fig_save = False
    args.repeat = 2
    args.max_iter = 2
else:
    args.logging = True
    args.num_epochs = 60
    args.pred_save = False
    args.fig_save = False
    args.repeat = 10
    args.max_iter = 30

""" Random args"""
args.PSO = True
args.gbpso = True
args.PSO_draw = False
""" CrossValidation args"""
args.fold_n = 5
args.tt = -0.99853354 # t = 298.15K
args.prt = -0.61106296 # p = 100kPa
# 2. Data Process/Load
if Data_Process == True:
    # encode_stack(args, mode)
    scaler_T, scaler_P = encode_stack(args, mode, standarder_return=True)
    torch.save(scaler_T, "scalerT.pkl")
    torch.save(scaler_P, "scalerP.pkl")
    sys.exit()
else:
    data = torch.load(args.data_save_dir)

""" Random mode/Custom mode """
if mode == 'Random':
    trainer = MLPTrainer(args, start_logging=args.logging)
    train_data, test_data, train_loader, test_loader = bulid_loader(data, mode, data_return=True)
    model = JDnn().cuda()
    if args.pred_save:
        y_pred, model = trainer.fit_val(model, train_loader, test_loader, model_return=True, pred_return=True, logging=args.logging)
    else:
        model = trainer.fit_val(model, train_loader, test_loader, model_return=True, logging=args.logging)
    if args.valid_mode:
        _, valid_data = train_test_split(test_data, test_size=0.1, random_state=810, shuffle=True)
        valid_data_loader = bulid_loader(valid_data, mode='WholeValidation')
        trainer.val(model, valid_data_loader, args.logging)
    if args.pred_save:
        train_y_true, test_y_true = train_data[:, -2], test_data[:, -2]
        trd = np.hstack((train_data[:, :4], train_y_true.reshape(-1, 1)))
        tsd = np.hstack((test_data[:, :4], test_y_true.reshape(-1, 1)))
        whole_data = np.hstack((np.vstack((trd, tsd)), y_pred))
        df = pd.DataFrame(whole_data, columns=['cation smiles', 'anion smiles', 'T(K)', 'P(kPa)', 'mf', 'mf pred'])
        df.to_excel('whole data_'+ time_str +'.xlsx', index=False)
    if args.filter:
        sort_smi, sort_y = filter(args, t=298.15, pr=100, mlp=model)
        if args.filter_save:
            filter_data = np.hstack((sort_smi, sort_y))
            df = pd.DataFrame(filter_data, columns=['smiles', 'mf pred'])
            df.to_excel('filter_y_' + time_str + '.xlsx', index=False)
    if args.PSO:
        T_scaler, P_scaler = torch.load("scalerT.pkl"), torch.load("scalerP.pkl")
        t_min = T_scaler.transform(np.array(298.15).reshape(-1, 1))[0][0]
        t_max = T_scaler.transform(np.array(300.15).reshape(-1, 1))[0][0]
        p_min = P_scaler.transform(np.array(100).reshape(-1, 1))[0][0]
        p_max = P_scaler.transform(np.array(101).reshape(-1, 1))[0][0]
        pso_opt(args, t_min=t_min, t_max=t_max, p_min=p_min, p_max=p_max, model=model, data=data, repeat_num=args.repeat)


""" CrossValidation mode """
if mode == 'CrossValidation':
    X, Y = data[:, 4:-2], data[:, -2]
    trainer = MLPTrainer(args, start_logging=args.logging)
    kf = KFold(n_splits=5, shuffle=True)
    results_fmt = ("train_mae: {:.3f}, test_mae: {:.3f}, max solubility: {:.3f}").format
    results_ft2 = ("aver_train_mae: {:.3f}, aver_test_mae: {:.3f}, aver_max solubility: {:.3f}").format
    train_mae_list = []
    test_mae_list = []
    max_solubility = []
    fold_y = np.zeros((3735, 1))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model = JDnn().cuda()
        train_loader, test_loader = build_xyloader(X_train, y_train), build_xyloader(X_test, y_test)
        _, train_mae, test_mae = trainer.fit_val(model, train_loader, test_loader, pred_return=True, logging=False, mae_return=True)
        # _, sort_y = filter_pt(args, t=args.tt, pr=args.prt, mlp=model)
        _, sort_y = filter(args, data=data, t=298.15, pr=100, mlp=model)
        max_solu = np.max(sort_y)
        fold_y = np.hstack((fold_y, sort_y))
        trainer.logger.info(results_fmt(train_mae, test_mae, max_solu))
        train_mae_list.append(train_mae)
        test_mae_list.append(test_mae)
        max_solubility.append(max_solu)
    av_trm = sum(train_mae_list)/args.fold_n
    av_tem = sum(test_mae_list)/args.fold_n
    av_max = sum(max_solubility)/args.fold_n
    trainer.logger.info(results_ft2(av_trm, av_tem, av_max))
    if args.pred_save:
        df = pd.DataFrame(fold_y)
        df.to_excel('cv_' + time_str + '.xlsx', index=False)
