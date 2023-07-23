import sys
sys.path.append("../..")
from myutils import evaluate_all, dir_path
import pandas as pd
import numpy as np
import torch

data_dir = "result_mul/"
data_dir_all = dir_path(k=1)+"NIHGCN/Data/CCLE/"

# 加载细胞系-药物矩阵
cell_drug = pd.read_csv(data_dir_all + "cell_drug_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
cell_sum = np.sum(cell_drug, axis=1)
drug_sum = np.sum(cell_drug, axis=0)


target_dim = [0,1]
n_kfold = 1


for dim in target_dim:
    auc_all = []
    ap_all = []
    acc_all = []
    f1_all = []
    mcc_all = []
    if dim == 0:
        name = "cell"
    else:
        name = "drug"
    for target_index in np.arange(cell_drug.shape[dim]):
        if dim:
            if drug_sum[target_index] < 10:
                continue
        else:
            if cell_sum[target_index] < 10:
                continue
        print(target_index)
        for i in np.arange(n_kfold):
            predict_data = pd.read_csv(data_dir + "ccle_{}_{}_predict_data.csv".format(name,target_index), index_col=0, header=0, nrows=1, skiprows=i)
            predict_data = predict_data.dropna(axis=1,how='any')
            predict_data = np.array(predict_data)[0]
            predict_data = torch.tensor(predict_data)
            true_data = pd.read_csv(data_dir + "ccle_{}_{}_true_data.csv".format(name,target_index), index_col=0, header=0, nrows=1, skiprows=i)
            true_data = true_data.dropna(axis=1,how='any')
            true_data = np.array(true_data)[0]
            true_data = torch.tensor(true_data)
            auc, ap, acc, f1, mcc = evaluate_all(true_data, predict_data)
            auc_all.append(auc)
            ap_all.append(ap)
            acc_all.append(acc)
            f1_all.append(f1)
            mcc_all.append(mcc)
    file_write_obj = open(data_dir +"result_all", 'a+')
    file_write_obj.writelines("dim: {}".format(str(dim)))
    file_write_obj.writelines("|final auc: {}|final ap: {}|final acc: {}|final f1: {}|final mcc: {}"
                              .format(str(np.mean(auc_all)), str(np.mean(ap_all)), str(np.mean(acc_all)), str(np.mean(f1_all)), str(np.mean(mcc_all))))
    file_write_obj.write('\n')
    file_write_obj.writelines("|final auc_var: {}|final ap_var: {}|final acc_var: {}|final f1_var: {}|final mcc_var: {}"
                              .format(str(np.var(auc_all)), str(np.var(ap_all)), str(np.var(acc_all)), str(np.var(f1_all)), str(np.var(mcc_all))))
    file_write_obj.write('\n')
    file_write_obj.close()