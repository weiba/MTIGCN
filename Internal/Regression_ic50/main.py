# coding: utf-8
import argparse
from load_data import load_data
from model import mtigcn_regression, Optimizer_regression
from sklearn.model_selection import KFold
from sampler import RandomSampler_ic50
from myutils import *


parser = argparse.ArgumentParser(description="Run MTIGCN")
parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
parser.add_argument('-data', type=str, default='gdsc', help='Dataset{gdsc or ccle}')
parser.add_argument('--lr', type=float,default=0.001,
                    help="the learning rate")
parser.add_argument('--wd', type=float,default=1e-5,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--layer_size', nargs='?', default=[1024,1024],
                    help='Output sizes of every layer')
parser.add_argument('--alpha', type=float,default=0.25,
                    help="the scale for balance gcn and ni")
parser.add_argument('--gamma', type=float,default=8,
                    help="the scale for sigmod")
parser.add_argument('--epochs', type=float,default=1000,
                    help="the epochs for model")
args = parser.parse_args()



#load data
res_ic50, res_binary, drug_finger, exprs, null_mask, pos_num, args = load_data(args)
cell_sim = exp_similarity(torch.from_numpy(exprs), sigma=torch.tensor(2), normalize=True)
drug_sim = jaccard_coef(torch.from_numpy(drug_finger))
args.epochs = 2000

loss_c_list = [0.1]
loss_d_list = [0.25]
loss_ic50_list = [0.25]
loss_binary_list = [0.75]
paras = distribute_compute(loss_c_list,loss_d_list,loss_ic50_list,loss_binary_list,1,0)
for para in paras:
    args.loss_c = para[0]
    args.loss_d = para[1]
    args.loss_ic50 = para[2]
    args.loss_binary = para[3]
    n_kfolds = 5
    k=5
    auc_all = []
    true_datas = pd.DataFrame()
    predict_datas = pd.DataFrame()
    for n_kfold in range(n_kfolds):
        kfold = KFold(n_splits=k, shuffle=True, random_state=n_kfold)
        for train_index, test_index in kfold.split(np.arange(pos_num)):
            sampler = RandomSampler_ic50(res_ic50, res_binary, train_index, test_index, null_mask)
            model = mtigcn_regression(adj_mat=sampler.train_data, cell_exprs=exprs, drug_finger=drug_finger,
                           layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, device=args.device).to(args.device)
            opt = Optimizer_regression(model, args.loss_ic50,args.loss_c,args.loss_d,args.loss_binary,cell_sim, drug_sim, sampler.train_ic50, sampler.test_ic50, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                            pcc, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device).to(args.device)
            auc_data, true_data, predict_data = opt()
            auc_all.append(auc_data)
            true_datas = true_datas.append(translate_result(true_data))
            predict_datas = predict_datas.append(translate_result(predict_data))
    pd.DataFrame(true_datas).to_csv("./result_data/{}_true_data.csv".format(args.data))
    pd.DataFrame(predict_datas).to_csv("./result_data/{}_predict_data.csv".format(args.data))
    file_write_obj = open("./result_data/resultall_id{}".format(0), 'a+')
    file_write_obj.writelines("loss_ic50: {}|loss_c: {}|loss_d: {}|loss_binary: {}|final auc: {}".format(str(args.loss_ic50),str(args.loss_c),str(args.loss_d),str(args.loss_binary),str(np.mean(auc_all))))
    file_write_obj.write('\n')
    file_write_obj.close()
