# coding: utf-8
import argparse
from load_data import load_data
from sampler import ExterSampler_ic50
from model import mtigcn,Optimizer_mul
from myutils import *

parser = argparse.ArgumentParser(description="Run MTIGCN")
parser.add_argument('-device', type=str, default="cuda:0",
                    help='cuda:number or cpu')
parser.add_argument('-data', type=str, default='tcga',
                    help='Dataset{pdx or tcga}')
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
res_ic50, res, drug_finger, exprs, null_mask, train_num, args = load_data(args)
cell_sim = exp_similarity(torch.from_numpy(exprs), sigma=torch.tensor(2), normalize=True)
drug_sim = jaccard_coef(torch.from_numpy(drug_finger))
args.epochs = 1000
true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()
n_kfolds = 25
for n_kfold in range(n_kfolds):
    train_index = np.arange(train_num)
    test_index = np.arange(res.shape[0]-train_num) + train_num
    sampler = ExterSampler_ic50(res_ic50, res, null_mask, train_index, test_index)
    model = mtigcn(adj_mat=sampler.train_data, cell_exprs=exprs, drug_finger=drug_finger,
                       layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, device=args.device).to(
        args.device)
    opt = Optimizer_mul(model, 0.01,0.1,0.1,0.75,cell_sim,drug_sim,sampler.train_ic50, sampler.test_ic50, sampler.train_data, sampler.test_data,
                        sampler.test_mask, sampler.train_mask,
                        roc_auc, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device).to(args.device)
    auc,true_data, predict_data = opt()
    true_datas = true_datas.append(translate_result(true_data))
    predict_datas = predict_datas.append(translate_result(predict_data))
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")

