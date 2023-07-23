from model import nihgcn_mul, Optimizer_mul
from sampler import SingleSampler_ic50

def mtigcn_single(cell_sim, drug_sim, cell_exprs, drug_finger, res_ic50, res_mat, null_mask, target_index,train_index, test_index,
               evaluate_fun, args):
    sampler = SingleSampler_ic50(res_ic50, res_mat, null_mask, target_index, train_index, test_index)
    model = nihgcn_mul(sampler.train_data, cell_exprs=cell_exprs, drug_finger=drug_finger,
                       layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, device=args.device)
    opt = Optimizer_mul(model, 0.1, 0.1, 0.25, 0.75, cell_sim, drug_sim, sampler.train_ic50, sampler.test_ic50,
                        sampler.train_data, sampler.test_data,
                        sampler.test_mask, sampler.train_mask, evaluate_fun,
                        lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device)
    auc, true_data, predict_data = opt()
    return true_data, predict_data
