import torch.nn
import torch.nn.functional as fun
from abc import ABC
import torch.optim as optim
from myutils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
import umap
import umap.plot
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns
from sklearn.metrics import silhouette_score,davies_bouldin_score,adjusted_rand_score,normalized_mutual_info_score
from tslearn.metrics import cdist_dtw
from sklearn.cluster import KMeans

class ConstructAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, device="cpu"):
        super(ConstructAdjMatrix, self).__init__()
        self.adj = original_adj_mat.to(device)
        self.device = device

    def forward(self):
        d_x = torch.diag(torch.pow(torch.sum(self.adj, dim=1)+1, -0.5))
        d_y = torch.diag(torch.pow(torch.sum(self.adj, dim=0)+1, -0.5))

        agg_cell_lp = torch.mm(torch.mm(d_x, self.adj), d_y)
        agg_drug_lp = torch.mm(torch.mm(d_y, self.adj.T), d_x)

        d_c = torch.pow(torch.sum(self.adj, dim=1)+1, -1)
        self_cell_lp = torch.diag(torch.add(d_c, 1))
        d_d = torch.pow(torch.sum(self.adj, dim=0)+1, -1)
        self_drug_lp = torch.diag(torch.add(d_d, 1))
        return agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp


class LoadFeature(nn.Module, ABC):
    def __init__(self, cell_exprs, drug_finger, device="cpu"):
        super(LoadFeature, self).__init__()
        cell_exprs = torch.from_numpy(cell_exprs).to(device)
        self.cell_feat = torch_z_normalized(cell_exprs,dim=1).to(device)
        self.drug_feat = torch.from_numpy(drug_finger).to(device)

    def forward(self):
        cell_feat = self.cell_feat
        drug_feat = self.drug_feat
        return cell_feat, drug_feat

class SelfAttention(nn.Module):
    def __init__(self, input_dim ,hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(input_dim, hid_dim)
        self.w_k = nn.Linear(input_dim, hid_dim)
        self.w_v = nn.Linear(input_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        if torch.cuda.is_available():
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()
        else:
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        if len(query.shape)>len(key.shape):
            bsz = query.shape[0]
        else:
            bsz = key.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        energy = torch.matmul(Q, K.T) / self.scale
        return self.fc(torch.matmul(self.do(fun.softmax(energy, dim=-1)), V))

class GEncoder(nn.Module, ABC):
    def __init__(self, agg_c_lp, agg_d_lp, self_c_lp, self_d_lp, cell_feat, drug_feat, layer_size, alpha):
        super(GEncoder, self).__init__()
        self.agg_c_lp = agg_c_lp
        self.agg_d_lp = agg_d_lp
        self.self_c_lp = self_c_lp
        self.self_d_lp = self_d_lp
        self.layers = layer_size
        self.alpha = alpha
        self.cell_feat = cell_feat
        self.drug_feat = drug_feat

        self.fc_cell = nn.Linear(self.cell_feat.shape[1], layer_size[0], bias=True)
        self.fc_drug = nn.Linear(self.drug_feat.shape[1], layer_size[0], bias=True)
        self.fc_cell1 = nn.Linear(self.cell_feat.shape[1], layer_size[0], bias=True)
        self.fc_drug1 = nn.Linear(self.drug_feat.shape[1], layer_size[0], bias=True)
        self.lc = nn.BatchNorm1d(layer_size[0])
        self.ld = nn.BatchNorm1d(layer_size[0])
        self.lc1 = nn.BatchNorm1d(layer_size[0])
        self.ld1 = nn.BatchNorm1d(layer_size[0])
        self.lm_cell = nn.Linear(layer_size[0], layer_size[1], bias=True)
        self.lm_drug = nn.Linear(layer_size[0], layer_size[1], bias=True)
        self.att1 = SelfAttention(256,256,1,0.2)
        self.att2 = SelfAttention(256,256,1,0.2)
        self.relu = torch.nn.LeakyReLU()
        self.relu1 = torch.nn.LeakyReLU()

    def forward(self):
        cell_fc = self.lc(self.fc_cell(self.cell_feat))
        drug_fc = self.ld(self.fc_drug(self.drug_feat))
        # cell_fc = self.fc_cell(self.cell_feat)
        # drug_fc = self.fc_drug(self.drug_feat)
        # cell_fc = self.att1(cell_f,drug_f,drug_f)+cell_f
        # drug_fc = self.att2(drug_f,cell_f,cell_f)+drug_f
        cell_gcn = torch.mm(self.self_c_lp, cell_fc) + torch.mm(self.agg_c_lp, drug_fc)
        drug_gcn = torch.mm(self.self_d_lp, drug_fc) + torch.mm(self.agg_d_lp, cell_fc)
        cell_ni = torch.mul(cell_gcn, cell_fc)
        drug_ni = torch.mul(drug_gcn, drug_fc)
        
        cell_emb = fun.relu(self.lm_cell((1-self.alpha)*cell_gcn + self.alpha*cell_ni))
        drug_emb = fun.relu(self.lm_drug((1-self.alpha)*drug_gcn + self.alpha*drug_ni))
        # cell_emb = fun.relu(cell_fc)
        # drug_emb = fun.relu(drug_fc)
        return cell_emb, drug_emb
        
class GDecoder(nn.Module, ABC):
    def __init__(self,gamma):
        super(GDecoder, self).__init__()
        self.gamma = gamma

    def forward(self, cell_emb, drug_emb):
        #Corr = torch.matmul(cell_emb,drug_emb.T)
        Corr = torch_corr_x_y(cell_emb, drug_emb)
        output = scale_sigmoid(Corr, alpha=self.gamma)
        #output = scale_sigmoid(Corr, alpha=1)
        return output

class GDecoder_regression(nn.Module, ABC):
    def __init__(self,gamma):
        super(GDecoder_regression, self).__init__()
        self.gamma = gamma

    def forward(self, cell_emb, drug_emb):
        #output = torch.matmul(cell_emb,drug_emb.T)
        output = torch.mul(torch_corr_x_y(cell_emb, drug_emb),self.gamma)
        return output

class GDecoder_sim(nn.Module, ABC):
    def __init__(self,gamma):
        super(GDecoder_sim, self).__init__()
        self.gamma = gamma

    def forward(self, cell_emb, drug_emb):
        #output = torch.matmul(cell_emb,drug_emb.T)
        cell_sim = torch.sigmoid(torch_corr_x_y(cell_emb, cell_emb)*15)
        drug_sim = torch.sigmoid(torch_corr_x_y(drug_emb, drug_emb)*15)

        return cell_sim, drug_sim

class mtigcn_regression(nn.Module):
    def __init__(self, adj_mat, cell_exprs, drug_finger, layer_size, alpha, gamma,
                 device="cpu"):
        super(mtigcn_regression, self).__init__()
        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)
        loadfeat = LoadFeature(cell_exprs, drug_finger, device=device)

        agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp = construct_adj_matrix()
        cell_feat, drug_feat = loadfeat()
        self.encoder = GEncoder(agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp,
                                cell_feat, drug_feat, layer_size, alpha)
        self.decoder_regression = GDecoder_regression(gamma=15)
        self.decoder = GDecoder(gamma=gamma)
        self.decoder_sim = GDecoder_sim(gamma=gamma)
        self.fc_c = nn.Linear(layer_size[1], layer_size[1], bias=True)
        self.fc_d = nn.Linear(layer_size[1], layer_size[1], bias=True)
        self.fc_c1 = nn.Linear(layer_size[1], layer_size[1], bias=True)
        self.fc_d1 = nn.Linear(layer_size[1], layer_size[1], bias=True)


    def forward(self):
        cell_emb, drug_emb = self.encoder()
        cell_e = self.fc_c(cell_emb)
        drug_e = self.fc_d(drug_emb)
        cell_e1 = self.fc_c1(cell_emb)
        drug_e1 = self.fc_c1(drug_emb)
        out = self.decoder(cell_emb, drug_emb)
        output_ic50 = self.decoder_regression(cell_e, drug_e)
        out_cellsim, out_drugsim = self.decoder_sim(cell_e1, drug_e1)
        return out,output_ic50, out_cellsim, out_drugsim

class mtigcn(nn.Module):
    def __init__(self, adj_mat, cell_exprs, drug_finger, layer_size, alpha, gamma,
                 device="cpu"):
        super(mtigcn, self).__init__()
        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)
        loadfeat = LoadFeature(cell_exprs, drug_finger, device=device)

        agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp = construct_adj_matrix()
        cell_feat, drug_feat = loadfeat()
        self.encoder = GEncoder(agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp,
                                cell_feat, drug_feat, layer_size, alpha)
        self.decoder_regression = GDecoder_regression(gamma=15)
        self.decoder = GDecoder(gamma=gamma)
        self.decoder_sim = GDecoder_sim(gamma=gamma)
        self.fc_c = nn.Linear(layer_size[1], layer_size[1], bias=True)
        self.fc_d = nn.Linear(layer_size[1], layer_size[1], bias=True)
        self.fc_c1 = nn.Linear(layer_size[1], layer_size[1], bias=True)
        self.fc_d1 = nn.Linear(layer_size[1], layer_size[1], bias=True)

    def forward(self):
        cell_emb, drug_emb = self.encoder()
        cell_e = self.fc_c(cell_emb)
        drug_e = self.fc_d(drug_emb)
        cell_e1 = self.fc_c1(cell_emb)
        drug_e1 = self.fc_c1(drug_emb)
        out = self.decoder(cell_emb, drug_emb)
        output_ic50 = self.decoder_regression(cell_e, drug_e)
        out_cellsim, out_drugsim = self.decoder_sim(cell_e1, drug_e1)
        return out,output_ic50, out_cellsim, out_drugsim

# def get_embeds(self):
#     cell_emb, drug_emb = self.encoder()
#     return cell_emb, drug_emb


class Optimizer_regression(nn.Module):
    def __init__(self, model,  a,b,c,d,cell_sim, drug_sim,train_ic50, test_ic50, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.001, wd=1e-05, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer_regression, self).__init__()
        self.model = model.to(device)
        self.cell_sim = cell_sim.to(device)
        self.drug_sim = drug_sim.to(device)
        self.train_ic50= train_ic50.to(device)
        self.test_ic50 = test_ic50.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.wd = wd
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        best_pcc = 0
        true_data = torch.masked_select(self.test_ic50,self.test_mask)
        for epoch in torch.arange(self.epochs+1):
            predict_data,pre_ic50,out_cellsim, out_drugsim = self.model()
            loss_ic50 = mse_loss(self.train_ic50, pre_ic50, self.train_mask)
            loss_cellsim = mse_loss(self.cell_sim, out_cellsim, torch.ones(out_cellsim.shape).cuda())
            loss_drugsim = mse_loss(self.drug_sim, out_drugsim, torch.ones(out_drugsim.shape).cuda())
            loss_binary = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            loss = self.a*loss_ic50 +self.b*loss_cellsim +self.c*loss_drugsim + self.d*loss_binary
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_data_masked = torch.masked_select(pre_ic50, self.test_mask)
            pcc = self.evaluate_fun(true_data, predict_data_masked)
            if pcc > best_pcc:
                best_pcc = pcc
                best_predict = torch.masked_select(pre_ic50, self.test_mask)
            #     #torch.save(self.model.state_dict(), 'NIHGCN_' +args.dataset + id + '.pkl')
            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "pcc:%.4f" % pcc)

        print("Fit finished.")
        return best_pcc, true_data, best_predict

class Optimizer_mul(nn.Module):
    def __init__(self, model, a,b,c,d,cell_sim, drug_sim, train_ic50, test_ic50, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.001, wd=1e-05, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer_mul, self).__init__()
        self.model = model.to(device)
        self.cell_sim = cell_sim.to(device)
        self.drug_sim = drug_sim.to(device)
        self.train_ic50= train_ic50.to(device)
        self.test_ic50 = test_ic50.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.wd = wd
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        best_auc = 0
        true_data = torch.masked_select(self.test_data, self.test_mask)
        for epoch in torch.arange(self.epochs+1):
            predict_data,pre_ic50,out_cellsim, out_drugsim = self.model()
            loss_ic50 = mse_loss(self.train_ic50, pre_ic50, self.train_mask)
            loss_cellsim = mse_loss(self.cell_sim, out_cellsim, torch.ones(out_cellsim.shape).cuda())
            loss_drugsim = mse_loss(self.drug_sim, out_drugsim, torch.ones(out_drugsim.shape).cuda())
            loss_binary = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            loss = self.a*loss_ic50 +self.b*loss_cellsim +self.c*loss_drugsim + self.d*loss_binary
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            auc = self.evaluate_fun(true_data, predict_data_masked)
            if auc > best_auc:
                best_auc = auc
                best_predict = torch.masked_select(predict_data, self.test_mask)
                #torch.save(self.model.state_dict(), 'NIHGCN_' +args.dataset + id + '.pkl')
            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "auc:%.4f" % auc)
        print("Fit finished.")
        return best_auc, true_data, best_predict

class Optimizer(nn.Module, ABC):
    def __init__(self, model, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.001, wd=1e-05, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer, self).__init__()
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.test_mask = test_mask.to(device)
        self.train_mask = train_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        true_data = torch.masked_select(self.test_data, self.test_mask)
        best_auc = 0
        for epoch in torch.arange(self.epochs):
            predict_data, cell_emb, drug_emb = self.model()
            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            auc = self.evaluate_fun(true_data, predict_data_masked)
            if auc > best_auc:
                best_auc = auc
                best_predict = torch.masked_select(predict_data, self.test_mask)
                #torch.save(self.model.state_dict(), 'NIHGCN_' +args.dataset + id + '.pkl')
            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "auc:%.4f" % auc)
        # all_feature = torch.cat((z_cell, z_drug), dim=0).cpu().detach().numpy()
        # X_embedded = TSNE(n_components=2, init="pca").fit_transform(all_feature)
        # Y = np.zeros((z_cell.shape[0]),dtype=int).tolist()+np.ones((z_drug.shape[0]),dtype=int).tolist()
        # figure = plt.figure(figsize=(5, 5), dpi=80)
        # color = get_color(Y)  # 为6个点配置颜色
        # x = X_embedded[:, 0]  # 横坐标
        # y = X_embedded[:, 1]  # 纵坐标
        # plt.scatter(x, y, color=color)  # 绘制散点图。
        # plt.show()
        # pair_drug = sp.coo_matrix(self.test_mask[:,67].cpu()).row
        # pair_cell = sp.coo_matrix(self.test_mask[:,67].cpu()).col
        # #target = np.array(sp.coo_matrix((2 * self.test_mask[:, 67].int() - self.test_data[:, 67].int()).cpu()).data)
        # target = np.array(sp.coo_matrix((2*self.test_mask[:,67].int()-self.test_data[:,67].int()).cpu()).data).astype(str)
        # target = np.where(target=='1','Sensitive','Resistant')
        # #cell_drug_emb = torch.cat((cell_emb[pair_cell, :], drug_emb[pair_drug, :]), 1).cpu().detach().numpy()
        # cell_drug_emb = cell_emb[pair_cell, :].cpu().detach().numpy()
        # all_feature = cell_drug_emb
        # X_embedded = TSNE(n_components=2).fit_transform(all_feature)
        # Y = target
        # plt.figure(figsize=(4, 4), dpi=300)
        # color = get_color(Y)  # 为6个点配置颜色
        # x = X_embedded[:, 0]  # 横坐标
        # y = X_embedded[:, 1]  # 纵坐标
        # X_tsne_data = np.vstack((X_embedded.T, target)).T
        # df_tsne = pd.DataFrame(X_tsne_data, columns=['Dimension1', 'Dimension2', 'class'])
        # print(df_tsne)
        # sns.scatterplot(data=df_tsne,hue='class', x='Dimension1', y='Dimension2')  # 绘制散点图。
        # plt.show()

        # reducer = umap.UMAP()
        # embedding = reducer.fit_transform(all_feature)
        # X_tsne_data = np.vstack((embedding.T, target.astype(object))).T
        # df_tsne = pd.DataFrame(X_tsne_data, columns=['Dimension1', 'Dimension2', 'class'])
        # #plt.scatter(embedding[:, 0], embedding[:, 1], c=target, s=35)
        # plt.gca().set_aspect('equal', 'datalim')
        # #plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
        # #plt.title('UMAP')
        # sns.scatterplot(data=df_tsne,hue='class', x='Dimension1', y='Dimension2')  # 绘制散点图。
        # plt.show()
        print("Fit finished.")
        return true_data, best_predict

class Optimizer_visible(nn.Module, ABC):
    def __init__(self, i,model, exprs, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.001, wd=1e-05, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer_visible, self).__init__()
        self.i = i
        self.exprs = exprs
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.test_mask = test_mask.to(device)
        self.train_mask = train_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        best_auc = 0
        best_emb = 0
        true_data = torch.masked_select(self.test_data, self.test_mask)
        for epoch in torch.arange(self.epochs):
            predict_data, cell_emb, drug_emb = self.model()
            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            auc = self.evaluate_fun(true_data, predict_data_masked)
            if auc > best_auc:
                best_emb = cell_emb
                best_auc = auc
                best_predict = torch.masked_select(predict_data, self.test_mask)
                #torch.save(self.model.state_dict(), 'NIHGCN_' +args.dataset + id + '.pkl')
            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "auc:%.4f" % auc)
        # sns.heatmap(torch.mul(predict_data[:,67:70],self.test_mask[:,67:70]).cpu().detach().numpy(), cmap='Blues',vmin=0, vmax=1)
        # plt.xticks(rotation=45)
        # plt.yticks(rotation=45)
        # plt.show()
        # all_feature = torch.cat((z_cell, z_drug), dim=0).cpu().detach().numpy()
        # X_embedded = TSNE(n_components=2, init="pca").fit_transform(all_feature)
        # Y = np.zeros((z_cell.shape[0]),dtype=int).tolist()+np.ones((z_drug.shape[0]),dtype=int).tolist()
        # figure = plt.figure(figsize=(5, 5), dpi=80)
        # color = get_color(Y)  # 为6个点配置颜色
        # x = X_embedded[:, 0]  # 横坐标
        # y = X_embedded[:, 1]  # 纵坐标
        # plt.scatter(x, y, color=color)  # 绘制散点图。
        # plt.show()
        #GDSC PD-0332991:67 AT-7519:119 CGP-082996:159
        #CCLE PD-0332991:8
        # pair_drug = sp.coo_matrix(self.test_mask[:,119].cpu()).row
        # pair_cell = sp.coo_matrix(self.test_mask[:,119].cpu()).col
        # #mean1 = torch.mean(best_emb, dim=1).view([-1, 1])
        # #best_emb = torch.sub(best_emb, mean1).cpu().detach().numpy()
        # #best_emb = torch_z_normalized(best_emb.cpu().detach(),dim=0).numpy()
        # #best_emb = preprocessing.scale(best_emb.cpu().detach().numpy())
        # best_emb = best_emb.cpu().detach().numpy()
        # #target = np.array(sp.coo_matrix((2 * self.test_mask[:, 8].int() - self.test_data[:, 67].int()).cpu()).data)
        # true = self.test_data[pair_cell,119].cpu().detach().numpy()
        # #target = np.array(sp.coo_matrix((2*self.test_mask[:,67].int()-self.test_data[:,67].int()).cpu()).data).astype(str)
        # #target = np.where(target=='1','Sensitive','Resistant')
        # #cell_drug_emb = torch.cat((cell_emb[pair_cell, :], drug_emb[pair_drug, :]), 1).cpu().detach().numpy()
        # cell_drug_emb = best_emb[pair_cell, :]
        # all_feature = cell_drug_emb
        # model = KMeans(n_clusters=2)
        # model.fit(all_feature)  # 完成聚类
        # pred_y = model.predict(all_feature)
        # target = pred_y
        # reducer = umap.UMAP()
        # embedding = reducer.fit_transform(all_feature)
        # X_tsne_data = np.vstack((embedding.T, target.astype(object))).T
        # #X_tsne_data = np.vstack((embedding.T, true.astype(object))).T
        # sc_score = silhouette_score(embedding, target.astype(object), metric='euclidean')
        # DBI = davies_bouldin_score(embedding, target.astype(object))
        # ARI = adjusted_rand_score(true,pred_y)
        # NMI = normalized_mutual_info_score(true,pred_y)
        # print("-----------emb---------------")
        # print(str(self.i)+"sc:"+str(sc_score))
        # print(str(self.i) + "DBI:" + str(DBI))
        # print(str(self.i) + "ARI:" + str(ARI))
        # print(str(self.i) + "NMI:" + str(NMI))
        # df_tsne = pd.DataFrame(X_tsne_data, columns=['UMAP 1', 'UMAP 2', 'class'])
        # #plt.scatter(embedding[:, 0], embedding[:, 1], c=target, s=35)
        # plt.gca().set_aspect('equal', 'datalim')
        # #plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
        # #plt.title('PD-0332991 Learned')
        # sns.scatterplot(data=df_tsne,hue='class', x='UMAP 1', y='UMAP 2')  # 绘制散点图。
        # plt.savefig("GDSC-PD-A{}.png".format(self.i), dpi=600)
        # plt.close()
        #
        # all_feature_orgin = self.exprs[pair_cell, :]
        # model_orgin = KMeans(n_clusters=2)
        # model_orgin.fit(all_feature_orgin)  # 完成聚类
        # pred_y = model_orgin.predict(all_feature_orgin)
        # target = pred_y
        # reducer_in = umap.UMAP()
        # embedding_orgin = reducer_in.fit_transform(all_feature_orgin)
        # X_tsne_data_orgin = np.vstack((embedding_orgin.T, target.astype(object))).T
        # in_sc_score = silhouette_score(embedding_orgin, target.astype(object), metric='euclidean')
        # in_DBI = davies_bouldin_score(embedding_orgin, target.astype(object))
        # in_ARI = adjusted_rand_score(true, pred_y)
        # in_NMI = normalized_mutual_info_score(true,pred_y)
        # print("-----------origin---------------")
        # print(str(self.i) + "sc:" + str(in_sc_score))
        # print(str(self.i) + "DBI:" + str(in_DBI))
        # print(str(self.i) + "ARI:" + str(in_ARI))
        # print(str(self.i) + "NMI:" + str(in_NMI))
        # df_tsne_orgin = pd.DataFrame(X_tsne_data_orgin, columns=['UMAP 1', 'UMAP 2', 'class'])
        # #plt.scatter(embedding[:, 0], embedding[:, 1], c=target, s=35)
        # plt.gca().set_aspect('equal', 'datalim')
        # #plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
        # #plt.title('PD-0332991 Original')
        # sns.scatterplot(data=df_tsne_orgin,hue='class', x='UMAP 1', y='UMAP 2')  # 绘制散点图。
        # plt.savefig("GDSC-PD-B{}.png".format(self.i), dpi=600)
        # plt.close()
        print("Fit finished.")
        return true_data, best_predict

class Optimizer_earlystop(nn.Module, ABC):
    def __init__(self, model, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.001, wd=1e-05, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer_earlystop, self).__init__()
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.test_mask = test_mask.to(device)
        self.train_mask = train_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.test_freq = test_freq
        self.tolerance = 300
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        true_data = torch.masked_select(self.test_data, self.test_mask)
        early_stop = EarlyStop(tolerance=self.tolerance, data_len=true_data.size()[0])
        for epoch in torch.arange(self.epochs):
            predict_data, cell_emb, drug_emb = self.model()
            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            auc = self.evaluate_fun(true_data, predict_data_masked)
            if early_stop.stop(auc=auc, epoch=epoch, predict_data=torch.masked_select(predict_data, self.test_mask)):
                break
            if epoch % self.test_freq == 0:
                index = early_stop.get_best_index()
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "auc:%.4f" % early_stop.auc_pre[index])
        index = early_stop.get_best_index()
        best_predict = early_stop.predict_data_pre[index, :]
        print("AUC:{:^6.4f}".format(early_stop.auc_pre[index]))
        return true_data, best_predict

class Optimizer_case(nn.Module):
    def __init__(self, model, a,b,c,d,cell_sim, drug_sim, train_ic50, test_ic50, train_data, test_mask, train_mask, evaluate_fun,
                 lr=0.001, wd=1e-05, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer_case, self).__init__()
        self.model = model.to(device)
        self.cell_sim = cell_sim.to(device)
        self.drug_sim = drug_sim.to(device)
        self.train_ic50= train_ic50.to(device)
        self.test_ic50 = test_ic50.to(device)
        self.train_data = train_data.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.wd = wd
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        best_auc = 0
        for epoch in torch.arange(self.epochs+1):
            predict_data,pre_ic50,out_cellsim, out_drugsim = self.model()
            loss_ic50 = mse_loss(self.train_ic50, pre_ic50, self.train_mask)
            loss_cellsim = mse_loss(self.cell_sim, out_cellsim, torch.ones(out_cellsim.shape).cuda())
            loss_drugsim = mse_loss(self.drug_sim, out_drugsim, torch.ones(out_drugsim.shape).cuda())
            loss_binary = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            loss = self.a*loss_ic50 +self.b*loss_cellsim +self.c*loss_drugsim + self.d*loss_binary
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_data_masked = torch.mul(predict_data, self.test_mask)
            #predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            # if auc > best_auc:
            #     best_auc = auc
            #     best_predict = torch.masked_select(predict_data, self.test_mask)
                #torch.save(self.model.state_dict(), 'NIHGCN_' +args.dataset + id + '.pkl')
            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item())
        drug_indices = 57
        #drug_indices = 149
        vals, cell_indices = predict_data_masked[:,drug_indices].topk(k=10, largest=True, sorted=True)
        print(vals)
        print(cell_indices)
        print("Fit finished.")
        return predict_data_masked, cell_indices.cpu(),drug_indices

def get_color(labels):
    colors=["b","#C05757","#3939EF","y","o"]
    color=[]
    for i in range(len(labels)):
        color.append(colors[labels[i]])
    return color