import torch as t
import numpy as np
from torch import nn
from torch_geometric.nn import conv
from utils import *
from dgi import DGI

class Model(nn.Module):
    def __init__(self, sizes, drug_sim, mic_sim):
        super(Model, self).__init__()
        np.random.seed(sizes.seed)
        t.manual_seed(sizes.seed)
        self.drug_size = sizes.drug_size
        self.mic_size = sizes.mic_size
        self.F1 = sizes.F1
        self.F2 = sizes.F2
        self.F3 = sizes.F3
        self.seed = sizes.seed
        self.h1_gamma = sizes.h1_gamma
        self.h2_gamma = sizes.h2_gamma
        self.h3_gamma = sizes.h3_gamma

        self.lambda1 = sizes.lambda1
        self.lambda2 = sizes.lambda2

        self.kernel_len = 4
        self.drug_ps = t.ones(self.kernel_len) / self.kernel_len
        self.mic_ps = t.ones(self.kernel_len) / self.kernel_len

        self.drug_sim = t.DoubleTensor(drug_sim)
        self.mic_sim = t.DoubleTensor(mic_sim)

        self.gcn_1 = conv.GCNConv(self.drug_size + self.mic_size, self.F1)
        # ---------------------------------------
        # self.dgi = DGI(self.drug_size + self.mic_size, self.F1, 'prelu')
        # self.b_xent = nn.BCEWithLogitsLoss()
        # ---------------------------------------
        self.gcn_2 = conv.GCNConv(self.F1, self.F2)
        self.gcn_3 = conv.GCNConv(self.F2, self.F3)

        self.alpha1 = t.randn(self.drug_size, self.mic_size).double()
        self.alpha2 = t.randn(self.mic_size, self.drug_size).double()

        self.drug_l = []
        self.mic_l = []

        self.drug_k = []
        self.mic_k = []

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, t.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return t.sum(t.mul(x1, x2), 1)

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        one = t.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = t.sum(-t.log(1e-8 + t.sigmoid(pos))-t.log(1e-8 + (one - t.sigmoid(neg1))))
        return con_loss

    def forward(self, input):
        t.manual_seed(self.seed)
        x = input['feature']
        adj = input['Adj']
        drugs_kernels = []
        mic_kernels = []

        # -------------------------------------------
        # idx = np.random.permutation(self.drug_size + self.mic_size)
        # shuf_fts = x[idx, :]
        # lbl_1 = t.ones(1, self.drug_size + self.mic_size)
        # lbl_2 = t.zeros(1, self.drug_size + self.mic_size)
        # lbl = t.cat((lbl_1, lbl_2), 1)
        # logits = self.dgi(x, shuf_fts, adj, False, None, None, None)
        # con_loss = self.b_xent(logits, lbl)
        # x, _ = self.dgi.embed(x, adj, False, None)
        # H1 = t.relu(x)
        # -------------------------------------------

        H1 = t.relu(self.gcn_1(x, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H1[:self.drug_size].clone(), 0, self.h1_gamma, True).double()))
        mic_kernels.append(t.DoubleTensor(getGipKernel(H1[self.drug_size:].clone(), 0, self.h1_gamma, True).double()))

        H2 = t.relu(self.gcn_2(H1, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H2[:self.drug_size].clone(), 0, self.h2_gamma, True).double()))
        mic_kernels.append(t.DoubleTensor(getGipKernel(H2[self.drug_size:].clone(), 0, self.h2_gamma, True).double()))

        H3 = t.relu(self.gcn_3(H2, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
        drugs_kernels.append(t.DoubleTensor(getGipKernel(H3[:self.drug_size].clone(), 0, self.h3_gamma, True).double()))
        mic_kernels.append(t.DoubleTensor(getGipKernel(H3[self.drug_size:].clone(), 0, self.h3_gamma, True).double()))

        drugs_kernels.append(self.drug_sim)
        mic_kernels.append(self.mic_sim)

        drug_k = sum([self.drug_ps[i] * drugs_kernels[i] for i in range(len(self.drug_ps))])
        self.drug_k = normalized_kernel(drug_k)
        mic_k = sum([self.mic_ps[i] * mic_kernels[i] for i in range(len(self.mic_ps))])
        self.mic_k = normalized_kernel(mic_k)
        self.drug_l = laplacian(drug_k)
        self.mic_l = laplacian(mic_k)

        # -------------------------------------------
        con_loss_1 = self.SSL(drug_k, self.drug_sim)
        con_loss_2 = self.SSL(mic_k, self.mic_sim)
        # -------------------------------------------

        out1 = t.mm(self.drug_k, self.alpha1)
        out2 = t.mm(self.mic_k, self.alpha2)

        out = (out1 + out2.T) / 2

        return out, con_loss_1+con_loss_2
