import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import scipy.sparse as sp
import random
import gc

from clac_metric import get_metrics
from utils import constructHNet, constructNet, get_edge_index, Sizes, set_output,Print
import torch as t
from torch import optim
from loss import Myloss

import CasMFGCL




def train(model, train_data, optimizer, sizes):
    model.train()
    regression_crit = Myloss()

    def train_epoch():
        model.zero_grad()
        score, con_loss = model(train_data)
        loss = regression_crit(train_data['Y_train'], score, model.dis_l, model.mic_l, model.alpha1,
                               model.alpha2, sizes, con_loss)
        model.alpha1 = t.mm(
            t.mm((t.mm(model.dis_k, model.dis_k) + model.lambda1 * model.dis_l).inverse(), model.dis_k),
            2 * train_data['Y_train'] - t.mm(model.alpha2.T, model.mic_k.T)).detach()
        model.alpha2 = t.mm(t.mm((t.mm(model.mic_k, model.mic_k) + model.lambda2 * model.mic_l).inverse(), model.mic_k),
                            2 * train_data['Y_train'].T - t.mm(model.alpha1.T, model.dis_k.T)).detach()
        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(1, sizes.epoch + 1):
        train_reg_loss = train_epoch()
        print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss.item()))
        Print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss.item()), output, timestamp=True)
    pass


def PredictScore(train_dis_mic_matrix, dis_matrix, mic_matrix, seed, sizes):
    np.random.seed(seed)
    train_data = {}
    train_data['Y_train'] = t.DoubleTensor(train_dis_mic_matrix)
    Heter_adj = constructHNet(train_dis_mic_matrix, dis_matrix, mic_matrix)


    Heter_adj = t.FloatTensor(Heter_adj)
    Heter_adj_edge_index = get_edge_index(Heter_adj)
    print(Heter_adj_edge_index)
    train_data['Adj'] = {'data': Heter_adj, 'edge_index': Heter_adj_edge_index}

    X = constructNet(train_dis_mic_matrix)
    X = t.FloatTensor(X)
    train_data['feature'] = X

    model = CasMFGCL.Model(sizes, dis_matrix, mic_matrix)
    print(model)
    for parameters in model.parameters():
        print(parameters, ':', parameters.size())

    optimizer = optim.Adam(model.parameters(), lr=sizes.learn_rate)

    train(model, train_data, optimizer, sizes)
    return model(train_data)


def random_index(index_matrix, sizes):
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)
    random.shuffle(random_index)
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp


def crossval_index(dis_mic_matrix, sizes):
    random.seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(dis_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(dis_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = [pos_index[i] + neg_index[i] for i in range(sizes.k_fold)]
    return index


def cross_validation_experiment(dis_mic_matrix, dis_matrix, mic_matrix, sizes):
    index = crossval_index(dis_mic_matrix, sizes)
    metric = np.zeros((1, 7))
    pre_matrix = np.zeros(dis_mic_matrix.shape)
    print("seed=%d, evaluating dis-microbe...." % (sizes.seed))
    Print("seed=%d, evaluating dis-microbe...." % (sizes.seed), output, timestamp=True)
    print(sizes.k_fold)
    for k in range(sizes.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        Print("------this is %dth cross validation------" % (k + 1), output, timestamp=True)
        train_matrix = np.matrix(dis_mic_matrix, copy=True)
        train_matrix[tuple(np.array(index[k]).T)] = 0
        dis_len = dis_mic_matrix.shape[0]
        dis_len = dis_mic_matrix.shape[1]
        dis_mic_res, _ = PredictScore(
            train_matrix, dis_matrix, mic_matrix, sizes.seed, sizes)
        predict_y_proba = dis_mic_res.reshape(dis_len, dis_len).detach().numpy()
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]
        metric_tmp = get_metrics(dis_mic_matrix[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)])

        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / sizes.k_fold)
    metric = np.array(metric / sizes.k_fold)
    return metric, pre_matrix


if __name__ == "__main__":

    dataname = "HMDAD"
    data_path = '../data/'
    data_set = dataname + '/'
    output, save_prefix = set_output("./results/log", dataname)

    dis_sim = np.loadtxt(data_path + data_set + 'diseasesimilarity.txt', delimiter='\t')
    mic_sim = np.loadtxt(data_path + data_set + 'microbesimilarity.txt', delimiter='\t')
    adj_triple = np.loadtxt(data_path + data_set + 'adj.txt')
    dis_mic_matrix = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),
                                    shape=(len(dis_sim), len(mic_sim))).toarray()

    average_result = np.zeros((1, 7), float)
    circle_time = 1
    sizes = Sizes(dis_sim.shape[0], mic_sim.shape[0])
    results = []

    result, pre_matrix = cross_validation_experiment(
        dis_mic_matrix, dis_sim, mic_sim, sizes)
    np.savetxt('pre_matrix_'+dataname+'.txt', pre_matrix)
    print(list(sizes.__dict__.values()) + result.tolist()[0][:2])
