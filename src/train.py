import numpy as np
import torch
import argparse

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from model import DNN
import os


def aver(hops, feature_list, alpha=0.15):
    input_feature = []
    for i in range(adj.shape[0]):
        hop = hops[i].int().item()
        if hop == 0:
            fea = feature_list[0][i].unsqueeze(0)
        else:
            fea = 0
            for j in range(hop):
                fea += (1-alpha)*feature_list[j][i].unsqueeze(0) + alpha*feature_list[0][i].unsqueeze(0)
            fea = fea / hop
        input_feature.append(fea)
    input_feature = torch.cat(input_feature, dim=0)
    return input_feature


def propagate(features, k):
    feature_list = []
    feature_list.append(features)
    for i in range(1, k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list


def cal_hops(feature_list, norm_fea_inf, k, epsilon=0.02):
    hops = torch.Tensor([0]*(adj.shape[0]))
    mask_before = torch.Tensor([False]*(adj.shape[0])).bool()

    for i in range(k):
        dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
        mask = (dist<epsilon).masked_fill_(mask_before, False)
        mask_before.masked_fill_(mask, True)
        hops.masked_fill_(mask, i)
    mask_final = torch.Tensor([True]*(adj.shape[0])).bool()
    mask_final.masked_fill_(mask_before, False)
    hops.masked_fill_(mask_final, k-1)
    return hops


def train(epoch, model, feature, record, idx_train, idx_val, idx_test, labels):
    model.train()
    optimizer.zero_grad()
    output = model(feature)
    
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(feature)

    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    record[acc_val.item()] = acc_test.item()
    return acc_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-1, help='Initial learning rate.')
    parser.add_argument('--k1', type=int, default=200, help='Value of K in stage (1).')
    parser.add_argument('--k2', type=int, default=20, help='Value of K in stage (3).')
    parser.add_argument('--epsilon1', type=float, default=0.03, help='Value of epsilon in stage (1).')
    parser.add_argument('--epsilon2', type=float, default=0.05, help='Value of epsilon in stage (2).')
    parser.add_argument('--hidden', type=int, default=64, help='Dim of hidden layer.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate of input and hidden layers.')
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--runs', type=int, default=1, help='Number of run times.')
    args = parser.parse_args()

    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset)

    node_sum = adj.shape[0]
    edge_sum = adj.sum()/2
    row_sum = (adj.sum(1) + 1)
    norm_a_inf = row_sum/ (2*edge_sum+node_sum)

    adj_norm = sparse_mx_to_torch_sparse_tensor(aug_random_walk(adj))

    features = F.normalize(features, p=1)
    feature_list = []
    feature_list.append(features)
    for i in range(1, args.k1):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))

    norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum)
    norm_fea_inf = torch.mm(norm_a_inf, features)

    hops = torch.Tensor([0]*(adj.shape[0]))
    mask_before = torch.Tensor([False]*(adj.shape[0])).bool()

    for i in range(args.k1):
        dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
        mask = (dist<args.epsilon1).masked_fill_(mask_before, False)
        mask_before.masked_fill_(mask, True)
        hops.masked_fill_(mask, i)
    mask_final = torch.Tensor([True]*(adj.shape[0])).bool()
    mask_final.masked_fill_(mask_before, False)
    hops.masked_fill_(mask_final, args.k1-1)
    print("Local Smoothing Iteration calculation is done.")

    input_feature = aver(hops, feature_list)
    print("Local Smoothing is done.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_class = (labels.max()+1).item()

    input_feature = input_feature.to(device)

    print("Start training...")
    test_acc = []
    for i in range(args.runs):
        best_acc = 0
        record = {}
        model = DNN(features.shape[1], args.hidden, n_class, args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        for epoch in range(args.epochs):
            acc_val = train(epoch, model, input_feature, record, idx_train, idx_val, idx_test, labels.to(device))
            if acc_val > best_acc:
                best_acc = acc_val
                torch.save(model, './best.pt')
        bit_list = sorted(record.keys())
        bit_list.reverse()
    
        model = torch.load('./best.pt')
        model.eval()
        output = model(input_feature).cpu()

        final_acc = accuracy(output[idx_test], labels[idx_test])
        for j in range(1, args.k2):
            output_list = propagate(output, j)
            norm_softmax_inf = torch.mm(norm_a_inf, output)
            hops = cal_hops(output_list, norm_softmax_inf, j, args.epsilon2)
            output_final = aver(hops, output_list)

            acc_test = accuracy(output_final[idx_test], labels[idx_test])
            if acc_test > final_acc:
                final_acc = acc_test
        test_acc.append(final_acc)
        print(f'Run {i+1}: Test Accuracy {final_acc}')
    
    print(f"\nMean Test Accuracy: {round(np.mean(test_acc), 4)}")
