import os
import time
import math
import json
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import os.path as osp

from tqdm import tqdm
from gnn_models_sag import ppi_model
from gnn_data import GNN_DATA
from utils import Metrictor_PPI, print_file
parser = argparse.ArgumentParser(description='HIGH-PPI_model_training')
prefix = '/mnt/workspace/HIGH-PPI'

parser.add_argument('--ppi_path', default=osp.join(prefix, 'protein_info/protein.actions.SHS27k.STRING.pro2.txt'), type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=osp.join(prefix, 'protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv'), type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default=osp.join(prefix, 'protein_info/vec5_CTC.txt'), type=str,
                    help='protein sequence vector path')
parser.add_argument('--p_feat_matrix', default=osp.join(prefix, 'protein_info/x_list.pt'), type=str,
                    help="protein feature matrix")
parser.add_argument('--p_adj_matrix', default=osp.join(prefix, 'protein_info/edge_list_12.npy'), type=str,
                    help="protein adjacency matrix")
# parser.add_argument('--model_path', default=osp.join(prefix, 'result_save/gnn_training_seed_1/gnn_model_valid_best.ckpt'), type=str,
#                     help="path for trained model")
parser.add_argument('--save_path', default=osp.join(prefix, 'result_save'), type=str,
                    help="save folder")
parser.add_argument('--index_folder', default=osp.join(prefix, 'train_val_split_data'), type=str,
                    help='training and test PPI index')
parser.add_argument('--seed', default=1, type=int,
                    help="random seed which belongs to an experiment")


def multi2big_x(x_ori):
    x_cat = torch.zeros(1, 7)
    x_num_index = torch.zeros(1553)
    for i in range(1553):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index

def multi2big_batch(x_num_index):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,1553):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch

def multi2big_edge(edge_ori, num_index):
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(1553)
    for i in range(1553):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = torch.tensor(num_index[:i])
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def test(model, graph, test_mask, device,batch, p_x_all, p_edge_all, save_path):
    valid_pre_result_list = []
    valid_label_list = []

    model.eval()  # 验证模式

    batch_size = 64

    valid_steps = math.ceil(len(test_mask) / batch_size)
    
    valid_pre_result_list = []
    valid_label_list = []
    true_prob_list = []
    with torch.no_grad():
        for step in tqdm(range(valid_steps)):
            if step == valid_steps-1:
                valid_edge_id = test_mask[step*batch_size:]
            else:
                valid_edge_id = test_mask[step*batch_size : step*batch_size + batch_size]

            # output = model(graph.x, graph.edge_index, valid_edge_id)
            output = model(batch, p_x_all, p_edge_all, graph.edge_index, valid_edge_id)
            label = graph.edge_attr_1[valid_edge_id]
            label = label.type(torch.FloatTensor).to(device)

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            valid_pre_result_list.append(pre_result.cpu().data)
            valid_label_list.append(label.cpu().data)
            true_prob_list.append(m(output).cpu().data)

    valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)  # pre_result是经过分类阈值后的结果
    valid_label_list = torch.cat(valid_label_list, dim=0)  # 这个是验证集真实的标签
    true_prob_list = torch.cat(true_prob_list, dim = 0)  # 这个是经过sigmoid后，但没经过阈值的结果
    # 将预测和真实的存起来
    res_path = os.path.join(save_path, "predict and label.pt")
    torch.save({'predict': true_prob_list, 'label': valid_label_list}, res_path)

    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

    metrics.show_result()

    print('recall: {}, precision: {}, F1: {}, AUPRC: {}'.format(metrics.Recall, metrics.Precision, \
        metrics.F1, metrics.Aupr))
    print(valid_pre_result_list)
    print(valid_label_list)

def main():

    args = parser.parse_args()
    seed = args.seed
    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    ppi_data.get_feature_origin(pseq_path=args.pseq_path,
                                vec_path=args.vec_path)


    ppi_data.generate_data()

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    truth_edge_num = len(ppi_list) // 2
    # fake_edge_num = len(ppi_data.fake_edge) // 2
    fake_edge_num = 0

    index_path = osp.join(args.index_folder, "train_val_split_{}.json".format(str(seed)))  # 
    with open(index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()
    graph.train_mask = index_dict['train_index']

    graph.val_mask = index_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    node_vision_dict = {}
    for index in graph.train_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 1
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 1

    for index in graph.val_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 0
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 0
    
    vision_num = 0
    unvision_num = 0
    for node in node_vision_dict:
        if node_vision_dict[node] == 1:
            vision_num += 1
        elif node_vision_dict[node] == 0:
            unvision_num += 1
    print("vision node num: {}, unvision node num: {}".format(vision_num, unvision_num))

    test1_mask = []
    test2_mask = []
    test3_mask = []

    for index in graph.val_mask:
        ppi = ppi_list[index]
        temp = node_vision_dict[ppi[0]] + node_vision_dict[ppi[1]]
        if temp == 2:
            test1_mask.append(index)
        elif temp == 1:
            test2_mask.append(index)
        elif temp == 0:
            test3_mask.append(index)
    print("test1 edge num: {}, test2 edge num: {}, test3 edge num: {}".format(len(test1_mask), len(test2_mask), len(test3_mask)))

    graph.test1_mask = test1_mask
    graph.test2_mask = test2_mask
    graph.test3_mask = test3_mask

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ppi_model()
    model.to(device)
    model_path = osp.join(args.save_path, "high_training_seed_{}/gnn_model_valid_best.ckpt".format(str(seed)))
    model.load_state_dict(torch.load(model_path)['state_dict'])

    graph.to(device)





    p_x_all = torch.load(args.p_feat_matrix)
    # p_edge_all = np.load('/apdcephfs/share_1364275/kaithgao/edge_list_12.npy', allow_pickle=True)
    p_edge_all = np.load(args.p_adj_matrix, allow_pickle=True)
    p_x_all, x_num_index = multi2big_x(p_x_all)
    # p_x_all = p_x_all[:,torch.arange(p_x_all.size(1))!=6] 
    p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)


    batch = multi2big_batch(x_num_index)+1


    save_path = args.save_path
    save_path = os.path.join(save_path, "high_training_seed_{}".format(str(seed)))


    test(model, graph, graph.val_mask, device,batch, p_x_all, p_edge_all, save_path)
    
if __name__ == "__main__":
    main()
