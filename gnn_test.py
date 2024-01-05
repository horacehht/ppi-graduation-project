import os
import os.path as osp
import time
import math
import json
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm

from gnn_data import GNN_DATA
from gnn_models import GIN_Net2
from utils import Metrictor_PPI, print_file

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Test Model')
parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default=None, type=str,
                    help='protein sequence vector path')
parser.add_argument('--index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--save_path', default=None, type=str,
                    help="gnn trained model")
parser.add_argument('--seed', default=1, type=int,
                    help='random seed which belongs to an experiment')

def test(model, graph, test_mask, device, save_path):
    valid_pre_label_list = []
    valid_label_list = []
    true_prob_list = []

    model.eval()

    batch_size = 256

    valid_steps = math.ceil(len(test_mask) / batch_size)
    with torch.no_grad():
        for step in tqdm(range(valid_steps)):
            if step == valid_steps-1:
                valid_edge_id = test_mask[step*batch_size:]
            else:
                valid_edge_id = test_mask[step*batch_size : step*batch_size + batch_size]

            output = model(graph.x, graph.edge_index, valid_edge_id)
            label = graph.edge_attr_1[valid_edge_id]
            label = label.type(torch.FloatTensor).to(device)

            m = nn.Sigmoid()
            predict_label = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            valid_pre_label_list.append(predict_label.cpu().data)
            valid_label_list.append(label.cpu().data)
            true_prob_list.append(m(output).cpu().data)

    valid_pre_label_list = torch.cat(valid_pre_label_list, dim=0)
    valid_label_list = torch.cat(valid_label_list, dim=0)
    true_prob_list = torch.cat(true_prob_list, dim=0)

    res_path = osp.join(save_path, "predict and label.pt")
    torch.save({'predict': true_prob_list, 'pre_label': valid_pre_label_list, 'label': valid_label_list}, res_path)
    
    metrics = Metrictor_PPI(valid_pre_label_list, valid_label_list, true_prob_list)

    metrics.show_result()
    print('recall: {}, precision: {}, F1: {}, AUPRC: {}'.format(metrics.Recall, metrics.Precision, \
    metrics.F1, metrics.Aupr))

def main():

    args = parser.parse_args()
    seed = args.seed

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)

    ppi_data.generate_data()

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    truth_edge_num = len(ppi_list) // 2
    # fake_edge_num = len(ppi_data.fake_edge) // 2
    fake_edge_num = 0
    
    with open(args.index_path, 'r') as f:
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
    model = GIN_Net2(in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1, pool_size=3, cnn_hidden=1)
    save_path = args.save_path
    save_path = osp.join(save_path, "gnn_training_seed_{}".format(str(seed)))
    model_path = osp.join(save_path, "gnn_model_valid_best.ckpt")
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.to(device)
    graph.to(device)

    test(model, graph, graph.val_mask, device, save_path)

if __name__ == "__main__":
    main()