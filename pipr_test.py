import os
import os.path as osp
import argparse
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipr_models import PIPRModel, SHS27K
from utils import Metrictor_PPI

parser = argparse.ArgumentParser(description="PIPR_model_training pytorch implementation")

parser.add_argument('--ppi_path', type=str, help="ppi path")
parser.add_argument('--pseq_path', type=str, help="protein sequence path")
parser.add_argument('--vec_path', type=str, help="amino acid vector path")
parser.add_argument('--index_path', type=str, help="the dataset partition json file path")
parser.add_argument('--batch_size', type=int, default=256, help="batch size")
parser.add_argument('--save_path', type=str, default="./result_save", help="saved model location")
parser.add_argument('--seed', type=int, default=1, help="the specific seed corresponding to the experiment")


def test(valid_loader, model, device, save_path):
    # ---------------------验证------------------------
    valid_pre_label_list = []  # 存预测出来的类别
    valid_label_list = []  # 
    true_prob_list = []
    
    model.eval()
    
    with torch.no_grad():
        for pro2seq, label in valid_loader:
            seq1 = pro2seq["p1"].transpose(1, 2).float().to(device)  # 蛋白1
            seq2 = pro2seq["p2"].transpose(1, 2).float().to(device)  # 蛋白2
            output = model(seq1, seq2).to(device)
            label = label.float().to(device)
            
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    ppi_path = args.ppi_path
    pseq_path = args.pseq_path
    vec_path = args.vec_path
    train_valid_index_path = args.index_path
    batch_size = args.batch_size
    save_path = args.save_path
    # ppi_path='./protein_info/protein.actions.SHS27k.STRING.pro2.txt'
    # pseq_path = "./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv"
    # vec_path = "./protein_info/vec5_CTC.txt"
    # train_valid_index_path = "./train_val_split_data/train_val_split_27.json"
    
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device)
    
    # 读入数据集
    valid_dataset = SHS27K(ppi_path, pseq_path, vec_path, train_valid_index_path, TRAIN=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
    
    model = PIPRModel(input_dim=13, hidden_dim=50, class_num=7).float().to(device)
    model_path = osp.join(args.save_path, "pipr_training_seed_{}".format(str(seed)), "pipr_model_valid_best.ckpt")
    model.load_state_dict(torch.load(model_path)['state_dict'])  # 载入模型
    model.to(device)
    
    save_path = osp.join(save_path, "pipr_training_seed_{}".format(str(seed)))
    
    test(valid_loader, model, device, save_path)
    
    
if __name__ == "__main__":
    main()