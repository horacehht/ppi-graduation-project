import json
import os
import os.path as osp
import time
import argparse
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from pipr_models import PIPRModel, SHS27K
from utils import Metrictor_PPI, print_file

parser = argparse.ArgumentParser(description="PIPR_model_training pytorch implementation")

parser.add_argument('--ppi_path', type=str, help="ppi path")
parser.add_argument('--pseq_path', type=str, help="protein sequence path")
parser.add_argument('--vec_path', type=str, help="amino acid vector path")
parser.add_argument('--index_path', type=str, help="the dataset partition json file path")
parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
parser.add_argument('--epoch', type=int, default=150, help="training epochs")
parser.add_argument('--save_path', type=str, default="./result_save", help="saved model location")
parser.add_argument('--seed', type=int, default=1, help="the specific seed corresponding to the experiment")


def train(train_loader, valid_loader, model, loss_fn, optimizer, device, save_path, 
          epochs, summary_writer, result_file_path, train_steps, valid_steps):
    global_step = 0  # 用于记录过了多少个batch
    global_best_valid_f1 = 0.0  # 用于记录最好的F1值
    global_best_valid_f1_epoch = 0  # 用于记录最好F1值在第几个epoch
    
    for epoch in range(epochs):
        
        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0  # 损失
        
        model.train()
        for pro2seq, label in train_loader:
            seq1 = pro2seq["p1"].transpose(1, 2).float().to(device)  # 蛋白1
            seq2 = pro2seq["p2"].transpose(1, 2).float().to(device)  # 蛋白2
            output = model(seq1, seq2).to(device)
            label = label.float().to(device)
            loss = loss_fn(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            m = nn.Sigmoid()
            predict_proba = m(output).to(device)
            predict_label = (m(output) > 0.5).type(torch.FloatTensor).to(device)
            
            metrics = Metrictor_PPI(predict_label.cpu().data, label.cpu().data, predict_proba.cpu().data)
            metrics.show_result()
            
            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()
            
            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1
            print_file("epoch: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                       .format(epoch,  loss.item(), metrics.Precision, metrics.Recall, metrics.F1))
        # 保存当前训练模型的参数
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))  
        
        # ---------------------验证------------------------
        valid_pre_label_list = []  # 存预测出来的类别
        valid_label_list = []  # 
        true_prob_list = []
        valid_loss_sum = 0.0
        
        model.eval()
        
        with torch.no_grad():
            for pro2seq, label in valid_loader:
                seq1 = pro2seq["p1"].transpose(1, 2).float().to(device)  # 蛋白1
                seq2 = pro2seq["p2"].transpose(1, 2).float().to(device)  # 蛋白2
                output = model(seq1, seq2).to(device)
                label = label.float().to(device)
                loss = loss_fn(output, label)
                
                valid_loss_sum += loss.item()
                
                m = nn.Sigmoid()
                predict_label = (m(output) > 0.5).type(torch.FloatTensor).to(device)
                
                valid_pre_label_list.append(predict_label.cpu().data)
                valid_label_list.append(label.cpu().data)
                true_prob_list.append(m(output).cpu().data)
                
        valid_pre_label_list = torch.cat(valid_pre_label_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim=0)
        
        metrics = Metrictor_PPI(valid_pre_label_list, valid_label_list, true_prob_list)

        metrics.show_result()

        # 训练集的一些指标参数
        recall = recall_sum / train_steps
        precision = precision_sum / train_steps
        f1 = f1_sum / train_steps
        loss = loss_sum / train_steps
        
        # 验证集的一些指标参数
        valid_loss = valid_loss_sum / valid_steps
        
        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, 'pipr_model_valid_best.ckpt'))
        
        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)
        
        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch"
                .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1,
                        global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)

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
    epochs = args.epoch
    batch_size = args.batch_size
    save_path = args.save_path
    # ppi_path='./protein_info/protein.actions.SHS27k.STRING.pro2.txt'
    # pseq_path = "./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv"
    # vec_path = "./protein_info/vec5_CTC.txt"
    # train_valid_index_path = "./train_val_split_data/train_val_split_27.json"
    
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device)
    
    # 这里读取的就是HIGH-PPI的数据划分文件
    train_dataset = SHS27K(ppi_path, pseq_path, vec_path, train_valid_index_path)
    valid_dataset = SHS27K(ppi_path, pseq_path, vec_path, train_valid_index_path, TRAIN=False)
    
    # 训练配置
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
    loss_fn = nn.BCEWithLogitsLoss().to(device)  # multi-label可以用BCE，他会逐个类别求，然后平均
    model = PIPRModel(input_dim=13, hidden_dim=50, class_num=7).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True, eps=1e-5)  # 按照PIPR论文配置
    
    if not osp.exists(save_path):
        os.mkdir(save_path)
    
    train_steps = math.ceil(len(train_dataset) / batch_size)  # 训练集要跑多少个batch
    valid_steps = math.ceil(len(valid_dataset) / batch_size)  # 同上
    time_stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    save_path = osp.join(save_path, "pipr_training_seed_{}".format(str(seed)))
    result_file_path = osp.join(save_path, "valid_results.txt")  # 训练日志，记录每一个epoch训练和验证的指标（epoch-level）
    os.mkdir(save_path)
    
    summary_writer = SummaryWriter(save_path)
    
    train(train_loader, valid_loader, model, loss_fn, optimizer, device, save_path, 
          epochs, summary_writer, result_file_path, train_steps, valid_steps)
    
    
if __name__ == "__main__":
    main()