import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from gnn_data import GNN_DATA
from torch.utils.data import DataLoader, Dataset

class RCNNUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim, pool_size, kernel_size=3):
        """
        RCNN Unit，1d CNN + Max pool 1D + BiGRU
        Args:
            input_dim (_type_): _description_
            hidden_dim (_type_): 
            kernel_size (_type_): 1d conv的卷积核大小
            pool_size (_type_): 1d max pooling的大小
        """
        super(RCNNUnit, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size)
        self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.pool = nn.MaxPool1d(pool_size)
        
    def forward(self, x):
        # 卷积提取局部特征
        x = self.conv(x)
        local_feat = self.pool(x)
        # 进双向的GRU提取时序特征
        local_feat = local_feat.transpose(1, 2)
        output, hidden = self.gru(local_feat)
        # 拿出GRU前向部分和反向部分，分别做残差，随后做concat
        forward_part = output[:, :, :self.hidden_dim]
        backward_part = output[:, :, self.hidden_dim:]
        output = torch.cat([local_feat + forward_part, local_feat + backward_part], dim=-1)
        
        return output.transpose(1, 2)
  
        
class PIPRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num):
        super(PIPRModel, self).__init__()
        self.encoder = nn.Sequential(
            RCNNUnit(input_dim, hidden_dim, pool_size=3, kernel_size=3),
            RCNNUnit(hidden_dim*2, hidden_dim, pool_size=3, kernel_size=3),
            RCNNUnit(hidden_dim*2, hidden_dim, pool_size=2, kernel_size=3),
            RCNNUnit(hidden_dim*2, hidden_dim, pool_size=2, kernel_size=3),
            RCNNUnit(hidden_dim*2, hidden_dim, pool_size=2, kernel_size=3),
            nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, (hidden_dim+7)//2),
            nn.LeakyReLU(0.3),
            nn.Linear((hidden_dim+7)//2, class_num)
        )
        
    def forward(self, seq1, seq2):
        # 蛋白1
        emb1 = self.encoder(seq1)
        protein1 = F.avg_pool1d(emb1, kernel_size=emb1.size(2)).squeeze(2)
        # 蛋白2
        emb2 = self.encoder(seq2)
        protein2 = F.avg_pool1d(emb2, kernel_size=emb2.size(2)).squeeze(2)
        # 分类器
        protein_pair = protein1*protein2
        logits = self.classifier(protein_pair)
        
        return logits


class SHS27K(Dataset):
    def __init__(self, ppi_path, pseq_path, vec_path, train_valid_index_path, TRAIN=True):
        super(SHS27K, self).__init__()
        self.TRAIN = TRAIN  # 是训练集还是验证集
        # 这里调用HIGH-PPI项目写好的数据接口
        ppi_data = GNN_DATA(ppi_path)
        ppi_data.get_feature_origin(pseq_path=pseq_path,
                                vec_path=vec_path)
        # 三个字典，方便通过索引获取蛋白质对应的array
        name2idx = ppi_data.protein_name  # 字典，蛋白名字-索引
        self.idx2name = dict(zip(name2idx.values(), name2idx.keys()))  # 反转键值，索引-蛋白
        self.protein_dict = ppi_data.protein_dict  # 蛋白-序列向量
        
        # ppi_list和ppi_label_list是同等长度的，相同索引是对应关系
        ppi_list = ppi_data.ppi_list
        ppi_list = ppi_list[:int(len(ppi_list) / 2)]  # e.g. [0, 1]
        ppi_label_list = ppi_data.ppi_label_list
        ppi_label_list = np.array(ppi_label_list[:int(len(ppi_label_list) / 2)])  # e.g. [0, 1]对应的multi-label
        
        # 读取HIGH-PPI的数据集划分文件
        f = open(train_valid_index_path, "r")
        train_valid_index = json.load(f)
        train_index = train_valid_index['train_index']
        valid_index = train_valid_index['valid_index']
        # 拿到划分后的训练集和验证集
        self.train_ppi_list = [ppi_list[index] for index in train_index]
        self.train_ppi_label_list = [ppi_label_list[index] for index in train_index]
        self.valid_ppi_list = [ppi_list[index] for index in valid_index]
        self.valid_ppi_label_list = [ppi_label_list[index] for index in valid_index]
        
        
    def __getitem__(self, ppi_index):
        label, p1, p2 = None, None, None
        if self.TRAIN:
            label = self.train_ppi_label_list[ppi_index]  # 标签
            p1, p2 = self.train_ppi_list[ppi_index]  # 取出ppi
        else:
            label = self.valid_ppi_label_list[ppi_index]  # 标签
            p1, p2 = self.valid_ppi_list[ppi_index]  # 取出ppi
        seq1 = self.protein_dict[self.idx2name[p1]]
        seq2 = self.protein_dict[self.idx2name[p2]]
        pro2seq = {"p1": seq1, "p2": seq2}
        
        return pro2seq, label
    
    def __len__(self):
        if self.TRAIN:
            return len(self.train_ppi_list)
        else:
            return len(self.valid_ppi_list)
        

if __name__ == '__main__':
    # 测试用例
    model = PIPRModel(input_dim=13, hidden_dim=50, class_num=7)
    seq1 = torch.randn((3, 13, 1000))  # []
    seq2 = torch.randn((3, 13, 1000))
    res = model(seq1, seq2)
    print(res.shape)

