import torch
import torch.nn as nn
import torch.nn.functional as F


class RUL_Transformer(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.5):
        super(RUL_Transformer, self).__init__()
        self.linear_in = nn.Linear(in_ch, 64)
        self.transformer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=64, batch_first=True, dropout=0.)
        self.fc1 = nn.Sequential(nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64))
        self.conv_block = nn.Sequential(nn.Conv1d(64, 16, kernel_size=5, padding=2),
                                nn.MaxPool1d(2),
                                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                                nn.MaxPool1d(2))
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Sequential(nn.Linear(32, 128),
                                nn.ReLU(),
                                nn.Linear(128, out_ch))
        
    def forward(self, x, mask=None):
        """
        input shape->(seq_len, batch, ch)
        """
        x = self.linear_in(x)
        if mask is None:
            x = torch.add(self.transformer(x), x)
        else:
            x = torch.add(self.transformer(x, src_key_padding_mask=mask), x)
        x = torch.add(self.fc1(x), x)
        x = self.conv_block(x.transpose(2, 1))
        x = self.drop(x)
        out = self.fc2(x.transpose(2, 1))
        return out
    

class Cycle_Consistency_Loss(nn.Module):
    def __init__(self, penalty=0.01):
        super(Cycle_Consistency_Loss, self).__init__()
        self.penalty = penalty

    def forward(self, seq, src_len, combinations):
        loss_i, loss_j = 0, 0
        src_len = src_len//4
        for c in combinations:
            seq1, seq2 = seq[c[0], :src_len[c[0]]], seq[c[1], :src_len[c[1]]]
            for i, p in enumerate(seq1):
                d1 = torch.sum(torch.square(seq2-p.repeat(len(seq2), 1)), dim=1)
                alpha = F.softmin(d1, dim=0).reshape(-1, 1)
                snn = torch.sum(alpha.repeat(1, 32)*seq2, dim=0) #  soft nearest neighbor
                d2 = torch.sum(torch.square(seq1-snn.repeat(len(seq1), 1)), dim=1)
                beta = F.softmin(d2, dim=0)
                u_id = torch.dot(beta, torch.arange(len(seq1)).cuda().float())
                std_id = torch.dot(beta, torch.square(torch.arange(len(seq1)).cuda().float()-u_id))
                loss_i+=(torch.square(i-u_id)/std_id)+self.penalty*torch.log(torch.sqrt(std_id))
            for j, q in enumerate(seq2):
                d1 = torch.sum(torch.square(seq1-q.repeat(len(seq1), 1)), dim=1)
                alpha = F.softmin(d1, dim=0).reshape(-1, 1)
                snn = torch.sum(alpha.repeat(1, 32)*seq1, dim=0) #  soft nearest neighbor
                d2 = torch.sum(torch.square(seq2-snn.repeat(len(seq2), 1)), dim=1)
                beta = F.softmin(d2, dim=0)
                u_id = torch.dot(beta, torch.arange(len(seq2)).cuda().float())
                std_id = torch.dot(beta, torch.square(torch.arange(len(seq2)).cuda().float()-u_id))
                loss_j+=(torch.square(j-u_id)/std_id)+self.penalty*torch.log(torch.sqrt(std_id))
        return (loss_i+loss_j)/len(combinations)
    

