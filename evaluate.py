import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from preprocessing import RUL_Transformer_Dataset
from model_architecture import RUL_Transformer


def cycle_consistency_visualize(one_to_one=True):
    data = RUL_Transformer_Dataset()
    model = RUL_Transformer(14, 32).cuda()
    model.load_state_dict(torch.load('RUL_Transformer_ep1000.pth'))
    model.eval()
    seq, src_len = data[0:2]
    with torch.no_grad():
        emb = model(torch.tensor(seq).cuda().float()).detach().cpu().numpy()
    seq1, seq2 = emb[0, :src_len[0]//4], emb[1, :src_len[1]//4]
    
    if one_to_one:
        for i, p in enumerate(seq1):
            target = nearest(p, seq2)
            target_back = nearest(seq2[target], seq1)
            if i==target_back:
                plt.plot([target*4, i*4], [0, 1], c='red')
    else:
        for i, p in enumerate(seq1):
            target = nearest(p, seq2)
            plt.plot([target*4, i*4], [0, 1], c='red')
    plt.scatter(np.arange(src_len[0]//4)*4, np.ones((src_len[0])//4), c='black', s=3)
    plt.scatter(np.arange(src_len[1]//4)*4, np.zeros((src_len[1])//4), c='black', s=3)
    plt.savefig('connection.png')
    plt.close()


def nearest(p, seq):
    shortest = 10000
    for i, q in enumerate(seq):
        dist = np.sum((p-q)**2)
        if dist<shortest:
            shortest=dist
            target = i
    return target
cycle_consistency_visualize()