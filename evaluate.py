import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from preprocessing import RUL_Transformer_Dataset
from model_architecture import RUL_Transformer



def cycle_consistency_visualize(one_to_one=True):
    model_names = [_ for _ in os.listdir() if _.endswith('.pth')]
    model_names.sort(key=lambda x:int(x[18:-4]))
    print(model_names)
    data_trn = RUL_Transformer_Dataset()
    data_val = RUL_Transformer_Dataset(train=False)
    seq, src_len = np.concatenate([data_trn[[1, 2]][0], data_val[[1, 2]][0]], axis=0), np.concatenate([data_trn[[1, 2]][1], data_val[[1, 2]][1]], axis=0)
    c_list = []
    model = RUL_Transformer(14, 32).cuda()
    model.load_state_dict(torch.load('RUL_Transformer_ep1900.pth'))
    model.eval()
    with torch.no_grad():
        emb = model(torch.tensor(seq).cuda().float()).detach().cpu().numpy()
    seq1, seq2 = emb[0, :src_len[0]//4], emb[3, :src_len[3]//4]
    # c = 0
    if one_to_one:
        for i, p in enumerate(seq1):
            target = nearest(p, seq2)
            target_back = nearest(seq2[target], seq1)
            if i==target_back:
                # c+=1
                plt.plot([target*4, i*4], [0, 1], c='red', lw=1)
    else:
        for i, p in enumerate(seq1):
            target = nearest(p, seq2)
            plt.plot([target*4, i*4], [0, 1], c='red', lw=1)
    # plt.plot(np.linspace(100, 2000, 20), c_list, c='blue')
    # plt.show()
    # plt.close()
    plt.scatter(np.arange(src_len[0]//4)*4, np.ones((src_len[0])//4), c='black', s=3)
    plt.scatter(np.arange(src_len[3]//4)*4, np.zeros((src_len[3])//4), c='black', s=3)
    plt.yticks(color='w')
    plt.show()
    # plt.savefig('connection'+str(ep)+'.png',dpi=300)
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