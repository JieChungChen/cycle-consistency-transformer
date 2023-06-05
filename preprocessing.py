import h5py
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset


def mat_to_npy(save_path='Severson_Dataset/extracted_info/'):
    filename = ['Severson_Dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat',
                'Severson_Dataset/2017-06-30_batchdata_updated_struct_errorcorrect.mat',
                'Severson_Dataset/2018-04-12_batchdata_updated_struct_errorcorrect.mat']
    # 各batch中discharge部分有問題的電池 要加以清理
    b1_err = [0, 1, 2, 3, 4, 8, 10, 12, 13, 18, 22, 5, 14, 15]
    b2_err = [1, 6, 9, 10, 21, 25, 7, 8, 15, 16]
    b3_err = [23, 32, 37]
    err_list = [b1_err, b2_err, b3_err]
    batch_name = ['b1c', 'b2c', 'b3c']
    for b in range(len(filename)): # batch數
        f = h5py.File(filename[b], 'r')
        batch = f['batch']
        num_cells = batch['summary'].shape[0]
        for i in range(num_cells): # 該batch下的電池cell數量
            if i in err_list[b]:
                print('skip err cell: batch %d, cell_id %d'%(b+1, i))
                continue
            cycle_life = np.array(f[batch['cycle_life'][i, 0]][0])
            Qd_summary = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            key = batch_name[b] + str(i).zfill(2)

            cycles = f[batch['cycles'][i, 0]]
            cycle_info = []
            for j in range(1, len(Qd_summary)): # 選擇前n個cyle
                temper = np.hstack((f[cycles['T'][j, 0]]))
                current = np.hstack((f[cycles['I'][j, 0]]))
                voltage = np.hstack((f[cycles['V'][j, 0]]))
                Qc = np.hstack((f[cycles['Qc'][j, 0]]))
                Qd = np.hstack((f[cycles['Qd'][j, 0]]))
                Qdd = np.diff(np.diff(Qd)) # 放電容量二次微分
                ch_s = 0 # 充電開始
                ch_e = np.where(current==0)[0][1] # 充電結束, 電流歸零
                dis_s = np.where(np.diff(Qd)>=1e-3)[0][0] # 放電開始
                dis_e = np.where(Qdd>1e-4)[0][-1]+1 # 放電結束
                
                discharge_info = feature_extraction([voltage[dis_s:dis_e], current[dis_s:dis_e], temper[dis_s:dis_e]])
                discharge_info = np.append(discharge_info, Qd_summary[j])
                cycle_info.append(np.expand_dims(discharge_info, axis=0))
            np.save(save_path+key+'_dis_info', np.concatenate(cycle_info, axis=0))
            print(np.concatenate(cycle_info, axis=0).shape)
            print(key+' finished')


def feature_extraction(seq):
    """extract mean, standard deviation, skewness, kurtosis, energy and power(3*6=18 features)"""
    feature_list = []
    better_feature_id = [0,1,2,3,4,5,6,7,8,9,10,11,13]
    for s in seq:
        feature_list.append(np.mean(s)) 
        feature_list.append(np.std(s))  
        feature_list.append(skew(s)) 
        feature_list.append(kurtosis(s))  
        feature_list.append(np.sum(s**2))
        feature_list.append(np.log(np.mean(s**2)))
    return np.array(feature_list)[better_feature_id]


def train_val_split(train_ratio=0.8, seed=15):
    load_path = 'Severson_Dataset/extracted_info/'
    filename = sorted(os.listdir(load_path))
    features = []
    for i in range((len(filename))):
        features.append(np.load(load_path+filename[i]))
    np.random.seed(seed)
    np.random.shuffle(features) 
    split_point = int(len(filename)*train_ratio)
    return features[:split_point], features[split_point:]

    
class RUL_Transformer_Dataset(Dataset):
    def __init__(self, train=True, norm=True):
        """
        train(bool): training or testing set
        norm(bool): apply normalization to target 
        """
        self.train = train
        self.trn, self.val = train_val_split()
        f_max, f_min = np.max(np.concatenate(self.trn), axis=0).reshape(1, -1), np.min(np.concatenate(self.trn), axis=0).reshape(1, -1)
        scale = (f_max - f_min).reshape(1, -1)
        self.trn_len, self.val_len = [], []
        for i, seq in enumerate(self.trn):
            l = len(seq)
            if norm: seq = 2*((seq-np.repeat(f_min, l, axis=0))/np.repeat(scale, l, axis=0))-1
            self.trn_len.append(l)
            self.trn[i] = np.pad(seq, ((0, 2000-l), (0, 0)), 'constant', constant_values=0).reshape(1, 2000, -1)
        for i, seq in enumerate(self.val):
            l = len(seq)
            if norm: seq = 2*((seq-np.repeat(f_min, l, axis=0))/np.repeat(scale, l, axis=0))-1
            self.val_len.append(l)
            self.val[i] = np.pad(seq, ((0, 2000-l), (0, 0)), 'constant', constant_values=0).reshape(1, 2000, -1)
        self.trn, self.val = np.concatenate(self.trn, axis=0), np.concatenate(self.val, axis=0)

    def __getitem__(self, index):
        if self.train:
            features, src_len = self.trn[index], np.array(self.trn_len)[index]
        else:
            features, src_len = self.trn[index], np.array(self.val_len)[index]
        return features, src_len

    def __len__(self):
        return len(self.trn)
    
    def visualize(self, s_id, f_id):
        plt.plot(self.trn[s_id, :, f_id])
        plt.show()
        plt.close()


if __name__ == '__main__':
    mat_to_npy()