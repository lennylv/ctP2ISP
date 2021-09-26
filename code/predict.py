import os
import pickle
import numpy as np
import torch
import random
from torch import nn
import copy
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch import Tensor
from train import MyModule


with open('../data/set448_lab.pkl','rb') as f:
    label = pickle.load(f)
with open('../data/set448_fea.pkl','rb') as f:
    input_data = pickle.load(f)


def pre_prepare_data(data,window=30,to_label=False):
    res = []
    lens = [len(s) for s in data]
    lens_index = []
    for k in range(len(data)):
        x = torch.from_numpy(np.array(data[k]))
        assert lens[k] - window >= 0 ,'The window size is too big!'
        x = torch.cat([copy.deepcopy(x[None,i:i+window]) for i in range(lens[k]-window+1)], 0)
        if to_label==True:
            y = np.zeros((lens[k]-window+1,500,38))
            for i in range(lens[k]-window+1):
                for j in range(lens[k]):
                    if j<500:
                        y[i][j] = data[k][j]
                    else:
                        break
            y = torch.from_numpy(y)
            x = torch.cat((x,y),1)
        lens_index.append(lens[k]-window+1)
        res.append(x)
    res = torch.cat(res,0)
    return res,lens_index   


def get_minibatches(n, minibatch_size, lens_index):
    minibatches = []
    indeies_of_s, index_of_s= [], []
    tmp = 0
    for i,len_index in enumerate(lens_index) :
        if tmp + len_index <= minibatch_size:
            index_of_s.append(i)
            tmp = tmp + len_index
        else:
            assert tmp != 0,'The batch size is too small!'
            indeies_of_s.append(index_of_s)
            index_of_s = [i]
            tmp = len_index
    indeies_of_s.append(index_of_s)
    sum_i = 0
    res_index = []
    for i in indeies_of_s:
        batch,res_i,sum_ii = [], [], 0
        for j in i:
            batch.extend([k for k in range(sum_i,sum_i+lens_index[j])])
            res_i.append([k for k in range(sum_ii,sum_ii+lens_index[j])])
            sum_i = sum_i+lens_index[j]
            sum_ii = sum_ii+lens_index[j]
        minibatches.append(batch)
        res_index.append(res_i)
    return minibatches,res_index


def prepare_data(seqs):
    x = torch.cat([s.unsqueeze(0) for s in seqs])
    x_lengths = torch.ones(len(seqs)) * seqs[0].shape[0]
    return x, x_lengths.long()


def gen_examples(en_sentences, cn_sentences, batch_size, lens_index):
    minibatches,res_index = get_minibatches(len(en_sentences), batch_size, lens_index)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_en_sentences)
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return all_ex,res_index


class EvalCriterion(nn.Module):
    def __init__(self):
        super(EvalCriterion, self).__init__()

    def getRawData(self,input,lens_index,t,to_label=False):
        raw_input = []
        for len_index in lens_index:
            r_sub,r_len = len(len_index), len(len_index) + window - 1
            tmp = torch.zeros(r_sub,r_len)
            for l in range(tmp.shape[0]):
                tmp[l][l:l+window] = input[len_index[l]]
            num = (tmp != 0).float()
            num = num.sum(0)
            num[num<=0] = 1
            tmp = tmp.sum(0) * (1/num) #avg
            if to_label:
                tmp = torch.sigmoid(tmp)
                tmp = (tmp>t).float()
            raw_input.append(tmp.long())
        return raw_input

    def forward(self, input:Tensor, target:Tensor, lens_index:Tensor, t):
        raw_input = self.getRawData(input.squeeze(),lens_index,t,True)
        raw_target = self.getRawData(target,lens_index,t)

        TP,TN,FP,FN = 0.,0.,0.,0.
        for inp,tar in zip(raw_input,raw_target):
            TP += (inp&tar).sum()
            TN += ((inp==0)&(tar==0)).sum()
            FP += (inp&(tar==0).long()).sum()
            FN += ((inp==0).long()&tar).sum()
        return TP,TN,FP,FN,raw_target,raw_input


def predict(model, data, lens_index):
    model.eval()
    a_max, p_max, r_max, f_max, m_max, t_max = 0, 0, 0, 0, 0, 0
    tp_max, tn_max, fp_max, fn_max = 0, 0, 0, 0
    au_roc_max, au_prc_max = 0, 0
    for i in range(1,51):
        t = i/100.0
        with torch.no_grad():
            TP,TN,FP,FN = 0,0,0,0
            y_true, y_pred = [], []
            for it, ((mb_x, mb_x_len, mb_y, mb_y_len),len_index) in enumerate(zip(data,lens_index)):
                mb_x = mb_x.to(device).float()
                mb_x_len = mb_x_len.to(device).long()
                mb_y = mb_y.to(device).long()
                mb_y_len = mb_y_len.to(device).long()
                mb_pred = model(mb_x, mb_x_len)
                
                a,b,c,d,y_true_enu,y_pred_enu = eval_num(mb_pred,mb_y,len_index,t)
                TP,TN,FP,FN = TP+a,TN+b,FP+c,FN+d
                for i in range(len(y_true_enu)):
                    y_true_enu[i] = y_true_enu[i].numpy()
                    y_true.extend(y_true_enu[i])
                for i in range(len(y_pred_enu)):
                    y_pred_enu[i] = y_pred_enu[i].numpy()
                    y_pred.extend(y_pred_enu[i])
                
                mb_pred = mb_pred.contiguous().view(-1)
                mb_y = mb_y.contiguous().view(-1).float()

        acc = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f_m = (2*precision*recall)/(precision+recall)
        MCC = (TP*TN-FP*FN)/torch.sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)).float())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        au_roc = auc(fpr, tpr)
        pre, rec, thresholds = precision_recall_curve(y_true,y_pred)
        au_prc = auc(rec, pre)

        if f_max<f_m:
            a_max, p_max, r_max, f_max, m_max, t_max = acc, precision, recall, f_m, MCC, t
            tp_max, tn_max, fp_max, fn_max = TP,TN,FP,FN
            au_roc_max, au_prc_max = au_roc, au_prc
        
    print(f'TP={tp_max},TN={tn_max},FP={fp_max},FN={fn_max}')
    print("ACC="+"%.3f"%a_max+", Precision="+"%.3f"%p_max+", Recall="+"%.3f"%r_max+", F_mea="+"%.3f"%f_max+", MCC="+"%.3f"%m_max+", threshold="+"%.3f"%t_max)
    print("AUROC="+"%.4f"%au_roc_max+", AUPRC="+"%.4f"%au_prc_max+'\n')


if __name__ == '__main__':
    window = 30
    input_data,lens_index = pre_prepare_data(input_data, window,True)
    label,_ = pre_prepare_data(label,window,False)

    batch_size = 4000
    test_data,res_index = gen_examples(input_data, label, batch_size,lens_index)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_num = EvalCriterion().to(device)

    dropout = 0.5
    hidden_size = 32
    model = MyModule(hidden_size,window,dropout)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids=[0,1])
    model = model.to(device)
    model_file = "../model/model.dat"
    model.load_state_dict(torch.load(model_file),False)
    predict(model,test_data,res_index)
