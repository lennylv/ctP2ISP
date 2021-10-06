import os
import pickle
import numpy as np
import torch
import random
from torch import nn
from torch.optim import lr_scheduler
import copy
from torch import Tensor


class MyModule(nn.Module):
    def __init__(self,hidden_size,window=30,dropout=0.5):
        super(MyModule, self).__init__()
        self.loc_seq_embed = nn.Embedding(21,hidden_size,padding_idx=0)
        self.loc_seq_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(),
                                        nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_seq_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(),
                                        nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_seq_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.loc_seq_transformer = nn.TransformerEncoder(self.loc_seq_encoder,num_layers=1)

        self.loc_ss_embed = nn.Embedding(4,hidden_size,padding_idx=0)
        self.loc_ss_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(),
                                        nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_ss_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(),
                                        nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_ss_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.loc_ss_transformer = nn.TransformerEncoder(self.loc_ss_encoder,num_layers=1)

        self.loc_pssm_embed = nn.Sequential(nn.Conv1d(20,hidden_size,kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.loc_pssm_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_pssm_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_pssm_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.loc_pssm_transformer = nn.TransformerEncoder(self.loc_pssm_encoder,num_layers=1)

        self.loc_asa_embed = nn.Sequential(nn.Conv1d(1,hidden_size,kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.loc_asa_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_asa_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_asa_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.loc_asa_transformer = nn.TransformerEncoder(self.loc_asa_encoder,num_layers=1)

        self.loc_angle_embed = nn.Sequential(nn.Conv1d(4,hidden_size,kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.loc_angle_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_angle_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_angle_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.loc_angle_transformer = nn.TransformerEncoder(self.loc_angle_encoder,num_layers=1)

        self.loc_res_embed = nn.Sequential(nn.Conv1d(11,hidden_size,kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.loc_res_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_res_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_res_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.loc_res_transformer = nn.TransformerEncoder(self.loc_res_encoder,num_layers=1)

        self.loc_mix_conv1 = nn.Sequential(nn.Conv1d(hidden_size*6,hidden_size*4,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size*4),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_mix_conv2 = nn.Sequential(nn.Conv1d(hidden_size*4,hidden_size*4,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size*4),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.loc_mix_encoder = nn.TransformerEncoderLayer(d_model=hidden_size*4,nhead=4,dim_feedforward=hidden_size*16,dropout=dropout)
        self.loc_mix_transformer = nn.TransformerEncoder(self.loc_mix_encoder,num_layers=1)


        self.glo_seq_window = nn.Sequential(nn.Conv1d(500,window,kernel_size=1),
                                            nn.ReLU())
        self.glo_seq_embed = nn.Embedding(21,hidden_size,padding_idx=0)
        self.glo_seq_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_seq_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_seq_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.glo_seq_transformer = nn.TransformerEncoder(self.glo_seq_encoder,num_layers=1)

        self.glo_ss_window = nn.Sequential(nn.Conv1d(500,window,kernel_size=1),
                                            nn.ReLU())
        self.glo_ss_embed = nn.Embedding(4,hidden_size,padding_idx=0)
        self.glo_ss_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(),
                                        nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_ss_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(),
                                        nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_ss_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.glo_ss_transformer = nn.TransformerEncoder(self.glo_ss_encoder,num_layers=1)

        self.glo_pssm_window = nn.Sequential(nn.Conv1d(500,window,kernel_size=1),
                                            nn.ReLU())
        self.glo_pssm_embed = nn.Sequential(nn.Conv1d(20,hidden_size,kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.glo_pssm_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_pssm_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_pssm_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.glo_pssm_transformer = nn.TransformerEncoder(self.glo_pssm_encoder,num_layers=1)

        self.glo_asa_window = nn.Sequential(nn.Conv1d(500,window,kernel_size=1),
                                            nn.ReLU())
        self.glo_asa_embed = nn.Sequential(nn.Conv1d(1,hidden_size,kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.glo_asa_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_asa_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_asa_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.glo_asa_transformer = nn.TransformerEncoder(self.glo_asa_encoder,num_layers=1)

        self.glo_angle_window = nn.Sequential(nn.Conv1d(500,window,kernel_size=1),
                                            nn.ReLU())
        self.glo_angle_embed = nn.Sequential(nn.Conv1d(4,hidden_size,kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.glo_angle_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_angle_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_angle_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.glo_angle_transformer = nn.TransformerEncoder(self.glo_angle_encoder,num_layers=1)

        self.glo_res_window = nn.Sequential(nn.Conv1d(500,window,kernel_size=1),
                                            nn.ReLU())
        self.glo_res_embed = nn.Sequential(nn.Conv1d(11,hidden_size,kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.glo_res_conv1 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_res_conv2 = nn.Sequential(nn.Conv1d(hidden_size,hidden_size,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_res_encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=1,dim_feedforward=hidden_size*4,dropout=dropout)
        self.glo_res_transformer = nn.TransformerEncoder(self.glo_res_encoder,num_layers=1)

        self.glo_mix_conv1 = nn.Sequential(nn.Conv1d(hidden_size*6,hidden_size*4,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size*4),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_mix_conv2 = nn.Sequential(nn.Conv1d(hidden_size*4,hidden_size*4,kernel_size=7,padding=3),
                                            nn.BatchNorm1d(hidden_size*4),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3,stride=1,padding=1))
        self.glo_mix_encoder = nn.TransformerEncoderLayer(d_model=hidden_size*4,nhead=4,dim_feedforward=hidden_size*16,dropout=dropout)
        self.glo_mix_transformer = nn.TransformerEncoder(self.glo_mix_encoder,num_layers=1)

        
        self.conv1 = nn.Sequential(nn.Conv1d(hidden_size*10,hidden_size*10,kernel_size=11,padding=5),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding=5))
        self.conv2 = nn.Sequential(nn.Conv1d(hidden_size*10,hidden_size*10,kernel_size=13,padding=6),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=13,stride=1,padding=6))
        self.conv3 = nn.Sequential(nn.Conv1d(hidden_size*10,hidden_size*10,kernel_size=15,padding=7),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=15,stride=1,padding=7))

        self.l1 = nn.Sequential(nn.Linear(hidden_size*40,hidden_size),
                                nn.ReLU())
        self.l2 = nn.Linear(hidden_size,1)

    def forward(self, x, lengths):
        loc_seq, loc_ss, loc_pssm, loc_asa, loc_angle,loc_res = x[:,:window,0].long(), x[:,:window,1].long(), x[:,:window,2:22], x[:,:window,22], x[:,:window,23:27], x[:,:window,27:]
        # seq
        loc_seq = self.loc_seq_embed(loc_seq)
        loc_seq_fea = loc_seq.permute(0,2,1)
        loc_seq_fea = self.loc_seq_conv1(loc_seq_fea)
        loc_seq_fea = self.loc_seq_conv2(loc_seq_fea)
        loc_seq_fea = loc_seq_fea.permute(0,2,1).permute(1,0,2)
        loc_seq_fea = self.loc_seq_transformer(loc_seq_fea)
        # ss
        loc_ss = self.loc_ss_embed(loc_ss)
        loc_ss_fea = loc_ss.permute(0,2,1)
        loc_ss_fea = self.loc_ss_conv1(loc_ss_fea)
        loc_ss_fea = self.loc_ss_conv2(loc_ss_fea)
        loc_ss_fea = loc_ss_fea.permute(0,2,1).permute(1,0,2)
        loc_ss_fea = self.loc_ss_transformer(loc_ss_fea)
        # pssm
        loc_pssm = loc_pssm.permute(0,2,1)
        loc_pssm_fea = self.loc_pssm_embed(loc_pssm)
        loc_pssm = loc_pssm_fea.permute(0,2,1)
        loc_pssm_fea = self.loc_pssm_conv1(loc_pssm_fea)
        loc_pssm_fea = self.loc_pssm_conv2(loc_pssm_fea)
        loc_pssm_fea = loc_pssm_fea.permute(0,2,1).permute(1,0,2)
        loc_pssm_fea = self.loc_pssm_transformer(loc_pssm_fea)
        # asa
        loc_asa = loc_asa.unsqueeze(2).permute(0,2,1)
        loc_asa_fea = self.loc_asa_embed(loc_asa)
        loc_asa = loc_asa_fea.permute(0,2,1)
        loc_asa_fea = self.loc_asa_conv1(loc_asa_fea)
        loc_asa_fea = self.loc_asa_conv2(loc_asa_fea)
        loc_asa_fea = loc_asa_fea.permute(0,2,1).permute(1,0,2)
        loc_asa_fea = self.loc_asa_transformer(loc_asa_fea)
        # angle
        loc_angle = loc_angle.permute(0,2,1)
        loc_angle_fea = self.loc_angle_embed(loc_angle)
        loc_angle = loc_angle_fea.permute(0,2,1)
        loc_angle_fea = self.loc_angle_conv1(loc_angle_fea)
        loc_angle_fea = self.loc_angle_conv2(loc_angle_fea)
        loc_angle_fea = loc_angle_fea.permute(0,2,1).permute(1,0,2)
        loc_angle_fea = self.loc_angle_transformer(loc_angle_fea)
        # res
        loc_res = loc_res.permute(0,2,1)
        loc_res_fea = self.loc_res_embed(loc_res)
        loc_res = loc_res_fea.permute(0,2,1)
        loc_res_fea = self.loc_res_conv1(loc_res_fea)
        loc_res_fea = self.loc_res_conv2(loc_res_fea)
        loc_res_fea = loc_res_fea.permute(0,2,1).permute(1,0,2)
        loc_res_fea = self.loc_res_transformer(loc_res_fea)
        # mix
        loc_mix = torch.cat((loc_seq, loc_ss, loc_pssm, loc_asa, loc_angle, loc_res),-1)
        loc_mix_fea = loc_mix.permute(0,2,1)
        loc_mix_fea = self.loc_mix_conv1(loc_mix_fea)
        loc_mix_fea = self.loc_mix_conv2(loc_mix_fea)
        loc_mix_fea = loc_mix_fea.permute(0,2,1).permute(1,0,2)
        loc_mix_fea = self.loc_mix_transformer(loc_mix_fea)

        glo_seq, glo_ss, glo_pssm, glo_asa, glo_angle, glo_res = x[:,window:,0].long(), x[:,window:,1].long(), x[:,window:,2:22], x[:,window:,22], x[:,window:,23:27], x[:,window:,27:]
        # seq
        glo_seq = self.glo_seq_embed(glo_seq)
        glo_seq = self.glo_seq_window(glo_seq)
        glo_seq_fea = glo_seq.permute(0,2,1)
        glo_seq_fea = self.glo_seq_conv1(glo_seq_fea)
        glo_seq_fea = self.glo_seq_conv2(glo_seq_fea)
        glo_seq_fea = glo_seq_fea.permute(0,2,1).permute(1,0,2)
        glo_seq_fea = self.glo_seq_transformer(glo_seq_fea)
        # ss
        glo_ss = self.glo_ss_embed(glo_ss)
        glo_ss = self.glo_ss_window(glo_ss)
        glo_ss_fea = glo_ss.permute(0,2,1)
        glo_ss_fea = self.glo_ss_conv1(glo_ss_fea)
        glo_ss_fea = self.glo_ss_conv2(glo_ss_fea)
        glo_ss_fea = glo_ss_fea.permute(0,2,1).permute(1,0,2)
        glo_ss_fea = self.glo_ss_transformer(glo_ss_fea)
        # pssm
        glo_pssm = self.glo_pssm_window(glo_pssm).permute(0,2,1)
        glo_pssm_fea = self.glo_pssm_embed(glo_pssm)
        glo_pssm = glo_pssm_fea.permute(0,2,1)
        glo_pssm_fea = self.glo_pssm_conv1(glo_pssm_fea)
        glo_pssm_fea = self.glo_pssm_conv2(glo_pssm_fea)
        glo_pssm_fea = glo_pssm_fea.permute(0,2,1).permute(1,0,2)
        glo_pssm_fea = self.glo_pssm_transformer(glo_pssm_fea)
        # asa
        glo_asa = glo_asa.unsqueeze(2)
        glo_asa = self.glo_asa_window(glo_asa).permute(0,2,1)
        glo_asa_fea = self.glo_asa_embed(glo_asa)
        glo_asa = glo_asa_fea.permute(0,2,1)
        glo_asa_fea = self.glo_asa_conv1(glo_asa_fea)
        glo_asa_fea = self.glo_asa_conv2(glo_asa_fea)
        glo_asa_fea = glo_asa_fea.permute(0,2,1).permute(1,0,2)
        glo_asa_fea = self.glo_asa_transformer(glo_asa_fea)
        # angle
        glo_angle = self.glo_angle_window(glo_angle).permute(0,2,1)
        glo_angle_fea = self.glo_angle_embed(glo_angle)
        glo_angle = glo_angle_fea.permute(0,2,1)
        glo_angle_fea = self.glo_angle_conv1(glo_angle_fea)
        glo_angle_fea = self.glo_angle_conv2(glo_angle_fea)
        glo_angle_fea = glo_angle_fea.permute(0,2,1).permute(1,0,2)
        glo_angle_fea = self.glo_angle_transformer(glo_angle_fea)
        # res
        glo_res = self.glo_res_window(glo_res).permute(0,2,1)
        glo_res_fea = self.glo_res_embed(glo_res)
        glo_res = glo_res_fea.permute(0,2,1)
        glo_res_fea = self.glo_res_conv1(glo_res_fea)
        glo_res_fea = self.glo_res_conv2(glo_res_fea)
        glo_res_fea = glo_res_fea.permute(0,2,1).permute(1,0,2)
        glo_res_fea = self.glo_res_transformer(glo_res_fea)
        # mix
        glo_mix = torch.cat((glo_seq, glo_ss, glo_pssm, glo_asa, glo_angle, glo_res),-1)
        glo_mix_fea = glo_mix.permute(0,2,1)
        glo_mix_fea = self.glo_mix_conv1(glo_mix_fea)
        glo_mix_fea = self.glo_mix_conv2(glo_mix_fea)
        glo_mix_fea = glo_mix_fea.contiguous().permute(0,2,1).permute(1,0,2)
        glo_mix_fea = self.glo_mix_transformer(glo_mix_fea)


        loc_fea = torch.cat((loc_mix_fea, loc_seq_fea, loc_ss_fea, loc_pssm_fea, loc_asa_fea, loc_angle_fea, loc_res_fea),-1).permute(1,0,2)
        glo_fea = torch.cat((glo_mix_fea, glo_seq_fea, glo_ss_fea, glo_pssm_fea, glo_asa_fea, glo_angle_fea, glo_res_fea),-1).permute(1,0,2)
        glo_fea = glo_fea.permute(0,2,1)
        glo_fea1 = self.conv1(glo_fea)
        glo_fea2 = self.conv2(glo_fea)
        glo_fea3 = self.conv3(glo_fea)
        glo_fea1, glo_fea2, glo_fea3 = glo_fea1.permute(0,2,1), glo_fea2.permute(0,2,1), glo_fea3.permute(0,2,1)

        fea = torch.cat((loc_fea, glo_fea1, glo_fea2, glo_fea3),-1)
        fea = self.l1(fea)
        res = self.l2(fea)

        return res

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
        return TP,TN,FP,FN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout = 0.5
hidden_size = 32
window = 30
eval_num = EvalCriterion().to(device)
model = MyModule(hidden_size,window,dropout)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model,device_ids=[0,1])
model = model.to(device)
pos_weight = torch.tensor(0.8)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device) 
optimizer = torch.optim.Adam(model.parameters())
scheduler = lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)


def evaluate(model, data, lens_index):
    model.eval()
    valid_loss = total_num_words = total_loss = 0.
    a_max, p_max, r_max, f_max, m_max, t_max = 0, 0, 0, 0, 0, 0
    tp_max, tn_max, fp_max, fn_max = 0, 0, 0, 0
    for i in range(1,51):
        t = i/100.0
        with torch.no_grad():
            TP,TN,FP,FN = 0,0,0,0
            for it, ((mb_x, mb_x_len, mb_y, mb_y_len),len_index) in enumerate(zip(data,lens_index)):
                mb_x = mb_x.to(device).float()
                mb_x_len = mb_x_len.to(device).long()
                mb_y = mb_y.to(device).long()
                mb_y_len = mb_y_len.to(device).long()
                mb_pred = model(mb_x, mb_x_len)
                
                a,b,c,d = eval_num(mb_pred,mb_y,len_index,t)
                TP,TN,FP,FN = TP+a,TN+b,FP+c,FN+d
                
                mb_pred = mb_pred.contiguous().view(-1)
                mb_y = mb_y.contiguous().view(-1).float()
                loss = loss_fn(mb_pred, mb_y)
                num_words = torch.sum(mb_y_len).item()
                total_loss += loss.item() * num_words
                total_num_words += num_words

        acc = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f_m = (2*precision*recall)/(precision+recall)
        MCC = (TP*TN-FP*FN)/torch.sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)).float())
        if f_max<f_m:
            a_max, p_max, r_max, f_max, m_max, t_max = acc, precision, recall, f_m, MCC, t
            tp_max, tn_max, fp_max, fn_max = TP,TN,FP,FN
            valid_loss = total_loss/total_num_words
        
    print(f'TP={tp_max},TN={tn_max},FP={fp_max},FN={fn_max}')
    print("ACC="+"%.4f"%a_max+", Precision="+"%.4f"%p_max+", Recall="+"%.4f"%r_max+", F_mea="+"%.4f"%f_max+", MCC="+"%.4f"%m_max+", threshold="+"%.4f"%t_max)
    print("Validation loss:", total_loss/total_num_words)
    return valid_loss,a_max,p_max,r_max,f_max,m_max,t_max


def train(model, data, epoch, set_num, best_F, t_loss, t_num):
    threshold = 0 
    model.train()
    total_num_words = total_loss = 0.
    for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
        mb_x = mb_x.to(device).float()
        mb_x_len = mb_x_len.to(device).long()
        mb_y = mb_y.to(device).long()
        mb_y_len = mb_y_len.to(device).long()
        mb_pred = model(mb_x, mb_x_len)

        mb_pred = mb_pred.contiguous().view(-1)
        mb_y = mb_y.contiguous().view(-1).float()
        loss = loss_fn(mb_pred, mb_y)
            
        num_words = torch.sum(mb_y_len).item()
        total_loss += loss.item() * num_words
        total_num_words += num_words
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        
    t_loss += total_loss
    t_num += total_num_words
    print("Train set%d have been trained!" %set_num)
    if set_num == 44:
        print("\nEpoch", epoch+1, "Training loss", t_loss/t_num)
        v_loss,v_acc,v_pre,v_rec,v_score,v_mcc,v_t= evaluate(model, dev_data, res_index_dev)
        if v_score > best_F:
            best_F = v_score
            threshold = v_t
            print("new best F_value:{0}(threshold:{1})".format(best_F,threshold),"\n")
            torch.save(model.state_dict(),os.path.join("../model",'model.dat'))
        scheduler.step()

    return best_F, total_loss, total_num_words


if __name__ == '__main__':
    print("Validation set start load.")
    with open('../data/window/dev_data.pkl','rb') as f:
        dev_data = pickle.load(f)
    with open('../data/window/res_index_dev.pkl','rb') as f:
        res_index_dev = pickle.load(f)
    print("Validation set is loaded!")

    epochs = 10
    best_F = 0
    t_loss, t_num = 0., 0.
    for i in range(epochs):
        for j in range(45):
            print("\nEpoch%d Train set%d start load."%(i+1,j))
            with open('../data/window/train_data%d.pkl' %j,'rb') as f:
                train_data = pickle.load(f)
                print("Train set%d is loaded!"%j)
                best_F, t_loss, t_num = train(model, train_data, i, j, best_F, t_loss, t_num)

