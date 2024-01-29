from sklearn import linear_model
import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import copy
from tqdm import tqdm

from collections import Counter

from collections import OrderedDict
import pandas as pd
from collections import Counter
from sklearn.manifold import TSNE

from torch import Tensor
from sklearn import preprocessing
from torch.nn.utils.rnn import pad_sequence
import math

import pickle
import copy
import argparse

def get_auc_ap(true_labels, predictions):
    fpr, tpr, thre = metrics.roc_curve(true_labels, predictions)
    auc = metrics.auc(fpr, tpr)
    ap = metrics.average_precision_score(true_labels, predictions)
    return auc, ap, fpr, tpr

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(RoPEMultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.output_proj = nn.Linear(self.d_head * num_heads, d_model, bias=True)
	    self.layernorm = nn.LayerNorm(d_model)
    
    def RoPE(self, x, rope_per): ### input x shape is [seq_len, batch_size, n_heads, d]
        d = int(rope_per * self.d_head)
        seq_len = x.shape[0]
        theta = 1. / (10000 ** (torch.arange(0, d, 2).float() / d)).to(x.device)
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        
        cos_cached = idx_theta2.cos()[:, None, None, :]
        sin_cached = idx_theta2.sin()[:, None, None, :]

        x_rope, x_pass = x[..., :d], x[..., d:]
        d_2 = d // 2
        neg_half_x = torch.cat([-x_rope[:, :, :, d_2:], x_rope[:, :, :, :d_2]], dim=-1)
        
        x_rope = (x_rope * cos_cached[:x.shape[0]]) + (neg_half_x * sin_cached[:x.shape[0]])
        return torch.cat((x_rope, x_pass), dim=-1)
        
    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True) -> Tensor:
        batch_size = value.size(0)
        
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)
        
        query = self.RoPE(query.permute(1, 0, 2, 3).contiguous(), 0.5)
        key = self.RoPE(key.permute(1, 0, 2, 3).contiguous(), 0.5) ### return shape is [seq_len, batch_size, n_heads, d]
        
        query = query.permute(1, 2, 0, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(1, 2, 0, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        
        context = F.scaled_dot_product_attention(query,key,value, attn_mask, dropout_p, is_causal)
        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)
	    context = self.layernorm(context + x)
        
        context = self.output_proj(context) + context
        return context

class RoPEAttenEncoder(nn.Module):
    
    def __init__(self, feat_dim, layers_num, d_model: int = 1024, num_heads: int = 8):
        super(RoPEAttenEncoder, self).__init__()
        self.feat_emb = nn.Linear(feat_dim, d_model)
        self.layers_num = layers_num
        self.attn = RoPEMultiHeadAttention(d_model, num_heads)
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, x, attn_mask=None, dropout_p=0.0, is_causal=True) -> Tensor: 
        x = self.feat_emb(x)
        for i in range(self.layers_num):
            x = self.attn(x, x, x, x, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            x = self.layernorm(x)
        return x

class PatLevelDataset(Dataset):
    def __init__(self, raw_data1, raw_data2, raw_label, raw_usrds_id, raw_time_stamps):
        super(PatLevelDataset, self).__init__()
        ### month-level data
        self.raw_data1 = raw_data1
        self.raw_data2 = raw_data2
        self.raw_label = raw_label
        self.pat_ids = raw_usrds_id
        self.raw_time_stamps = raw_time_stamps
        
    def __len__(self):
        ### return length of patients
        return len(self.pat_ids)
    
    def __getitem__(self, idx):
        ### based on usrds id change month-level data into patient-level data
        pat_feats1 = torch.cat([item.reshape(1, -1) for item in self.raw_data1[idx]])
        pat_feats2 = torch.cat([item.reshape(1, -1) for item in self.raw_data2[idx]])
        pat_labels = self.raw_label[idx]
        pat_time_stamps = self.raw_time_stamps[idx]
        return self.pat_ids[idx], pat_feats1,pat_feats2, torch.tensor(pat_labels), torch.tensor(pat_time_stamps)
    
def PatLevel_collate_fn(batch_data):
    batch_pat_usrds_id, batch_pat_feats1, batch_pat_feats2, batch_pat_labels, batch_pat_timestamps = zip(*batch_data)
    
    batch_pat_lens = [len(pat_feat) for pat_feat in batch_pat_feats1]
    batch_pad_pat_feats1 = pad_sequence(batch_pat_feats1, batch_first=True, padding_value=-100) ### normalized features range is 0-1
    batch_pad_pat_feats2 = pad_sequence(batch_pat_feats2, batch_first=True, padding_value=-100) ### normalized features range is 0-1
    batch_pad_pat_labels = pad_sequence(batch_pat_labels, batch_first=True, padding_value=-100) ### labels are all in [0, 1]
    batch_pad_pat_mask = (batch_pad_pat_labels!=-100)
   
    return batch_pat_usrds_id, batch_pat_lens, batch_pat_timestamps, batch_pad_pat_mask, batch_pad_pat_feats1, batch_pad_pat_feats2, batch_pad_pat_labels

class AttentionCPCCL:
    
    def __init__(self,
                 feat1_dim, feat2_dim, 
                 device='cuda:0',
                 CL_temp: float=0.5,
                 attn1_d_model: int=1024, attn1_num_heads: int=8, attn1_layers: int=1,
                 attn2_d_model: int=1024, attn2_num_heads: int=8, attn2_layers: int=2,
                 cpc_steps: int=1, cpc_k: int=12,
		 window_size: int=2,
                ):
        
        ### set encoders
        self.encoder1 = RoPEAttenEncoder(feat1_dim, attn1_layers, attn1_d_model, attn1_num_heads)
        self.encoder2 = RoPEAttenEncoder(feat2_dim, attn2_layers, attn2_d_model, attn2_num_heads)
        
        self.cpc_steps = cpc_steps
	    self.cpc_k = cpc_k
        self.cpcout_proj1 = nn.ModuleList(
            [nn.Linear(in_features=attn1_d_model, out_features=attn2_d_model, bias=True) for i in range(self.cpc_steps)]
        )
        self.cpcout_proj2 = nn.ModuleList(
            [nn.Linear(in_features=attn2_d_model, out_features=attn1_d_model, bias=True) for i in range(self.cpc_steps)]
        )
        self.lsoftmax = nn.LogSoftmax(dim=1)
    
        ### hyper-parameter
        self.CL_temp = CL_temp
        self.device = device
        self.window_size = window_size
    
    def CL_posloss(self, batch_embs1, batch_embs2, batch_pad_pat_mask, 
                   batch_pat_timestamps, batch_pat_usrds_id, batch_pat_lens, survival_prob):
        
        ### cat patients all embeddings and remove the pad token
        feat1 = batch_embs1[batch_pad_pat_mask]
        feat2 = batch_embs2[batch_pad_pat_mask]
        
        features = F.normalize(torch.cat([feat1, feat2]), dim=1) 
        similarity_matrix = torch.matmul(features, features.T)
        
        ### generate positive pair mask
        #### time close
        time_mask = torch.cat([batch_pat_timestamps, batch_pat_timestamps])
        time_mask = ((time_mask.view(1, -1) - time_mask.view(-1, 1)).abs() < self.window_size)
        #### same patients
        batch_pat = torch.repeat_interleave(torch.tensor(batch_pat_usrds_id), torch.tensor(batch_pat_lens))
        pat_mask = torch.cat([batch_pat, batch_pat])
        pat_mask = (pat_mask.view(1, -1) == pat_mask.view(-1,1))
        #### final mask
        final_sim_mask = time_mask & pat_mask
        
        ### generate negative mask
        sim_mask_neg_truncated = final_sim_mask.clone()
        final_sim_mask = final_sim_mask.to(self.device, non_blocking=True)
        row_pos_samples = final_sim_mask.sum(dim=1).tolist()
        target_pos_samples = max(row_pos_samples)
        for i in range(sim_mask_neg_truncated.size(0)):
            cur_pos_samples = row_pos_samples[i]
            while cur_pos_samples < target_pos_samples:
                j = random.choice(range(sim_mask_neg_truncated.size(1)))
                if not sim_mask_neg_truncated[i, j]:
                    sim_mask_neg_truncated[i, j] = True
                    cur_pos_samples += 1
        sim_mask_neg_truncated = sim_mask_neg_truncated.to(self.device, non_blocking=True)
    
        ### generate logits
        pos_indices = torch.where(final_sim_mask & ~torch.eye(final_sim_mask.size(0), dtype=torch.bool, device=self.device))
    
        pos_logits = similarity_matrix[pos_indices[0], pos_indices[1]].view(-1, 1)
        neg_logits = similarity_matrix[~sim_mask_neg_truncated].view(
            sim_mask_neg_truncated.size(0), sim_mask_neg_truncated.size(1) - target_pos_samples
        )
        neg_logits = neg_logits[pos_indices[0]]
        
        assert survival_prob.shape == similarity_matrix.shape
        logits = torch.cat([pos_logits * survival_prob[pos_indices[0], pos_indices[1]].view(-1,1), neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)    
        return logits / self.CL_temp, labels
    
    def CPCloss(self, batch_embs1, batch_embs2, batch_pad_pat_mask): ### the embs input shape is batch_size * seq_len * emb_dim
        nce = 0
        
        t_samples = torch.empty((len(batch_embs1))).long() # for each pat, we have different length cpc
        for i, mask in enumerate(batch_pad_pat_mask):
            if torch.sum(mask) < self.cpc_steps + self.cpc_k:
                t_samples[i] = -1
            elif torch.sum(mask) == self.cpc_steps + self.cpc_k:
                t_samples[i] = 0
            else:
                t_samples[i] = torch.randint(torch.sum(mask) - self.cpc_steps - self.cpc_k, size=(1,)).long()
        t_samples_mask = t_samples >= 0
        
        ### the cpc for embs1
        encode_samples1 = torch.empty((sum(t_samples_mask), self.cpc_steps, batch_embs1.shape[-1])).float() ### size is batch_size * cpc_timestep * emb_dim
        for i in range(1, self.cpc_steps+1):
            for k, t_sample in enumerate(t_samples[t_samples_mask]):
                encode_samples1[k, i-1, :] = batch_embs1[t_samples_mask][k, t_sample + i + (self.cpc_k - 1), :]
        
        c_t1 = torch.empty((sum(t_samples_mask), batch_embs1.shape[-1])).float()
        for k, t_sample in enumerate(t_samples[t_samples_mask]):
            c_t1[k] = batch_embs1[t_samples_mask][k, t_sample, :] ### size is batch_size * 1 * emb_dim
            
        pred1 = torch.empty((sum(t_samples_mask), self.cpc_steps, batch_embs1.shape[-1])).float() ### size is batch_size * cpc_timestep  * emb_dim
        for i in range(0, self.cpc_steps):
            pred1[:, i, :] = self.cpcout_proj1[i](c_t1)
        
        for i in range(0, self.cpc_steps):
            total = torch.mm(encode_samples1[:, i, :], torch.transpose(pred1[:, i, :],0,1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
            
        ### the cpc for embs2
        encode_samples2 = torch.empty((sum(t_samples_mask), self.cpc_steps, batch_embs2.shape[-1])).float()
        for i in range(1, self.cpc_steps+1):
            for k, t_sample in enumerate(t_samples[t_samples_mask]):
                encode_samples2[k, i-1, :] = batch_embs2[t_samples_mask][k, t_samplei + i + (self.cpc_k - 1), :]
        
        c_t2 = torch.empty((sum(t_samples_mask), batch_embs2.shape[-1])).float()
        for k, t_sample in enumerate(t_samples[t_samples_mask]):
            c_t2[k] = batch_embs2[t_samples_mask][k, t_sample, :] 
            
        pred2 = torch.empty((sum(t_samples_mask), self.cpc_steps, batch_embs2.shape[-1])).float()
        for i in range(0, self.cpc_steps):
            pred2[:, i, :] = self.cpcout_proj2[i](c_t2)
        
        for i in range(0, self.cpc_steps):
            total = torch.mm(encode_samples2[:, i, :], torch.transpose(pred2[:, i, :],0,1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
            
        nce /= -1.*sum(t_samples_mask)*self.cpc_steps
        nce /= 2.
        return nce
        
    
    def CLtrain(self, 
                train_data1, train_data2, train_labels, train_usrds_id, train_timestamps,
                attn1_mask=None, attn1_dropout_p=0.0, attn1_is_causal=True, 
                attn2_mask=None, attn2_dropout_p=0.0, attn2_is_causal=True, 
                batch_size=32, max_epochs=10, learning_rate: float=0.1):
        train_dataset = PatLevelDataset(train_data1, train_data2, train_labels, train_usrds_id, train_timestamps)
        train_dataloader = DataLoader(train_dataset, collate_fn=PatLevel_collate_fn, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(list(self.encoder1.parameters()) + list(self.encoder2.parameters()),
                                     lr=learning_rate)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * max_epochs)
        cl_criterion = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        
        self.encoder1.to(self.device); self.encoder2.to(self.device);
        for ep in range(max_epochs):
            self.encoder1.train(); self.encoder2.train();
            for iteration, (batch_pat_usrds_id, batch_pat_lens, batch_pat_timestamps, 
                            batch_pad_pat_mask, batch_pad_pat_feats1, batch_pad_pat_feats2, batch_pad_pat_labels) in enumerate(train_dataloader):
                
                survival_prob = []
                for i in range(len(batch_pat_timestamps)):
                    kmf = lifelines.KaplanMeierFitter()
                    kmf.fit([int(item[-1] - item[0]) for idx, item in enumerate(batch_pat_timestamps) if idx != i], 
                            [int(item[mask][-1]) for idx, (item, mask) in enumerate(zip(batch_pad_pat_labels, batch_pad_pat_mask)) if idx != i])
                    survival_prob.append(np.array(kmf.survival_function_at_times(batch_pat_timestamps[i].numpy())))
                survival_prob = np.concatenate(survival_prob)
                survival_prob = np.concatenate((survival_prob, survival_prob))
                survival_prob = torch.tensor(1 - np.abs(survival_prob.reshape(-1,1) - survival_prob)).to(self.device)

                batch_pad_pat_feats1 = batch_pad_pat_feats1.float().to(self.device)
                batch_pad_pat_feats2 = batch_pad_pat_feats2.float().to(self.device)
                batch_pad_pat_labels = batch_pad_pat_labels.float().to(self.device)
 
                batch_embs1 = self.encoder1(batch_pad_pat_feats1, attn1_mask, attn1_dropout_p, attn1_is_causal)
                batch_embs2 = self.encoder2(batch_pad_pat_feats2, attn2_mask, attn2_dropout_p, attn2_is_causal)  
                
                batch_pat_timestamps = torch.cat(batch_pat_timestamps)
                CLpos_logits, CLpos_labels = self.CL_posloss(batch_embs1, batch_embs2, batch_pad_pat_mask, 
                                                             batch_pat_timestamps, batch_pat_usrds_id, batch_pat_lens, survival_prob)
                CLpos_loss = cl_criterion(CLpos_logits, CLpos_labels)
                
                CPC_loss = self.CPCloss(batch_embs1, batch_embs2, batch_pad_pat_mask)
                
                cl_loss = CLpos_loss + CPC_loss

                optimizer.zero_grad()
                cl_loss.backward()
                optimizer.step()
                lr_sched.step()
                
                if iteration % 100 == 0:
                    print('[%s] Epoch: %d; Iteration: %d; CL Loss: %.4f; CPC Loss: %.4f' % 
                              (str(datetime.now()), ep, iteration, CLpos_loss, CPC_loss))

def CL_get_encoded_feats(cl_model, extracted_data1, extracted_data2, extracted_labels, extracted_usrds_id, extracted_timestamps):
    extracted_dataset = PatLevelDataset(extracted_data1, extracted_data2, extracted_labels, extracted_usrds_id, extracted_timestamps)
    extracted_dataloader = DataLoader(extracted_dataset, collate_fn=PatLevel_collate_fn, batch_size=32, shuffle=False)
        
    cl_model.encoder1.eval(); cl_model.encoder2.eval()
    CL_feat1, CL_feat2 = [], []
    with torch.no_grad():
        for (batch_pat_usrds_id, batch_pat_lens, batch_pat_timestamps, batch_pad_pat_mask,
             batch_pad_pat_feats1, batch_pad_pat_feats2, batch_pad_pat_labels) in tqdm(extracted_dataloader):
            batch_pad_pat_feats1 = batch_pad_pat_feats1.float().to(cl_model.device) 
            batch_pad_pat_feats2 = batch_pad_pat_feats2.float().to(cl_model.device) 
            embs1 = cl_model.encoder1(batch_pad_pat_feats1, attn_mask=None, dropout_p=0., is_causal=True).cpu()
            embs2 = cl_model.encoder2(batch_pad_pat_feats2, attn_mask=None, dropout_p=0., is_causal=True).cpu()
            
            remove_pad_embs1 = embs1[batch_pad_pat_mask]
            remove_pad_embs2 = embs2[batch_pad_pat_mask]
            CL_feat1.append(remove_pad_embs1); CL_feat2.append(remove_pad_embs2)
    
    norm_CL_feat1 = F.normalize(torch.cat(CL_feat1))
    norm_CL_feat2 = F.normalize(torch.cat(CL_feat2))
    return  norm_CL_feat1, norm_CL_feat2


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
    parser.add_argument('--generate_data_path', help='The directory for saving generate data', type=str, default='./generate_data/')	
    parser.add_argument('--encoders_res', help='The directory for saving pretrained encoders and the generate features', type=str, default='./encoders_res/')
	
	args = parser.parse_args()
	with open(args.generate_data_path + 'generate_data.pkl', 'rb') as f:
		all_data = pickle.load(f) 
	
	pre_encoders = AttentionCPCCL(
		feat1_dim = 1024 // 2, feat2_dim = 1024 // 2,
    		device='cuda:0', 
    		CL_temp=0.1,
    		attn1_layers=2, attn1_d_model=512, attn1_num_heads=4,
    		attn2_layers=2, attn2_d_model=512, attn2_num_heads=4,
    		cpc_steps=1, cpc_k = 12,
		    window_size = 2,
	)
	pre_encoders.CLtrain(
    		all_data['train']['data1'], all_data['train']['data2'], all_data['train']['label'], 
    		all_data['train']['id'], 
    		all_data['train']['timestamp'],
    		attn1_mask=None, attn1_dropout_p=0.2, attn1_is_causal=True, 
    		attn2_mask=None, attn2_dropout_p=0.2, attn2_is_causal=True, 
    		batch_size=64, max_epochs=50, learning_rate=1e-4
	)

    train_data1_feats, train_data2_feats = CL_get_encoded_feats(pre_encoders, all_data['train']['data1'], all_data['train']['data2'], all_data['train']['label'],
                                                                    all_data['train']['id'], all_data['train']['timestamp'])

	val_data1_feats, val_data2_feats = CL_get_encoded_feats(pre_encoders, all_data['val']['data1'], all_data['val']['data2'], all_data['val']['label'],
                                                                all_data['val']['id'], all_data['val']['timestamp'])

	test_data1_feats, test_data2_feats = CL_get_encoded_feats(pre_encoders, all_data['test']['data1'], all_data['test']['data2'], all_data['test']['label'],
                                                                  all_data['test']['id'], all_data['test']['timestamp'])
	
	torch.save(
		{'encoder1': pre_encoders.encoder1.state_dict(),
     		'encoder2': pre_encoders.encoder2.state_dict(),
     		'train_feat': {'data1': train_data1_feats, 'data2': train_data2_feats},
     		'val_feat': {'data1': val_data1_feats, 'data2': val_data2_feats},
     		'test_feat': {'data1': test_data1_feats, 'data2': test_data2_feats},
		},
        args.encoders_res + 'encoders_ck_feats.pt')	


