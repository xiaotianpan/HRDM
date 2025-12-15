import torch
import torch.nn as nn
import numpy as np
#from models.utils import img_text_similarlity


from model.helpers import SinusoidalPosEmb
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe


class StateEncoder(nn.Module):
    def __init__(self, vis_input_dim,  embed_dim, dropout=0.2,time_horz=3):#512,768,128lang_input_dim,
        super().__init__()
        self.time_horz = time_horz
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.state_encoder = nn.Sequential(
            nn.Linear(vis_input_dim, 2 * vis_input_dim),
            nn.Linear(2 * vis_input_dim, embed_dim)
        )
        self.time_mlp = nn.Sequential(  # should be removed for Noise and Deterministic Baselines
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.pos_encoder = PositionalEmbedding(
            d_model=embed_dim,
            max_len=self.time_horz
        )
        self.dropout = nn.Dropout(dropout)
        self.query_embed = nn.Linear(embed_dim, embed_dim)


    def process_state_query(self, state_feat,t):


        query = self.pos_encoder(state_feat)  # [batch_size, time_horz+1, embed_dim]

        query = query.permute(1, 0, 2)

        task_query = self.query_embed(t.clone().detach()).expand(self.time_horz, -1, -1)
        #print(task_query.shape)
        query = query + task_query

        return query

    def forward(self, state_feat):

        state_feat = self.state_encoder(self.dropout(state_feat))



        return state_feat
