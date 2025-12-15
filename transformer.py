import torch.nn as nn
import torch

from encoder import StateEncoder
from decoder import StateDecoder
# from models.action_decoder import ActionDecoder
# from models.utils import viterbi_path


class ProcedureModel(nn.Module):
    def __init__(
            self,
            vis_input_dim,
            #lang_input_dim,
            embed_dim,
            time_horz,
            attn_heads,
            mlp_ratio,
            num_layers,
            dropout
            #num_classes,
            #num_tasks,
            #args
    ):

        super().__init__()

        self.att_heads = attn_heads
        self.mlp_ratio = mlp_ratio
        self.num_layers = num_layers
        self.dropout = dropout

        self.time_horz = time_horz
        self.embed_dim = embed_dim


        self.state_encoder = StateEncoder(
            vis_input_dim,
            #lang_input_dim,
            embed_dim,
            dropout=0.2,
            time_horz=time_horz,
        )

        self.state_decoder = StateDecoder(
            embed_dim=embed_dim,
            output_dim=vis_input_dim,
            time_horz=time_horz,
            att_heads=self.att_heads,
            mlp_ratio=self.mlp_ratio,
            num_layers=self.num_layers,
            dropout=self.dropout,

        )


        self.dropout = nn.Dropout(self.dropout)



    def forward(
            self,
            visual_features,t,
    ):



        state_feat_encode = self.state_encoder(visual_features)

        state_feat_decode = self.state_decoder(
            state_feat_encode,
            t
        )
        

        return state_feat_decode
