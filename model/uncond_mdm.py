import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDM(nn.Module):
    #figure out whats going on with this
    def __init__(self, modeltype, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d',
                 dataset='addb', **kwargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kwargs.get('normalize_encoder_output', False)

        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                 num_layers=self.num_layers)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
        self.output_process = OutputProcess(self.data_rep, self.input_feats, 
                                            self.latent_dim, self.njoints,
                                            self.nfeats)
        
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)

        x = self.input_process(x)
        xseq = torch.cat((emb, x), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[1:]
        output = self.output_process(output)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequeunce_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequeunce_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(nn.Linear(self.latent_dim, time_embed_dim),
                                        nn.SiLU(),
                                        nn.Linear(time_embed_dim, time_embed_dim))
        
    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
    
class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        print(x.shape)
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0 ,1 ,2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]
            first_pose = self.poseEmbedding(first_pose)
            vel = x[1:]
            vel = self.velEmbedding(vel)
            return torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError
        
class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]
            first_pose = self.poseFinal(first_pose)
            vel = output[1:]
            vel = self.velFinal(vel)
            output = torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3 ,0)
        return output
