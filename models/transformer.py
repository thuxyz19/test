import torch
import torch.nn as nn
import math
from mvn_new.utils import multiview
import torch.nn.functional as F

class MotionNet(nn.Module):

    def __init__(self, num_joints, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, activation='gelu', disturb=1.0, mask_subsequent=True):
        super().__init__()
        self.embedding = nn.Linear(num_joints * 3, d_model)
        self.position = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.motion = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                     dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model // 2),
                                     nn.GELU(),
                                     nn.Linear(d_model // 2, (num_joints + 1) * 3))
        #self.angle_decoder = nn.Sequential(nn.Linear(d_model, num_joints),
        #                                   nn.Sigmoid())
        self.motion_disturb = disturb
        self.mask_subsequent = mask_subsequent

    def forward(self, keypoints_3d_history):
        # keypoints_3d_history [batch, (frames-1), 17, 3]
        batch_size, frames_history = keypoints_3d_history.shape[:2]
        keypoints_3d_history = keypoints_3d_history.transpose(1, 0).contiguous()
        src = self.embedding(keypoints_3d_history.view(-1, 51)).view(frames_history, batch_size, -1)
        src = self.position(src)
        if self.mask_subsequent:
            src_mask = self.motion.generate_square_subsequent_mask(frames_history).cuda()
            tgt_mask = self.motion.generate_square_subsequent_mask(frames_history).cuda()
            mem_mask = self.motion.generate_square_subsequent_mask(frames_history).cuda()
        else:
            src_mask = torch.zeros(frames_history, frames_history).cuda().float()
            tgt_mask = torch.zeros(frames_history, frames_history).cuda().float()
            mem_mask = torch.zeros(frames_history, frames_history).cuda().float()
        pred = self.motion(src=src, tgt=src, src_mask=src_mask,
                           tgt_mask=tgt_mask, memory_mask=mem_mask)  # frames-1, batch, 256
        pred = self.decoder(pred.view(-1, pred.shape[-1])) \
            .view(frames_history, batch_size, -1) \
            .transpose(1, 0).contiguous() \
            .view(batch_size, frames_history, 17 + 1, 3)  # batch, frames-1, 17, 3
        #axis = axis / torch.clamp_min(torch.norm(axis, p=2, dim=-1, keepdim=True), min=1e-5)
        #angle = 2 * 3.1415926 * (self.angle_decoder(pred.view(-1, pred.shape[-1])).view(frames_history, batch_size, -1).transpose(1, 0).contiguous() + 1.0)
        #pred = axis * angle.unsqueeze(-1)
        return pred


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ViewSelect(nn.Module):

    def __init__(self, num_joints, d_model=32, nhead=4, num_encoder_layers=3, dropout=0.1, activation='gelu'):
        super().__init__()
        self.num_joints = num_joints
        encoder_layer_skeleton_2d = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation=activation)
        self.transformer_encoder_skeleton_2d = nn.TransformerEncoder(encoder_layer_skeleton_2d, num_layers=num_encoder_layers)
        self.embedding_skeleton_2d = nn.Linear(2, d_model)
        self.position_skeleton_2d = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer_position_2d = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation=activation)
        self.transformer_encoder_position_2d = nn.TransformerEncoder(encoder_layer_position_2d,
                                                                     num_layers=num_encoder_layers)
        self.embedding_position_2d = nn.Linear(2, d_model)
        self.position_position_2d = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer_position_3d = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation=activation)
        self.transformer_encoder_position_3d = nn.TransformerEncoder(encoder_layer_position_3d,
                                                                     num_layers=num_encoder_layers)
        self.embedding_position_3d = nn.Linear(3, d_model)
        self.position_position_3d = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.embedding_heat_var = nn.Linear(1, d_model)

        self.decoder = nn.Sequential(nn.Linear(d_model * 4, d_model),
                                     nn.GELU(),
                                     nn.Linear(d_model, 1))

    def forward(self, keypoints_2d, keypoints_3d, var):
        # heatmaps (B, frames, 4, 17, H, W)
        # keypoints_3d_list (4, B, frames, 17, 3)
        B, frames, views, _, _ = keypoints_2d.shape
        # keypoints_2d = multiview.get_heat_2d(heatmaps.view(-1, views, self.num_joints, H, W)).view(-1, 17, 2)  # B*frames*4, 17, 2

        keypoints_2d = keypoints_2d.view(B*frames*views, 17, 2)
        skeleton_2d = keypoints_2d.transpose(1, 0).contiguous()
        skeleton_2d = self.embedding_skeleton_2d(skeleton_2d.view(-1, 2)).view(self.num_joints, B*frames*views, -1)
        skeleton_2d = self.position_skeleton_2d(skeleton_2d)  # 17, B*frames*4, d_model
        skeleton_2d = self.transformer_encoder_skeleton_2d(skeleton_2d).transpose(1, 0).contiguous()  # B*frames*4, 17, d_model

        position_2d = keypoints_2d.view(B, frames, views, self.num_joints, 2).transpose(1, 0).contiguous().view(frames, -1, 2)  # frames, B*4*17, 2
        position_2d = self.embedding_position_2d(position_2d.view(-1, 2)).view(frames, B*views*self.num_joints, -1)
        position_2d = self.position_position_2d(position_2d)  # frames, B*4*17, d_model
        position_2d = self.transformer_encoder_position_2d(position_2d).view(frames, B, views, self.num_joints, -1).transpose(1, 0).contiguous().view(B*frames*views, 17, -1)  # B*frames*4, 17, d_model

        keypoints_3d_list = keypoints_3d.view(views, B, frames, 17, 3)
        keypoints_3d_list = keypoints_3d_list.transpose(1, 0).contiguous()  # B, 4, frames, 17, 3
        position_3d = keypoints_3d_list.view(B*views, frames*self.num_joints, 3).transpose(1, 0).contiguous()  # frames*17, B*4, 3
        position_3d = self.embedding_position_3d(position_3d.view(-1, 3)).view(frames*self.num_joints, B*views, -1)
        position_3d = self.position_position_3d(position_3d)  # frames*17, B*4, d_model
        position_3d = self.transformer_encoder_position_3d(position_3d).transpose(1, 0).contiguous().view(B, views, frames, self.num_joints, -1).transpose(2, 1).contiguous().view(B*frames*views, self.num_joints, -1)  # B*frames*4, 17, d_model

        # heat_var = multiview.get_heat_variance(heatmaps.view(-1, views, self.num_joints, H, W)).view(B*frames*views, self.num_joints, 1)

        heat_var = self.embedding_heat_var(var.view(-1, 1)).view(B*frames*views, self.num_joints, -1)

        feature = torch.cat([skeleton_2d, position_2d, position_3d, heat_var], dim=-1)  # B*frames*4, 17, d_model*4
        rank = self.decoder(feature.view(B*frames*views*self.num_joints, -1)).view(B, frames, views, self.num_joints).transpose(3, 2).contiguous()
        rank = rank.view(B*frames, self.num_joints, views)

        return rank









