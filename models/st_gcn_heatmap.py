import torch
import torch.nn as nn

from mvn_new.models.utils.tgcn import HeatmapGraphical
from mvn_new.models.utils.graph_frames import Graph
from mvn_new.models.utils.graph_frames_withpool_2 import Graph_pool
from mvn_new.models.non_local_embedded_gaussian import NONLocalBlock3D, MultiHeadTransformer

BN_MOMENTUM = 0.1
inter_channels = [16, 16, 16]

fc_out = inter_channels[-1]
fc_unit = 16

class GlobalAveragePoolingHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=False),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=False),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)

        batch_size, n_channels = x.shape[:2]
        x = x.view(batch_size, n_channels, -1)
        x = x.mean(dim=-1)

        out = self.head(x)

        return out


class Transformer(nn.Module):
    def __init__(self, use_confidences=True, mask_subsequent=False):
        super().__init__()
        self.in_channels = 18
        self.out_channels = 17
        self.use_confidences = use_confidences
        self.encoder = nn.Sequential(MultiHeadTransformer(in_channels=self.in_channels, n_views=4, seq_len=5, mask_subsequent=mask_subsequent),
                                     MultiHeadTransformer(in_channels=self.in_channels, n_views=4, seq_len=5, mask_subsequent=mask_subsequent),
                                     MultiHeadTransformer(in_channels=self.in_channels, n_views=4, seq_len=5, mask_subsequent=mask_subsequent))
        if self.use_confidences:
            self.alg_confidence = GlobalAveragePoolingHead(self.in_channels, 1)

        self.fcn = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        )


    def forward(self, x):
        N, Views, T, V, C, H, W = x.size()
        x = x.view(N, Views, T, V, C*H*W).permute(0, 4, 1, 2, 3).contiguous()
        x = self.encoder(x)  # N, C*H*W, Views, T, V
        x = x.view(N, -1, Views * T * V).transpose(1, 2).contiguous().view(N * Views * T * V, -1, H, W)
        alg_confidences = None
        if self.use_confidences:
            alg_confidences = self.alg_confidence(x).view(-1, V)  # N*Views*T, V
        x = self.fcn(x)  # N*Views*T*V, 17, H, W
        # output
        x = x.view(N, Views, T, V, self.out_channels, H, W)
        return x, alg_confidences


class TransformerViews(nn.Module):
    def __init__(self, mask_subsequent=False):
        super().__init__()
        self.in_channels = 18 + 3
        self.out_channels = 17
        self.encoder = nn.Sequential(MultiHeadTransformer(in_channels=self.in_channels, n_views=4, seq_len=5, mask_subsequent=mask_subsequent),
                                     MultiHeadTransformer(in_channels=self.in_channels, n_views=4, seq_len=5, mask_subsequent=mask_subsequent),
                                     MultiHeadTransformer(in_channels=self.in_channels, n_views=4, seq_len=5, mask_subsequent=mask_subsequent))

        self.alg_confidence = GlobalAveragePoolingHead(self.in_channels, 1)



    def forward(self, x):
        N, Views, T, V, C, H, W = x.size()
        x = x.view(N, Views, T, V, C*H*W).permute(0, 4, 1, 2, 3).contiguous()
        x = self.encoder(x)  # N, C*H*W, Views, T, V
        x = x.view(N, -1, Views * T * V).transpose(1, 2).contiguous().view(N * Views * T * V, -1, H, W)
        alg_confidences = self.alg_confidence(x).view(-1, V)  # N*Views*T, V

        return alg_confidences




class Model(nn.Module):
    """

    Args:
        in_channels (int): Number of channels in the input data
        cat: True: concatinate coarse and fine features
            False: add coarse and fine features

    Shape:
        - Input: :math:`(N, Views_{in}, T_{in}, V_{in}, in_channels, H_{in}, W_{in})`
            :math:`N` is a batch size,
            :math:`Views` is the number of views
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes for each frame,
            :math:`H_{in}`, 'W_{in}' is the spatial shape of the heatmap of a node.
    Return:
        out_all_frame: True: return all frames 3D results
                        False: return target frame result

        x_out: final output.

    """

    def __init__(self, use_confidences=True, is_transformer=False):
        super().__init__()

        # load graph
        self.momentum = 0.1
        self.in_channels = 18
        self.out_channels = 17
        self.cat = True
        self.inplace = False
        self.use_confidences = use_confidences

        # original graph
        self.graph = Graph(seqlen=5, n_views=4)
        # get adjacency matrix of K clusters
        # self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda() # K, T*V, T*V
        self.A = self.graph.A

        # pooled graph
        self.graph_pool = Graph_pool(seqlen=5, n_views=4)
        # self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cuda()
        self.A_pool = self.graph_pool.A


        # build networks
        kernel_size = len(self.A)
        kernel_size_pool = len(self.A_pool)

        self.data_bn = nn.BatchNorm1d(self.in_channels * self.graph.num_node_each, self.momentum)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(self.in_channels, inter_channels[0], kernel_size, residual=False),
            st_gcn(inter_channels[0], inter_channels[1], kernel_size),
            st_gcn(inter_channels[1], inter_channels[2], kernel_size),
        ))


        self.st_gcn_pool = nn.ModuleList((
            st_gcn(inter_channels[-1], fc_unit, kernel_size_pool),
            st_gcn(fc_unit, fc_unit,kernel_size_pool),
        ))


        self.conv4 = nn.Sequential(
            nn.Conv2d(fc_unit, fc_unit, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(fc_unit, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.00)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(fc_unit*2, fc_out, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(fc_out, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.0)
        )

        fc_in = self.in_channels + fc_out if self.cat else inter_channels[-1]

        # self.non_local = NONLocalBlock3D(in_channels=fc_out*2, sub_sample=False)
        if is_transformer:
            self.non_local = MultiHeadTransformer(in_channels=fc_in, n_views=4, seq_len=5)
        else:
            self.non_local = NONLocalBlock3D(in_channels=fc_in, sub_sample=False)

        # fcn for final layer prediction

        # fc_in = fc_in * self.A['root'].shape[0]
        self.fcn = nn.Sequential(
            nn.Dropout(0.0, inplace=False),
            nn.BatchNorm2d(fc_in),
            nn.Conv2d(fc_in, self.out_channels, kernel_size=1)
        )

        if self.use_confidences:
            self.alg_confidence = GlobalAveragePoolingHead(fc_in, 1)



    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        x = nn.MaxPool1d(p)(x)  # B, F, V/p
        return x


    def forward(self, x, out_all_frame=True):

        N, Views, T, V, C, H, W = x.size()
        x_origin = x.clone()
        # x = x.view(N, Views*T*V, C, 1, -1)  # N, (Views*T*V), C, H, W

        # forwad GCN
        gcn_list = list(self.st_gcn_networks)
        for i_gcn, gcn in enumerate(gcn_list):
            x, _ = gcn(x, self.A)  # N, Views, T, V, C, H, W

        N, _, _, _, C, H, W = x.size()
        x = x.view(N*Views*T, V, -1).transpose(1, 2).contiguous()  # (N*Views*T), (C*H*W), V
        x_origin = x_origin.view(N * Views * T, V, -1).transpose(1, 2).contiguous()  # (N*Views*T), (C*H*W), V
        # Pooling
        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_i = x[:, :, self.graph.part[i]]
            x_i = self.graph_max_pool(x_i, num_node)
            x_sub1 = torch.cat((x_sub1, x_i), -1) if i > 0 else x_i # Final to (N*Views*T), (C*H*W), (NUM_SUB_PARTS)
        x_sub1 = x_sub1.transpose(1, 2).contiguous().view(N, Views, T, len(self.graph.part), -1, H, W)
        x_sub2 = x_sub1.clone()
        x_sub1, _ = self.st_gcn_pool[0](x_sub1, self.A_pool)  # N, Views, T, NUM_SUB_PARTS, C, H, W
        x_sub1, _ = self.st_gcn_pool[1](x_sub1, self.A_pool)  # N, Views, T, NUM_SUB_PARTS, C, H, W
        x_sub1 = x_sub1.view(N*Views*T, len(self.graph.part), -1).transpose(1, 2).contiguous()
        x_sub2 = x_sub2.view(N * Views * T, len(self.graph.part), -1).transpose(1, 2).contiguous()

        x_pool_1 = self.graph_max_pool(x_sub1, len(self.graph.part))  # (N*Views*T), (C*H*W), 1
        x_pool_1 = x_pool_1.view(N*Views*T, -1, H, W)
        x_pool_1 = self.conv4(x_pool_1).view(N*Views*T, -1, 1)  # (N*Views*T), (C*H*W), 1
        x_up_sub = torch.cat((x_pool_1.repeat(1, 1, len(self.graph.part)).view(N*Views*T, -1, H, W, len(self.graph.part)), x_sub2.view(N*Views*T, -1, H, W, len(self.graph.part))), 1)  # N*Views*T, C+C, H, W, len(self.graph.part)
        x_up_sub = x_up_sub.view(N*Views*T, -1, len(self.graph.part)).transpose(2, 1).contiguous().view(N*Views*T*len(self.graph.part), -1, H, W)
        x_up_sub = self.conv2(x_up_sub)  # (N*Views*T*len(self.graph.part)), -1, H, W

        _, C, H, W = x_up_sub.size()
        x_up_sub = x_up_sub.view(N*Views*T, -1, C*H*W).transpose(1, 2)
        # upsample
        x_up = x_up_sub[:, :, (4, 3, 3, 3, 2, 2, 2, 4, 4, 4, 4, 0, 0, 0, 1, 1, 1)]
        # for i in range(len(self.graph.part)):
        #     num_node = len(self.graph.part[i])
        #     x_up[:, :, self.graph.part[i]] = x_up_sub[:, :, i].unsqueeze(-1).repeat(1, 1, num_node)


        #for non-local and fcn
        x = torch.cat((x_origin.view(N*Views*T, -1, H, W, V), x_up.view(N*Views*T, -1, H, W, V)), 1)  # (N*Views*T), C, H, W, V
        x = x.view(N, Views, T, -1, V).permute(0, 3, 1, 2, 4)  # N, C*H*W, Views, T, V

        x = self.non_local(x)  # N, C*H*W, Views, T, V
        x = x.view(N, -1, Views*T*V).transpose(1, 2).contiguous().view(N*Views*T*V, -1, H, W)
        alg_confidences = None
        if self.use_confidences:
            alg_confidences = self.alg_confidence(x).view(-1, V)  # N*Views*T, V
        x = self.fcn(x)  # N*Views*T*V, 17, H, W


        # output
        x = x.view(N, Views, T, V, self.out_channels, H, W)
        if out_all_frame:
            x_out = x
        else:
            x_out = x[:, :, self.pad].unsqueeze(2)
        return x_out, alg_confidences

class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size :number of the node clusters

        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, Views, T, V, in_channels, H, W)` format
        - Input[1]: Input graph adjacency matrix in :math:`[(V_{k}, V_{k})] * K` format
        - Output[0]: Outpu graph sequence in :math:`(N, Views, T, V, out_channels, H, W)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`[(V_{k}, V_{k})] * K` format

        where
            :math:`N` is a batch size,
            :math:`Views` is the number of views
            :math:`K` is the kernel size
            :math:`V_{k}` is the number of nodes in the K-th type of connections
            :math:`T` is a length of sequence,
            :math:`V` is the number of graph nodes of each frame.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout=0.00,
                 residual=True):

        super().__init__()
        self.inplace = False
        self.relations = ['root', 'close', 'further', 'sym', 'forward', 'backward', 'n_views']
        assert len(self.relations) == kernel_size
        self.momentum = 0.1
        self.gcn = nn.ModuleList()
        self.tcn = nn.ModuleList()
        self.residual = nn.ModuleList()
        for k in range(kernel_size):
            self.gcn.append(HeatmapGraphical(in_channels, out_channels))

            self.tcn.append(nn.Sequential(
                nn.BatchNorm2d(out_channels, momentum=self.momentum),
                nn.ReLU(inplace=self.inplace),
                nn.Dropout(0.00),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (1, 1),
                    (1, 1),
                    padding=0,
                ),
                nn.BatchNorm2d(out_channels, momentum=self.momentum),
                nn.Dropout(dropout, inplace=self.inplace),
            ))

            if not residual:
                self.residual.append(ZerosMapping())

            elif in_channels == out_channels:
                self.residual.append(Identity())

            else:
                self.residual.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=(1, 1)),
                    nn.BatchNorm2d(out_channels, momentum=self.momentum),
                ))

        self.relu = nn.ReLU(inplace=self.inplace)

    def forward(self, x, A):
        N, Views, T, V, in_channels, H, W = x.size()
        x_agg = []
        for k in range(len(self.relations)):
            key = self.relations[k]
            if key == 'root' or key == 'close' or key == 'further' or key == 'sym':
                x_k = x.view(N, -1, V, in_channels, H, W)
            elif key == 'forward' or key == 'backward':
                x_k = x.transpose(2, 3).contiguous().view(N, -1, T, in_channels, H, W)
            elif key == 'n_views':
                x_k = x.view(N, Views, -1, in_channels, H, W).transpose(1, 2).contiguous()
            else:
                raise Exception(f'{key} is not a valid key of relationship')
            A_k = torch.from_numpy(A[key]).float().cuda()
            b, g, v, c, h, w = x_k.size()
            x_k = x_k.view(b*g*v, c, h, w)
            res = self.residual[k](x_k)
            x_k = x_k.view(b, g, v, c, h, w)
            x_k, _ = self.gcn[k](x_k, A_k)
            b, g, v, c, h, w = x_k.size()
            x_k = x_k.view(b*g*v, c, h, w)
            x_k = self.tcn[k](x_k) + res
            x_k = x_k.view(b, g, v, c, h, w)
            out_channels = x_k.shape[-3]
            if key == 'root' or key == 'close' or key == 'further' or key == 'sym':
                x_k = x_k.view(N, Views, T, V, out_channels, H, W)
            elif key == 'forward' or key == 'backward':
                x_k = x_k.view(N, Views, V, T, out_channels, H, W).transpose(2, 3).contiguous()
            elif key == 'n_views':
                x_k = x_k.transpose(1, 2).contiguous().view(N, Views, T, V, out_channels, H, W)
            else:
                raise Exception(f'{key} is not a valid key of relationship')
            x_agg.append(x_k)
        x_agg = torch.stack(x_agg, 0).sum(0)
        return self.relu(x_agg), A


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ZerosMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.0

