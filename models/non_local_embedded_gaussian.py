import torch
from torch import nn
from torch.nn import functional as F


def _generate_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class MultiHeadTransformer(nn.Module):
    def __init__(self, in_channels, inter_channels=None, n_views=4, seq_len=4, H=16, W=16, mask_subsequent=False):
        super(MultiHeadTransformer, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.la, self.ra = [11, 12, 13], [14, 15, 16]
        self.ll, self.rl = [4, 5, 6], [1, 2, 3]
        self.cb = [0, 7, 8, 9, 10]
        self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

        self.heads = ['joints', 'temporal', 'views']
        self.g = nn.ModuleList(
            [nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
            ]
        )
        self.theta = nn.ModuleList(
            [nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             ]
        )
        self.phi = nn.ModuleList(
            [nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0),
             ]
        )
        self.W = nn.Conv1d(in_channels=3 * self.inter_channels, out_channels=self.in_channels,
                       kernel_size=1, stride=1, padding=0)
        # self.LN = nn.LayerNorm([self.in_channels * H * W, n_views, seq_len, 17])
        self.LN = nn.GroupNorm(num_groups=1, num_channels=self.in_channels * H * W)
        self.feed_forward = nn.Sequential(
                nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                nn.LayerNorm([self.in_channels, H*W]),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
        )
        # self.LN_final = nn.LayerNorm([self.in_channels * H * W, n_views, seq_len, 17])
        self.LN_final = nn.GroupNorm(num_groups=1, num_channels=self.in_channels * H * W)
        if mask_subsequent:
            self.mask = _generate_subsequent_mask(sz=seq_len).cuda()
        else:
            self.mask = None

    def forward(self, x):
        '''
        :param x: (b, c*h*w, views, t, v)
        :return:
        '''
        B, C, Views, T, V = x.shape
        Y = []
        for head in self.heads:
            if head == 'joints':
                x_h = x.permute(0, 2, 3, 1, 4).contiguous().view(-1, C, V)
                x_pool = x_h.view(B*Views*T, self.in_channels, C // self.in_channels, V).mean(2)  # (b*views*t), c, v
                x_g = x_h.view(B*Views*T, self.in_channels, C // self.in_channels, V).transpose(1, 2).contiguous().view(-1, self.in_channels, V)  # (b*views*t*h*w), C, v

                g_x = self.g[0](x_g).view(B*Views*T, C // self.in_channels, self.inter_channels, V).transpose(1, 2).contiguous().view(-1, self.inter_channels * C // self.in_channels, V)
                g_x = g_x.permute(0, 2, 1)  # b*views*t, v, c'*h*w

                theta_x = self.theta[0](x_pool).view(B*Views*T, self.inter_channels, V)
                theta_x = theta_x.permute(0, 2, 1)  # b*views*t, v, c'
                phi_x = self.phi[0](x_pool).view(B*Views*T, self.inter_channels, V)  # b*views*t, c', v
                f = torch.matmul(theta_x, phi_x)
                f_div_C = F.softmax(f, dim=-1)
                y = torch.matmul(f_div_C, g_x)  # b*views*t, v, c'*h*w
                y = y.view(B, Views, T, V, -1).permute(0, 4, 1, 2, 3).view(B, self.inter_channels, C // self.in_channels, Views, T, V)
                Y.append(y)
            elif head == 'temporal':
                x_h = x.permute(0, 2, 4, 1, 3).contiguous().view(-1, C, T)
                x_pool = x_h.view(B*Views*V, self.in_channels, C // self.in_channels, T).mean(2)  # (b*views*v), c, t
                x_g = x_h.view(B*Views*V, self.in_channels, C //  self.in_channels, T).transpose(1, 2).contiguous().view(-1, self.in_channels, T)  # (b*views*v*h*w), C, t

                g_x = self.g[1](x_g).view(B*Views*V, C // self.in_channels, self.inter_channels, T).transpose(1, 2).contiguous().view(-1, self.inter_channels * C // self.in_channels, T)
                g_x = g_x.permute(0, 2, 1)  # b*views*v, t, c'*h*w

                theta_x = self.theta[1](x_pool).view(B * Views * V, self.inter_channels, T)
                theta_x = theta_x.permute(0, 2, 1)  # b*views*v, t, c'
                phi_x = self.phi[1](x_pool).view(B * Views * V, self.inter_channels, T)  # b*views*v, c', t
                f = torch.matmul(theta_x, phi_x)
                if self.mask is not None:
                    f = f + self.mask.unsqueeze(0)
                f_div_C = F.softmax(f, dim=-1)
                y = torch.matmul(f_div_C, g_x)  # b*views*v, t, c'*h*w
                y = y.view(B, Views, V, T, -1).permute(0, 4, 1, 3, 2).view(B, self.inter_channels, C // self.in_channels, Views, T, V)
                Y.append(y)
            elif head == 'views':
                ys = []
                for part in range(len(self.part)):
                    V_p = len(self.part[part])
                    x_h = x[:, :, :, :, self.part[part]]  # b, c*h*w, views, t, v
                    x_h = x_h.transpose(2, 3).contiguous().view(B, C, T, -1).transpose(1, 2).contiguous()  # b, t, c, views*v
                    x_pool = x_h.view(B * T, self.in_channels, C // self.in_channels, Views*V_p).mean(2)  # (b*t), c, views*v
                    x_g = x_h.view(B * T, self.in_channels, C // self.in_channels, Views*V_p).transpose(1, 2).contiguous().view(
                        -1, self.in_channels, Views*V_p)  # (b*t*h*w), C, views*v

                    g_x = self.g[2 + part](x_g).view(B * T, C // self.in_channels, self.inter_channels, Views*V_p).transpose(1, 2).contiguous().view(-1,
                                                                                          self.inter_channels * C // self.in_channels,
                                                                                          Views*V_p)
                    g_x = g_x.permute(0, 2, 1)  # b*t, views*v, c'*h*w

                    theta_x = self.theta[2 + part](x_pool).view(B * T, self.inter_channels, Views*V_p)
                    theta_x = theta_x.permute(0, 2, 1)  # b*t, views*v, c'
                    phi_x = self.phi[2 + part](x_pool).view(B * T, self.inter_channels, Views*V_p)  # b*t, c', views*v
                    f = torch.matmul(theta_x, phi_x)
                    f_div_C = F.softmax(f, dim=-1)
                    y = torch.matmul(f_div_C, g_x)  # b*t, views*v, c'*h*w
                    y = y.view(B, T, Views, V_p, -1).permute(0, 4, 2, 1, 3)  # b, c'*h*w, views, t, v
                    ys.append(y)
                ys = torch.cat(ys, -1)
                ys = ys[:, :, :, :, (12, 9, 10, 11, 6, 7, 8, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5)]
                Y.append(ys.view(B, self.inter_channels, C // self.in_channels, Views, T, V))
            else:
                raise Exception(f'{head} is not valid key of heads.')
        Y = torch.cat(Y, 1)
        Y = Y.view(B, -1, Views*T*V).permute(0, 2, 1).contiguous().view(B*Views*T*V, -1, C // self.in_channels)
        z = self.W(Y).view(B, Views, T, V, -1).permute(0, 4, 1, 2, 3) + x
        z = self.LN(z)  # b, c*h*w, views, t, v
        f = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.in_channels, C // self.in_channels)  # b*views*t*v, c, h*w
        f = self.LN_final(self.feed_forward(f).view(B, Views, T, V, C).permute(0, 4, 1, 2, 3) + z)
        return f


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=False):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, views, t, v)
        :return:
        '''

        batch_size, channels = x.shape[:2]

        x_pool = x.view(batch_size, self.in_channels, channels//self.in_channels, *x.shape[2:]).mean(2)
        x_g = x.view(batch_size, self.in_channels, channels//self.in_channels, *x.shape[2:]).transpose(1, 2).contiguous().view(-1, self.in_channels, *x.shape[2:])  # (b*h*w), C, views, t, v

        g_x = self.g(x_g).view(batch_size, channels//self.in_channels, self.inter_channels, -1).transpose(1, 2).contiguous().view(batch_size, self.inter_channels * channels//self.in_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # b, views*t*v, c'*h*w

        theta_x = self.theta(x_pool).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # b, views*t*v, c'
        phi_x = self.phi(x_pool).view(batch_size, self.inter_channels, -1)  # b, c', views*t*v
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)  # b, views*t*v, c'*h*w
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, channels//self.in_channels, *x.size()[2:]).transpose(1, 2).contiguous().view(-1, self.inter_channels, *x.size()[2:])
        W_y = self.W(y).view(batch_size, channels//self.in_channels, self.in_channels, *x.size()[2:]).transpose(1, 2).contiguous().view(batch_size, -1, *x.size()[2:])
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20)
        net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.zeros(2, 3, 20, 20)
        net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.randn(2, 3, 8, 20, 20)
        net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())


