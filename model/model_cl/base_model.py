import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model_base

from block import Attention, FFN, Gated_linear, Encoder, Bottleneck_Attention, Decoder, PositionalEmbedding, Encoder_new, Encoder_rope
from model.stf_model.gwnet import gcn

class Backbone(nn.Module):
    def __init__(self, model_name, config, data=None):
        super().__init__()
        self.model_name = model_name.lower()
        self.graph_model = False
        if model_name.lower() == 'st_transformer' or model_name.lower() == 'stf':
            self.model = ST_Transformer(config.emb_dim, config.s_layer, config.t_layer, config.s_head, config.t_head, config.dropout)
        elif model_name.lower() == 'st_transformer_new' or model_name.lower() == 'stf2':
            self.model = ST_Transformer_2(config.emb_dim, config.s_layer, config.t_layer, config.s_head, config.t_head, config.dropout)
        elif model_name.lower() == 'st_transformer_rope' or model_name.lower() == 'stf3':
            self.emb = nn.Linear(config.dim_in, config.emb_dim)
            self.model = ST_Transformer_3(config.emb_dim, config.s_layer, config.t_layer, config.s_head, config.t_head, config.dropout)
        elif model_name.lower() == 'ste_transformer' or model_name.lower() == 'ste':
            self.emb = ST_Embedding(config.dim_in, config.emb_dim, config.num_nodes, config.num_times)
            self.model = ST_Transformer_2(config.emb_dim, config.s_layer, config.t_layer, config.s_head, config.t_head, config.dropout)
        elif model_name.lower() == 'stf_transformer_4' or model_name.lower() == 'stf4':
            self.emb = Transformer_emb_v2(config.input_window, config.dim_in, config.emb_dim)
            self.model = ST_Transformer_v4(config.emb_dim, config.s_layer, config.t_layer, config.s_head, config.t_head, config.dropout)
            self.pred = Transformer_pred_v2(config.output_window, config.dim_out, config.emb_dim)
        elif model_name.lower() == 'stb_transformer' or model_name.lower() == 'stb':
            self.model = STB_Transformer(config.emb_dim, config.s_layer, config.t_layer, config.ib_num, config.s_head, config.t_head, config.dropout)
        elif model_name.lower() == 'stp':
            self.model = ST_Transformer_with_prompt(config.emb_dim, config.s_layer, config.t_layer, config.s_head, config.t_head, config.dropout)
        elif model_name.lower() == 'gtg':
            self.emb = lambda x:x.transpose(1, 2).flatten(2, 3)
            self.model = GTG(config)
            self.graph_model = True
        elif model_name.lower() == 'gwnet':
            self.emb = nn.Linear(config.dim_in, 32)
            self.model = gwnet('cuda:0', config.num_nodes)
            self.pred = gwnet_predictor(dim3=config.output_window * config.dim_in)
            self.graph_model = True
        elif model_name.lower() in ['itransformer', 'itrans', 'itr', 'itf', 'i_transformer']:
            num_head = config.get('num_head', config.get('num_heads', config.get('s_head', 8)))
            e_layer = config.get('e_layer', config.get('s_layer', 2))
            revin_state = {}
            self.emb = RevIN(
                Transformer_emb_v2(config.input_window, config.dim_in, config.emb_dim),
                mode='emb',
                state=revin_state
            )
            self.model = ST_Transformer_v4(
                config.emb_dim,
                s_layer=e_layer,
                s_head=num_head,
                dropout=config.dropout,
                encoder='old'
            )
            self.pred = RevIN(
                Transformer_pred_v2(config.output_window, config.dim_out, config.emb_dim),
                mode='pred',
                state=revin_state
            )
        if not hasattr(self, 'emb'):
            self.emb = Transformer_emb(config.dim_in, config.emb_dim)
        if not hasattr(self, 'pred'):
            self.pred = nn.Linear(config.emb_dim, config.dim_in)

    def forward(self, x, g=None, method='emd'):
        # B, T, N, D -> B, H, N, D
        if 'e' in method:
            x = self.emb(x)
        if 'm' in method:
            if self.graph_model:
                x = self.model(x, g)
            else:
                x = self.model(x)
        if 'd' in method:
            x = self.pred(x)
        return x

class ST_Transformer(nn.Module):
    def __init__(self, emb_dim, s_layer=2, t_layer=2, s_head=2, t_head=2, dropout=0.1):
        super().__init__()

        self.s_attn = Encoder(emb_dim, s_layer, s_head, dropout)
        self.t_attn = Encoder(emb_dim, t_layer, t_head, dropout, dim=1)

    def forward(self, x):
        x = self.t_attn(x)
        x = self.s_attn(x)
        return x

class ST_Transformer_2(nn.Module):
    def __init__(self, emb_dim, s_layer=2, t_layer=2, s_head=2, t_head=2, dropout=0.0):
        super().__init__()

        self.s_attn = Encoder_new(emb_dim, s_layer, s_head, dropout)
        self.t_attn = Encoder_new(emb_dim, t_layer, t_head, dropout, dim=1)

    def forward(self, x):
        x = self.t_attn(x)
        x = self.s_attn(x)
        return x

class ST_Transformer_3(nn.Module):
    def __init__(self, emb_dim, s_layer=2, t_layer=2, s_head=2, t_head=2, dropout=0.0):
        super().__init__()

        self.s_attn = Encoder_new(emb_dim, s_layer, s_head, dropout)
        self.t_attn = Encoder_rope(emb_dim, t_layer, t_head, dropout, dim=1)

    def forward(self, x):
        x = self.t_attn(x)
        x = self.s_attn(x)
        return x

class ST_Transformer_v4(nn.Module):
    def __init__(self, emb_dim, s_layer=2, t_layer=2, s_head=2, t_head=2, dropout=0.0, encoder='new'):
        super().__init__()

        if encoder == 'new':
            self.s_attn = Encoder_new(emb_dim, s_layer, s_head, dropout)
        else:
            self.s_attn = Encoder(emb_dim, s_layer, s_head, dropout)

    def forward(self, x):
        x = self.s_attn(x)
        return x

class ST_Transformer_with_prompt(nn.Module):
    def __init__(self, emb_dim, s_layer=2, t_layer=2, s_head=2, t_head=2, dropout=0.1):
        super().__init__()

        self.s_attn = Decoder(emb_dim, s_layer, s_head, dropout)
        self.t_attn = Decoder(emb_dim, t_layer, t_head, dropout, dim=1)

    def forward(self, x, pt=None, ps=None):
        pt = x if pt is None else pt
        ps = x if ps is None else ps
        x = self.t_attn(x, pt)
        x = self.s_attn(x, ps)
        return x

class STB_Transformer(nn.Module):
    def __init__(self, emb_dim, s_layer=1, t_layer=2, ib_num=5, s_head=2, t_head=2, dropout=0.1):
        super().__init__()

        self.s_attn = nn.ModuleList([Bottleneck_Attention(emb_dim, ib_num, s_head, dropout) for i in range(s_layer)])
        self.t_attn = Encoder(emb_dim, t_layer, t_head, dropout, dim=1)

    def forward(self, x):
        x = self.t_attn(x)
        for l in self.s_attn:
            x = l(x)
        return x

class Transformer_emb(nn.Module):
    def __init__(self, fea_dim, emb_dim):
        super().__init__()

        self.lin = nn.Linear(fea_dim, emb_dim)
        self.pe = PositionalEmbedding(emb_dim, 12)

    def forward(self, x):
        x = self.lin(x)
        pe = self.pe(x)
        return x + pe.unsqueeze(-2)

class Transformer_emb_v2(nn.Module):
    def __init__(self, win_in, fea_dim, emb_dim):
        super().__init__()

        self.lin = nn.Linear(win_in * fea_dim, emb_dim)

    def forward(self, x):
        x = x.transpose(1, 2).flatten(2)
        x = self.lin(x)
        return x

class Transformer_pred_v2(nn.Module):
    def __init__(self, win_out, fea_dim, emb_dim):
        super().__init__()

        self.lin = nn.Linear(emb_dim, win_out * fea_dim)
        self.t = win_out

    def forward(self, x):
        x = self.lin(x).unflatten(-1, (self.t, -1)).transpose(1, 2)
        return x

class ST_Embedding(nn.Module):
    def __init__(self, fea_dim, emb_dim, num_nodes, num_times):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.emb = nn.Linear(fea_dim, emb_dim)
        self.pe = PositionalEmbedding(emb_dim, 12)
        self.emb_spa = nn.Parameter(torch.empty(num_nodes, emb_dim))
        self.emb_tod = nn.Embedding(num_times, emb_dim)
        self.emb_dow = nn.Embedding(7, emb_dim)
        nn.init.xavier_uniform_(self.emb_spa)

    def forward(self, x):
        x, p = x
        emb = self.emb(x)
        emb = emb + self.pe(emb).unsqueeze(-2)
        emb = emb + self.emb_tod(p[..., [0]]) + self.emb_dow(p[..., [-1]])
        if hasattr(self, 'new_emb'):
            emb = emb + torch.cat((self.emb_spa, self.new_emb), 0)
        else:
            emb = emb + self.emb_spa
        return emb

    def prepare_for_new_task(self, graph, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            if hasattr(self, 'new_emb'):
                self.emb_spa = nn.Parameter(torch.cat([self.emb_spa, self.new_emb], dim=0))
            self.new_emb = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.emb_dim, dtype=self.emb_spa.dtype, device=self.emb_spa.device))
            nn.init.xavier_uniform_(self.new_emb)
            self.num_nodes = new_num_nodes

class GTG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gcn1 = BatchGCNConv(config.gcn['in_channel'], config.gcn['hidden_channel'], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(config.gcn['hidden_channel'], config.gcn['out_channel'], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=config.kernel_size, dilation=config.dilation,
                              padding=int((config.kernel_size - 1) * config.dilation / 2))
        self.activation = nn.GELU()

    def forward(self, x, adj):
        #B,N,T->B,N,H
        B, N, D = x.shape
        x_ = x

        x = F.relu(self.gcn1(x, adj))  # [bs, N, feature]
        x = x.reshape((B * N, 1, -1))  # [bs * N, 1, feature]

        x = self.tcn1(x)  # [bs * N, 1, feature]

        x = x.reshape((B, N, -1))  # [bs, N, feature]
        x = self.gcn2(x, adj)  # [bs, N, feature]

        x = x + x_
        return x

class GTG_predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout
        self.win_out = config.output_window
        self.dim_out = config.dim_out
        self.fc = nn.Linear(config.gcn["out_channel"], self.win_out * self.dim_out)
        self.activation = nn.GELU()

    def forward(self, x):
        #B,N,D->B,H,N,D
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.unflatten(-1, (self.win_out, self.dim_out)).transpose(1, 2)
        return x

class BatchGCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super(BatchGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x, adj):
        # x: [bs, N, in_features], adj: [N, N]
        input_x = torch.matmul(adj, x)  # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        output = self.weight_neigh(
            input_x)  # [bs, N, in_features] * [in_features, out_features] = [bs, N, out_features]
        if self.weight_self is not None:
            output += self.weight_self(x)  # [bs, N, out_features]
        return output

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=False, aptinit=None,
                 in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,
                 kernel_size=2,blocks=4,layers=2, graph=None):
        super(gwnet, self).__init__()

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        # self.gcn_bool = False
        # self.addaptadj = False

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        if supports is None and graph is not None:
            supports = self.get_support(graph)
        self.supports = supports

        receptive_field = 1

        self.supports_len = 2
        if supports is not None:
            self.supports_len = len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.receptive_field = receptive_field

    def forward(self, input, graph=None):
        #input B, D, N, T -> B, T, N, D
        #output B, H, D, 1 -> B, N, D
        if graph is not None and (self.supports is None or self.supports[0].shape != graph.shape):
            self.supports = self.get_support(graph)
        input = input.transpose(1, 3)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.T)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # new_supports = [F.softmax(F.relu(torch.einsum('bdnt,bdmt->btnm', x, x)), dim=1)]
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                raise 'no graph'
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        return skip.squeeze().transpose(1, 2)

    def get_support(self, g):
        def asym_adj_torch(adj):
            adj = adj.float()
            rowsum = torch.sum(adj, dim=1, keepdim=True)
            d_inv = torch.where(rowsum == 0, torch.tensor(0.0, device=adj.device), 1.0 / rowsum)
            normalized_adj = d_inv * adj
            return normalized_adj
        support = [asym_adj_torch(g), asym_adj_torch(g.T)]
        return support

class gwnet_predictor(nn.Module):
    def __init__(self, dim1=256, dim2=512, dim3=12, win_out=12):
        super().__init__()

        self.lin1 = nn.Linear(dim1, dim2)
        self.lin2 = nn.Linear(dim2, dim3)
        self.win_out = win_out


    def forward(self, x):
        x = self.lin1(F.relu(x))
        x = self.lin2(F.relu(x))
        x = x.unflatten(-1, (self.win_out, -1)).transpose(1, 2)
        return x

class RevIN(nn.Module):
    def __init__(self, layer, mode='emb', eps=1e-5, state=None):
        super().__init__()
        self.layer = layer
        self.mode = mode
        self.eps = eps
        self.state = {} if state is None else state

    def forward(self, x):
        if self.mode in ['emb', 'norm']:
            if isinstance(x, tuple):
                x = x[0]
            mean = x.mean(1, keepdim=True).detach()
            x = x - mean
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
            self.state['mean'] = mean
            self.state['stdev'] = stdev
            return self.layer(x / stdev)

        if self.mode in ['pred', 'denorm']:
            x = self.layer(x)
            mean = self.state.get('mean')
            stdev = self.state.get('stdev')
            if mean is None or stdev is None:
                return x
            return x * stdev + mean
        raise ValueError(f"Unknown RevIN mode: {self.mode}")
