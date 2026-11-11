import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Model_cl
from model.model_cl.base_model import Backbone

class Model(Model_cl):
    def __init__(self, config, data):
        super().__init__(config)
        self.model = EAC_Model(config)
        self.data = data
        self.graph = self.get_graph(config.start_year)

    def prepare_for_new_task(self, **kwargs):
        self.model.flag = True
        for name, param in self.model.named_parameters():
            if "U_cls" in name or "V_cls" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.graph = self.get_graph(self.year)
        self.model.expand_adaptive_params(self.config.num_nodes_year[self.year])

    # def forward(self, *data):
    #     x = self.norm.norm(data[0])  # B, N, T
    #     output = self.model(x, data[-1])
    #     output = self.norm.inorm(output)
    #     return output

    def calculate_loss(self, data):
        x, y = self.get_data(data)
        output = self.forward(*x)
        loss = self.loss(output, y)
        return loss

class EAC_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = config.backbone
        self.model = Backbone(self.backbone, config)
        base_node_size = config.num_nodes
        self.rank = config.rank
        self.flag = True

        # Initialize subspace and adjust matrix
        if self.backbone == 'gtg':
            self.U_cls = nn.Parameter(torch.empty(base_node_size, self.rank).uniform_(-0.1, 0.1))
            self.V_cls = nn.Parameter(torch.empty(self.rank, config.gcn["in_channel"]).uniform_(-0.1, 0.1))
        else:
            self.U_cls = nn.Parameter(torch.empty(base_node_size, self.rank).uniform_(-0.1, 0.1))
            self.V_cls = nn.Parameter(torch.empty(self.rank, config.input_window * config.dim_in).uniform_(-0.1, 0.1))

        # self.year = args.year
        self.num_nodes = base_node_size

    def forward(self, x, adj=None):
        B, T, N, D = x.shape
        adaptive_params = torch.mm(self.U_cls, self.V_cls) # N, TD
        if self.flag:
            x = x + adaptive_params.unflatten(-1, (T, D)).transpose(0, 1)
        x = self.model(x, adj)
        return x

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U_cls.dtype,
                                                  device=self.U_cls.device).uniform_(-0.1, 0.1))
            self.U_cls = nn.Parameter(torch.cat([self.U_cls, new_params], dim=0))
            self.num_nodes = new_num_nodes