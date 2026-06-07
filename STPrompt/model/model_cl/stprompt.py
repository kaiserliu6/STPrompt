import torch
import torch.nn as nn

from model import Model_cl
from model.model_cl.base_model import Backbone
from block import LoRA_attn

class Model(Model_cl):
    def __init__(self, config, data):
        super().__init__(config)
        config.input_dim = self.dim_in
        config.output_dim = self.dim_out
        self.data = data
        self.graph = self.get_graph(config.start_year)
        self.model = ST_Prompt(config, self.graph)

    def prepare_for_new_task(self, **kwargs):
        if self.config.lora:
            if self.year == self.config.start_year + 1:
                for i in range(self.config.s_layer):
                    self.model.model.model.s_attn.attn[i].attn = LoRA_attn(self.model.model.model.s_attn.attn[i].attn, 4, 4)
            elif self.year > self.config.start_year:
                for i in range(self.config.s_layer):
                    self.model.model.model.s_attn.attn[i].attn.merge_lora_parameters()
        if self.config.cl_strategy == 'Retrain':
            self.config.num_nodes = self.config.num_nodes_year[self.year]
            self.model = ST_Prompt(self.config).to(self.device)
            return
        self.graph = self.get_graph(self.year)
        self.model.prepare_for_new_task(self.config.num_nodes_year[self.year])
        for name, para in self.model.named_parameters():
            if not 'prompt' in name and not 'gate' in name:
                para.requires_grad = False

    def forward(self, *data):
        x = self.norm.norm(data[0])
        output = self.model(x, data[1], None) if self.config.backbone in ['stf', 'stf2', 'stf3', 'stf4', 'itf'] else self.model(x, data[1], data[-1])
        output = self.norm.inorm(output)
        return output

    def calculate_loss(self, data):
        x, y = self.get_data(data)
        output = self.forward(*x)
        loss = self.loss(output, y)
        return loss

    def get_data(self, data):
        x, y, xe, ye = data
        x, y, xe = x.cuda().to(torch.float32), y.cuda().to(torch.float32), xe.cuda().to(torch.int)
        return (x, xe, self.graph), y

class ST_Prompt(nn.Module):
    def __init__(self, config, graph=None):
        super().__init__()
        self.config = config
        self.model = Backbone(config.backbone, config, graph)
        self.num_nodes = config.num_nodes
        if config.get('joint_traing', False):
            self.prompt = self._build_prompt(self.num_nodes)
            self.prompt.prepare_for_new_task(self.num_nodes)
        else:
            self.prompt = lambda x: 0

    def _build_prompt(self, num_nodes):
        return ST_Embedding(
            self.config.tod_mem, self.config.spa_mem, self.config.emb_dim, self.config.num_times,
            num_nodes
        )

    def forward(self, x, p, g=None):
        st_emb = self.prompt(p)
        x_emb = self.model.emb(x)
        if x_emb.ndim == 3 and not isinstance(st_emb, int) and st_emb.ndim == 4:
            st_emb = st_emb.mean(1)
        x = self.model.model(x_emb + st_emb) if g is None else self.model.model(x_emb + st_emb, g)
        x = self.model.pred(x)
        return x
        
    def prepare_for_new_task(self, new_num_nodes):
        if isinstance(self.prompt, ST_Embedding):
            self.prompt.prepare_for_new_task(new_num_nodes)
            self.num_nodes = new_num_nodes
        else:
            self.num_nodes = new_num_nodes
            self.prompt = self._build_prompt(new_num_nodes).cuda()
            self.prompt.prepare_for_new_task(new_num_nodes)

class ST_Embedding(nn.Module):
    def __init__(self, tod_mem, spa_mem, mem_dim, num_times, num_nodes):
        super().__init__()
        self.spa_prompt = Spatial_Prompt(spa_mem, mem_dim, num_nodes)
        self.tem_prompt = Temporal_Prompt(tod_mem, mem_dim, num_times)

    def forward(self, p):
        t_out = self.tem_prompt(p)
        if t_out.ndim == 3:
            t_out = t_out.unsqueeze(2)
        s_out = self.spa_prompt()
        # return t_out
        # return s_out
        return t_out + s_out

    def prepare_for_new_task(self, new_num_nodes):
        self.tem_prompt.prepare_for_new_task(new_num_nodes=new_num_nodes)
        self.spa_prompt.prepare_for_new_task(new_num_nodes)

class Spatial_Prompt(nn.Module):
    def __init__(self, spa_mem, mem_dim, num_nodes, graph=None):
        super().__init__()
        self.spa_mem = spa_mem
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.mem_spa = nn.Parameter(torch.empty(spa_mem, mem_dim))
        self.emb_spa = nn.Parameter(torch.empty(num_nodes, spa_mem))
        nn.init.zeros_(self.mem_spa)
        nn.init.xavier_uniform_(self.emb_spa)

    def forward(self):
        return self.emb_spa @ self.mem_spa

    def prepare_for_new_task(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.spa_mem,
                                                  dtype=self.emb_spa.dtype,
                                                  device=self.emb_spa.device).uniform_(-0.1, 0.1))
            self.emb_spa = nn.Parameter(torch.cat([self.emb_spa, new_params], dim=0))
            self.num_nodes = new_num_nodes

class Temporal_Prompt(nn.Module):
    def __init__(self, tod_mem, mem_dim, num_times):
        super().__init__()
        self.mem_tod = nn.Parameter(torch.empty(tod_mem, mem_dim))
        self.emb_tod = nn.Embedding(num_times, tod_mem)
        self.emb_dow = nn.Embedding(7, mem_dim)
        nn.init.xavier_uniform_(self.mem_tod)
        nn.init.zeros_(self.emb_tod.weight)

    def forward(self, p):
        return self.emb_tod(p[..., 0]) @ self.mem_tod + self.emb_dow(p[..., -1])

    def prepare_for_new_task(self, *args, **kwargs):
        pass
