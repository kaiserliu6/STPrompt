import torch
import torch.nn as nn

from block import LoRA_attn
from model import Model_cl
from model.model_cl.base_model import Backbone


class Model(Model_cl):
    def __init__(self, config, data):
        super().__init__(config)
        config.input_dim = self.dim_in
        config.output_dim = self.dim_out
        self.data = data
        self.graph = self.get_graph(config.start_year)
        self.model = STpromptModel(config, self.graph)

    def prepare_for_new_task(self, **kwargs):
        if self.config.cl_strategy == 'Retrain':
            self.config.num_nodes = self.config.num_nodes_year[self.year]
            self.model = STpromptModel(self.config).to(self.device)
            return
        self.apply_lora()
        self.graph = self.get_graph(self.year)
        self.model.prepare_for_new_task(self.config.num_nodes_year[self.year], self.graph)
        for name, param in self.model.named_parameters():
            param.requires_grad = 'prompt' in name

    def forward(self, *data):
        x = self.norm.norm(data[0])
        if self.config.backbone in ['stf', 'stf2', 'stf3', 'stf4', 'itf', 'stf5']:
            output = self.model(x, data[1])
        else:
            output = self.model(x, data[1], data[-1])
        return self.norm.inorm(output)

    def calculate_loss(self, data):
        x, y = self.get_data(data)
        return self.loss(self.forward(*x), y)

    def get_data(self, data):
        x, y, xe, ye = data
        x = x.cuda().to(torch.float32)
        y = y.cuda().to(torch.float32)
        xe = xe.cuda().to(torch.int)
        return (x, xe, self.graph), y

    def apply_lora(self):
        if not self.config.get('lora', False):
            return
        if self.year == self.config.start_year + 1:
            for i in range(self.config.s_layer):
                self.model.model.model.s_attn.attn[i].attn = LoRA_attn(
                    self.model.model.model.s_attn.attn[i].attn, 4, 4
                )
        elif self.year > self.config.start_year:
            for i in range(self.config.s_layer):
                self.model.model.model.s_attn.attn[i].attn.merge_lora_parameters()


class STpromptModel(nn.Module):
    def __init__(self, config, graph=None):
        super().__init__()
        self.config = config
        self.model = Backbone(config.backbone, config, graph)
        self.num_nodes = config.num_nodes
        self.prompt = None
        if config.get('joint_training', False):
            self.prompt = self._build_prompt(self.num_nodes, graph)

    def _build_prompt(self, num_nodes, graph):
        return ST_prompt(
            self.config.tod_mem,
            self.config.spa_mem,
            self.config.emb_dim,
            self.config.num_times,
            num_nodes,
            graph,
            self.config,
        )

    def forward(self, x, p, graph=None):
        x = self.model.emb(x)
        if self.prompt is None:
            x = self.model.model(x) if graph is None else self.model.model(x, graph)
        else:
            x = self.prompt(x, p, self.model.model, graph)
        return self.model.pred(x)

    def prepare_for_new_task(self, new_num_nodes, graph):
        self.num_nodes = new_num_nodes
        if self.prompt is None:
            reference_param = next(self.model.parameters())
            self.prompt = self._build_prompt(new_num_nodes, graph).to(
                device=reference_param.device,
                dtype=reference_param.dtype,
            )
            return
        self.prompt.prepare_for_new_task(graph, new_num_nodes)


class ST_prompt(nn.Module):
    def __init__(self, tod_mem, spa_mem, mem_dim, num_times, num_nodes, graph, config):
        super().__init__()
        out_dim = config.get('out_dim', mem_dim)
        prompt_dim = mem_dim * 2 + out_dim
        self.dims = [mem_dim, out_dim, mem_dim]
        self.node_prompt = config.get('node_prompt', True)
        self.use_tem_prompt = config.get('tem_prompt', True)
        self.prompt_prompt = config.get('prompt_prompt', False)
        if self.prompt_prompt:
            self.spa_prompt = SpatialPrompt(spa_mem, prompt_dim, num_nodes)
            if self.use_tem_prompt:
                self.tem_prompt = TemporalPrompt(
                    tod_mem,
                    prompt_dim,
                    num_times,
                    config.get('tod_slot', num_times),
                    config.get('override', False),
                )
        else:
            self.spa_prompt = SpatialEmbedding(prompt_dim, num_nodes)
            if self.use_tem_prompt:
                self.tem_prompt = TemporalEmbedding(
                    prompt_dim,
                    num_times,
                    config.get('tod_slot', num_times),
                    config.get('override', False),
                )

    def forward(self, x, p, model, graph=None):
        prompt_params = self.spa_prompt()
        if self.use_tem_prompt:
            temporal_params = self.tem_prompt(p)
            if temporal_params.ndim == 3:
                temporal_params = temporal_params.unsqueeze(2)
            prompt_params = prompt_params + temporal_params
        if x.ndim == 3 and prompt_params.ndim == 4:
            prompt_params = prompt_params.mean(1)
        st_emb, gate, scale = torch.split(prompt_params, self.dims, dim=-1)
        x_ = x
        x = x + st_emb
        if not self.node_prompt:
            return model(x) if graph is None else model(x, graph)
        scaled_x = x * (1 + scale)
        # model_output = model(scaled_x) if graph is None else model(scaled_x, graph)
        # return (1+gate) * model_output
        model_x = model(scaled_x) if graph is None else model(scaled_x, graph)
        if model_x.shape == scaled_x.shape:
            model_output = x + (1 + gate) * (model_x - scaled_x)
        else:
            if gate.ndim == 4 and model_x.ndim == 3:
                gate = gate.mean(1)
            model_output = (1 + gate) * model_x
        return model_output

    def prepare_for_new_task(self, graph, new_num_nodes):
        if self.use_tem_prompt:
            self.tem_prompt.prepare_for_new_task(new_num_nodes=new_num_nodes)
        self.spa_prompt.prepare_for_new_task(graph, new_num_nodes)

class SpatialEmbedding(nn.Module):
    def __init__(self, emb_dim, num_nodes):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.emb_spa = nn.Parameter(torch.empty(num_nodes, emb_dim))
        nn.init.xavier_uniform_(self.emb_spa)

    def forward(self):
        if hasattr(self, 'new_emb'):
            return torch.cat((self.emb_spa, self.new_emb), dim=0)
        return self.emb_spa

    def prepare_for_new_task(self, graph, new_num_nodes):
        if new_num_nodes <= self.num_nodes:
            return
        if hasattr(self, 'new_emb'):
            self.emb_spa = nn.Parameter(torch.cat([self.emb_spa, self.new_emb], dim=0))
        self.new_emb = nn.Parameter(
            torch.empty(
                new_num_nodes - self.num_nodes,
                self.emb_dim,
                dtype=self.emb_spa.dtype,
                device=self.emb_spa.device,
            )
        )
        nn.init.xavier_uniform_(self.new_emb)
        self.num_nodes = new_num_nodes


class SpatialPrompt(nn.Module):
    def __init__(self, spa_mem, mem_dim, num_nodes):
        super().__init__()
        self.spa_mem = spa_mem
        self.num_nodes = num_nodes
        self.mem_spa = nn.Parameter(torch.empty(spa_mem, mem_dim))
        self.emb_spa = nn.Parameter(torch.empty(num_nodes, spa_mem))
        nn.init.zeros_(self.mem_spa)
        nn.init.xavier_uniform_(self.emb_spa)
        # nn.init.uniform_(self.emb_spa, -0.1, 0.1)

    def forward(self):
        return self.emb_spa @ self.mem_spa

    def prepare_for_new_task(self, graph, new_num_nodes):
        if new_num_nodes <= self.num_nodes:
            return
        new_params = torch.empty(
            new_num_nodes - self.num_nodes,
            self.spa_mem,
            dtype=self.emb_spa.dtype,
            device=self.emb_spa.device,
        ).uniform_(-0.1, 0.1)
        self.emb_spa = nn.Parameter(torch.cat([self.emb_spa, new_params], dim=0))
        self.num_nodes = new_num_nodes


def _resolve_tod_slot(num_times, tod_slot):
    tod_slot = int(tod_slot or num_times)
    if tod_slot <= 0:
        raise ValueError(f'tod_slot must be positive, got {tod_slot}')
    if num_times % tod_slot != 0:
        raise ValueError(
            f'tod_slot must divide num_times, got tod_slot={tod_slot}, num_times={num_times}'
        )
    return tod_slot, num_times // tod_slot


def _tod_index(p, num_times, group_size, offset=0):
    index = p[..., 0].long()
    if offset:
        index = (index + offset) % num_times
    return index // group_size


class TemporalEmbedding(nn.Module):
    def __init__(self, emb_dim, num_times, tod_slot=None, override=False):
        super().__init__()
        self.num_times = num_times
        self.tod_slot, self.group_size = _resolve_tod_slot(num_times, tod_slot)
        self.override = override
        self.emb_tod = nn.Embedding(self.tod_slot, emb_dim)
        if self.override:
            self.emb_tod2 = nn.Embedding(self.tod_slot, emb_dim)
        self.emb_dow = nn.Embedding(7, emb_dim)

    def forward(self, p):
        output = self.emb_tod(_tod_index(p, self.num_times, self.group_size))
        output = output + self.emb_dow(p[..., -1].long())
        if self.override:
            output = output + self.emb_tod2(
                _tod_index(p, self.num_times, self.group_size, self.group_size // 2)
            )
        return output

    def prepare_for_new_task(self, *args, **kwargs):
        pass


class TemporalPrompt(nn.Module):
    def __init__(self, tod_mem, mem_dim, num_times, tod_slot=None, override=False):
        super().__init__()
        self.mem_tem = nn.Parameter(torch.empty(tod_mem, mem_dim))
        self.num_times = num_times
        self.tod_slot, self.group_size = _resolve_tod_slot(num_times, tod_slot)
        self.override = override
        self.emb_tod = nn.Embedding(self.tod_slot, tod_mem)
        if self.override:
            self.emb_tod2 = nn.Embedding(self.tod_slot, tod_mem)
        self.emb_dow = nn.Embedding(7, tod_mem)
        nn.init.xavier_uniform_(self.mem_tem)
        # nn.init.uniform_(self.mem_tem, -0.1, 0.1)
        nn.init.zeros_(self.emb_tod.weight)
        if self.override:
            nn.init.zeros_(self.emb_tod2.weight)
        nn.init.zeros_(self.emb_dow.weight)

    def forward(self, p):
        embedding = self.emb_tod(_tod_index(p, self.num_times, self.group_size))
        embedding = embedding + self.emb_dow(p[..., -1].long())
        if self.override:
            embedding = embedding + self.emb_tod2(
                _tod_index(p, self.num_times, self.group_size, self.group_size // 2)
            )
        return embedding @ self.mem_tem

    def prepare_for_new_task(self, *args, **kwargs):
        pass