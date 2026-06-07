import torch
from model import Model_cl
from model.model_cl.base_model import Backbone

class Base_CL(Model_cl):
    def __init__(self, config, data=None, cl_strategy=None):
        config.load_yaml_config('naive', config.dataset_name, 4)
        super().__init__(config)
        self.data = data
        self.model = Backbone(config.backbone, config)
        if cl_strategy is not None:
            config.cl_strategy = cl_strategy
        self.strategy = config.cl_strategy
        self.config = config
        self.graph = self.get_graph(config.start_year)

    def prepare_for_new_task(self, **kwargs):
        self.graph = self.get_graph(self.year)
        if self.strategy == 'Retrain':
            self.model = Backbone(self.config.backbone, self.config).to(self.device)
        if self.config.backbone in ['ste']:
            self.model.emb.prepare_for_new_task(self.graph, self.config.num_nodes_year[self.year])
        elif self.strategy == 'NewNode':
            self.subset = list()
            cur_node_size = self.config.num_nodes_year[self.year]
            pre_node_size = self.config.num_nodes_year[self.year - 1]
            self.subset.extend(list(range(pre_node_size, cur_node_size)))
        elif self.strategy == 'AllNode':
            pass

class Retrain(Base_CL):
    def __init__(self, config, data=None):
        super().__init__(config, data, 'Retrain')

class Pretrain(Base_CL):
    def __init__(self, config, data=None):
        super().__init__(config, data, 'Pretrain')

class NewNode(Base_CL):
    def __init__(self, config, data=None):
        super().__init__(config, data, 'NewNode')

    def get_graph(self, year):
        if self.config.need_graph:
            g = torch.tensor(self.data.load_graph(year), dtype=self.dtype).to(self.device)
            graph = g / (g.sum(1, keepdims=True) + 1e-6)
            self.cur_graph = graph
            if year > self.config.start_year:
                self.cur_graph = graph[self.config.num_nodes_year[year - 1]: , self.config.num_nodes_year[year - 1]: ]
            return graph

    def forward(self, *data):
        x = self.norm.norm(data[0])
        if self.config.need_time:
            x = (x, data[1])
        if self.graph is not None:
            output = self.model(x, data[-1])
        else:
            output = self.model(x)
        output = self.norm.inorm(output)
        return output

    def calculate_loss(self, data):
        x, y = self.get_data(data)
        output = self.forward(*x, self.cur_graph)
        loss = self.loss(output, y)
        return loss

    def predict(self, data):
        x, y = self.get_data(data)
        output = self.forward(*x, self.graph)
        return output, y

    # def forward(self, *data):
    #     x = self.norm.norm(data[0])
    #     if self.config.need_graph:
    #         output = self.model(x, self.cur_graph)
    #     else:
    #         output = self.model(x)
    #     output = self.norm.inorm(output)
    #     return output

class AllNode(Base_CL):
    def __init__(self, config, data=None):
        super().__init__(config, data, 'AllNode')