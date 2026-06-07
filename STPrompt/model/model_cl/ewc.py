import torch
import torch.nn as nn
from torch import autograd

from model import Model_cl
from model.model_cl.base_model import Backbone


class EWC():
    def __init__(self, model, ewc_lambda = 0, ewc_type = 'ewc'):
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.ewc_type = ewc_type

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params(self, loader, lossfunc, graph=None, device='cuda:0'):
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        est_fisher_info = {name: 0.0 for name in _buff_param_names}
        for i, batch in enumerate(loader):
            x, y = batch
            x, y = x.cuda().to(torch.float32), y.cuda().to(torch.float32)
            pred = self.model.forward(x, graph)
            log_likelihood = lossfunc(y, pred)
            grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters(), allow_unused=True)
            for name, grad in zip(_buff_param_names, grad_log_liklihood):
                if grad is not None:
                    est_fisher_info[name] += grad.data.clone() ** 2
        for name in _buff_param_names:
            if est_fisher_info[name] is not None:
                self.register_buffer(name + '_estimated_fisher', est_fisher_info[name])

    def register_ewc_params(self, loader, lossfunc, graph=None, device='cuda:0'):
        self._update_fisher_params(loader, lossfunc, graph, device)
        self._update_mean_params()

    def compute_consolidation_loss(self):
        losses = []
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            if not hasattr(self, '{}_estimated_fisher'.format(_buff_param_name)):
                continue
            estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
            if estimated_fisher == None:
                losses.append(0)
            elif self.ewc_type == 'l2':
                losses.append((10e-6 * (param - estimated_mean) ** 2).sum())
            else:
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
        return 1 * (self.ewc_lambda / 2) * sum(losses)

    def forward(self, data, adj): 
        return self.model(data, adj)

    def register_buffer(self, name, value):
        setattr(self, name, value)

class EWC_Model(Model_cl):
    def __init__(self, config, data=None):
        conf = {
            'batch_size': 64,
            'scheduler': 'steplr',
            'step_size': 20,
            'emb_dim': 64,
            's_layer': 1,
            't_layer': 2,
            's_head': 2,
            't_head': 2,
            'dropout': 0.1,
            'need_graph': False,
            'backbone': 'stf'
        }
        config.update(conf, 5)
        super().__init__(config)
        self.data = data
        self.model = Backbone(config.backbone, config)
        self.config = config
        self.get_graph(config.start_year)

    def prepare_for_new_task(self, **kwargs):
        if not isinstance(self.model, EWC):
            self.model = EWC(self.model)
        self.get_graph(self.year)