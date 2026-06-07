import torch
from model import Model_base

class Model_cl(Model_base):
    def __init__(self, config=None, data=None):
        super().__init__(config)
        self.year = config.start_year
        self.data = data
        self.graph = self.get_graph(config.start_year) if config is not None and data is not None else None

    # for Continual Learning Models
    # the model should update its state upon new dataset comes
    # ie, expand_adaptive_params for EAC model
    def prepare_for_new_task(self, **kwargs):
        pass

    def get_graph(self, year):
        if self.config.need_graph:
            g = torch.tensor(self.data.load_graph(year), dtype=self.dtype, device=self.device)
            return g / (g.sum(1, keepdim=True) + 1e-6)
        else:
            return None

    # def get_data(self, data):
    #     x, y = data
    #     x, y = x.cuda().to(torch.float32), y.cuda().to(torch.float32)
    #     return (x, self.graph), y

    def forward(self, *data):
        x = self.norm.norm(data[0])
        if self.config.need_time:
            x = (x, data[-1])
        if self.graph is not None:
            output = self.model(x, self.graph)
        else:
            output = self.model(x)
        output = self.norm.inorm(output)
        return output

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_ewc_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params(self, loader, lossfunc, graph=None):
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        est_fisher_info = {name: 0.0 for name in _buff_param_names}
        for i, batch in enumerate(loader):
            x, y = batch
            x, y = x.cuda().to(torch.float32), y.cuda().to(torch.float32)
            pred = self.model.forward(x, graph)
            log_likelihood = lossfunc(y, pred)
            grad_log_liklihood = torch.autograd.grad(log_likelihood, self.model.parameters(), allow_unused=True)
            for name, grad in zip(_buff_param_names, grad_log_liklihood):
                if grad is not None:
                    est_fisher_info[name] += grad.data.clone() ** 2
        for name in _buff_param_names:
            if est_fisher_info[name] is not None:
                self.register_ewc_buffer(name + '_estimated_fisher', est_fisher_info[name])

    def register_ewc_params(self, loader, lossfunc, graph=None):
        self._update_fisher_params(loader, lossfunc, graph)
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
        return 1 * (self.config.ewc_lambda / 2) * sum(losses)

    def register_ewc_buffer(self, name, value):
        self.ewc[name] = value