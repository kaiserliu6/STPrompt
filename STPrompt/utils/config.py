import os.path
import sys
from logging import getLogger
import yaml
import json

"""
当前config的优先级和读取顺序为（越大表明优先级越高，相同优先级按照列表顺序读取不覆盖）：
max: config_dict中的设置，通常由main传入，主要包括任务名和评估设定
9: hyper_config中的设置，通常为参数测试时设置
3: config_file_list中的文件的设置（暂时废弃）
3: （推荐）yaml文件设置，默认文件名为{model_name}.yaml，根据dataset_name读取
3: json文件设置，默认文件名为{model_name}_{dataset_name}.json，必须为每个数据集指定单独设置文件
max: 读取命令行参数
1: 默认设置

7: 数据集文件中的设置
5: 模型文件中的设置
"""

class Config():
    def __init__(self, model_name=None, dataset_name=None, config_dict=None, config_file_list=None, hyper_config=None):
        super().__setattr__('config', {})
        super().__setattr__('config_prior', {})
        if config_dict is not None:
            self.config.update(config_dict)
        self.update(hyper_config, 9)
        self._load_config_files(model_name, dataset_name)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.update(self._load_cmd_line(), 9)
        for key in self.config:
            self.config[key] = self._para_format(self.config[key])
            # self.config_prior[key] = 3
        self.default_conf()


    def update(self, dict, prior=3):
        if dict is None:
            return
        for key in dict:
            if key in self.config and self.config_prior[key] > prior:
                pass
            else:
                self.config[key] = self._para_format(dict[key])
                self.config_prior[key] = prior

    def load_yaml_config(self, modelname, dataname, prior=5):
        file_dir = f'config/{modelname.lower()}.yaml'
        if os.path.exists(file_dir):
            with open(file_dir, 'r', encoding='utf-8') as f:
                conf = yaml.safe_load(f.read())
                config = conf['default']
                if dataname in conf:
                    config.update(conf[dataname])
                if 'backbone' in self.config:
                    backbone = self.config['backbone']
                elif 'backbone' in config:
                    backbone = config['backbone']
                else:
                    backbone = None
                if backbone in config:
                    config.update(config[backbone])
            self.update(config, prior)

    def load_json_config(self, modelname, dataname, prior=5):
        file_dir = f'config/{modelname.lower()}_{dataname.lower()}'
        if os.path.exists(file_dir + '.json'):
            with open(file_dir + '.json', 'r', encoding='utf-8') as f:
                conf = json.load(f)
            self.update(conf, prior)
        elif os.path.exists(file_dir + '.conf'):
            with open(file_dir + '.conf', 'r', encoding='utf-8') as f:
                conf = json.load(f)
            self.update(conf, prior)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.update({key:value})

    def __call__(self, key, value=None):
        if value:
            self.config[key] = value
        else:
            return self.config[key]

    def __getattr__(self, item):
        if item in self.config:
            return self.config[item]
        else:
            return None

    def __setattr__(self, key, value):
        self.update({key:value}, 9)

    def contain(self, key):
        return key in self.config

    def get(self, key, item=None):
        return self.config.get(key, item)

    def _para_format(self, para):
        if not isinstance(para, str):
            return para
        try:
            value = eval(para)
            if value is not None and not isinstance(
                    value, (str, int, float, list, tuple, dict, bool)
            ):
                value = para
        except (NameError, SyntaxError, TypeError):
            if isinstance(para, str):
                if para.lower() == "true":
                    value = True
                elif para.lower() == "false":
                    value = False
                else:
                    value = para
            else:
                value = para
        return value

    def _load_config_files(self, model_name, dataset_name):
        self.load_yaml_config(model_name, dataset_name, prior=3)
        self.load_json_config(model_name, dataset_name, prior=3)

    def _load_cmd_line(self):
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if (
                    cmd_arg_name in cmd_config_dict
                    and cmd_arg_value != cmd_config_dict[cmd_arg_name]
                ):
                    raise SyntaxError(
                        "There are duplicate commend arg '%s' with different value."
                        % arg
                    )
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning(
                "command line args [{}] will not be used".format(
                    " ".join(unrecognized_args)
                )
            )
        # cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def default_conf(self):
        tr_conf = {
            'epoch': 100,
            'clip_grad_norm': True,
            'max_grad_norm': 5.,
            'log_step': 1,
            'early_stop': 50,
            # 'metric_list': ['masked_MAE', 'masked_RMSE', 'masked_MAPE'],
            'metric_list': ['MAE', 'RMSE', 'MAPE'],
            'metric_len': 12,
            'metirc_threshold': 0, # when given a list, evaluate all values in list; evaluate truth greater (not greater or equal) than the value given; when given None, won't apply
            'short_valid': True,
            'task_name': 'ODFP',
            'trainer_type': 'base',
            'val_use_loss': True,
            'val_step': 1,
            'save_result':False,
            'save_dir':'cache/result/',
            'dual_evaluator':True,
            'eval_mode': 'average'
        }
        self.update(tr_conf, 1)
        lr_conf = {
            'optimizer': 'adam',
            'scheduler': 'none',
            'learning_rate': 0.01,
        }
        self.update(lr_conf, 1)
        ds_config = {
            'data_path': 'data/',
            'input_window': 12,
            'output_window': 12,
            'train_size': 0.6,
            'test_size': 0.2,
            'scaler_mode': 'standard',
            'need_data': False,
            'need_od': False,
            # 'pre_norm': 'x'
        }
        self.update(ds_config, 1)
        md_conf = {
            'batch_size': 1024,
            'use_weather': False,
            'use_weekday': False,
            'need_time': False,
            'num_heads': 8,
            'dropout': 0.1,
            'pre_norm': 'x_only' #'x_only' or 'all'
        }
        self.update(md_conf, 1)

    def optimizer_kwargs(self):
        opt_name = self.config.get('optimizer', 'adam').lower()
        param_map = {
            'lr': 'learning_rate',
            'weight_decay': 'weight_decay',
            'eps': 'lr_epsilon',
            'betas': 'lr_betas',
            'momentum': 'lr_momentum',
            'alpha': 'lr_alpha'
        }
        whitelist = {
            'adam': {'lr', 'weight_decay', 'eps', 'betas'},
            'adamw': {'lr', 'weight_decay', 'eps', 'betas'},
            'sgd': {'lr', 'weight_decay', 'momentum'},
            'adagrad': {'lr', 'weight_decay', 'eps'},
            'rmsprop': {'lr', 'weight_decay', 'momentum', 'alpha', 'eps'},
            'sparse_adam': {'lr', 'eps', 'betas'}
        }
        valid_keys = whitelist.get(opt_name, whitelist['adam'])
        v = lambda x:self.config.get(param_map[x])
        para_dict = {k: v(k) for k in valid_keys if v(k) is not None}
        return para_dict

    def scheduler_kwargs(self):
        name = self.config.get('scheduler', 'none').lower()
        param_map = {
            'milestones': 'milestones',
            'gamma': 'lr_decay_ratio',
            'step_size': 'step_size',
            'T_max': 'lr_T_max',
            'eta_min': 'lr_eta_min',
            'lr_lambda': 'lr_lambda',
            'patience': 'lr_patience',
            'factor': 'lr_decay_ratio',
            'threshold': 'lr_threshold'
        }
        whitelist = {
            'multisteplr': {'milestones', 'gamma'},
            'steplr': {'step_size', 'gamma'},
            'exponentiallr': {'gamma'},
            'cosineannealinglr': {'T_max', 'eta_min'},
            'lambdalr': {'lr_lambda'},
            'reducelronplateau': {'patience', 'factor', 'threshold'}
        }
        valid_keys = whitelist.get(name, set())
        v = lambda x: self.config.get(param_map[x])
        para_dict = {k: v(k) for k in valid_keys if v(k) is not None}
        return para_dict

if __name__ == '__main__':
    config = Config(config_dict={'A':1,'B':'abc','C':'true'})
    print(config.A,config['B'],config('C'))
    config.A = 'true'
    config['C'] = 'FALSE'
    print(config.A, config['B'], config('C'), config.D)