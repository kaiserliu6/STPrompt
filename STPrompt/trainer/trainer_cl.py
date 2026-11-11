import os
import torch
from datetime import datetime
import gc

from utils import ensure_dir
from trainer import Trainer

"""
This Trainer is designed fo Continual Learning
Only support TF task for now
"""

class Trainer_CL(Trainer):
    def __init__(self, model_name, dataset_name, config_file_list = None, config_dict=None, random_seed=None):
        super().__init__(model_name, dataset_name, config_file_list, config_dict, random_seed)
        self.start_year = self.config.start_year
        self.end_year = self.config.end_year
        self.cl_run_batch = []
        self.cl_train_time = []
        self.cl_infer_time = []
        self.cl_peak_mem = []
        self.saved_model_path = None

    # base function called to train the model
    def run(self, epochs=None):
        result = {}
        self.cur_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.saved_model_path = self.config.base_model_dir if isinstance(self.config.base_model_dir, str) else None
        self.cl_run_batch = []
        self.cl_train_time = []
        self.cl_infer_time = []
        self.cl_peak_mem = []
        for year in range(self.start_year, self.end_year + 1):
            self.run_batch = 0
            self.train_time = 0
            self.infer_time = 0
            self.peak_mem = 0
            self.model.year = year
            self.logger.info(f'Training on {year}')
            if year == self.start_year:
                if self.config.joint_training or self.saved_model_path is None:
                    self.cache_dir = f'cache/{self.dataset_name}/{self.model_name}_{year}_{self.cur_time}.temp'
                    self.load_data(year)
                    self.train()
                else:
                    self.cache_dir = self.saved_model_path
                    self.load_data(year, True)
                if self.config.retrain_first_year and self.config.peft and self.model_name != 'Pretrain':
                    self.load_model(self.saved_model_path)
                    self.model.prepare_for_new_task()
                    self.cache_dir = f'cache/{self.dataset_name}/{self.model_name}_{year}_{self.cur_time}.temp'
                    self.load_data(year)
                    self.reintialize()
                    self.train()
            elif self.config.cl_strategy == 'Pretrain':
                if not self.saved_model_path:
                    raise FileNotFoundError(f'No saved checkpoint is available before evaluating {year}.')
                self.cache_dir = self.saved_model_path
                self.test_dl = self.data.get_dataloader(getattr(self.model, 'norm', None), year=year, test_only=True)
                self.model.graph = self.model.get_graph(year)
            else:
                if not self.saved_model_path:
                    raise FileNotFoundError(f'No saved checkpoint is available before training {year}.')
                self.load_model(self.saved_model_path)
                self.model.prepare_for_new_task()
                self.cache_dir = f'cache/{self.dataset_name}/{self.model_name}_{year}_{self.cur_time}.temp'
                # self.config.train_size = 0.2
                self.train_dl, self.val_dl, self.test_dl = self.data.get_dataloader(getattr(self.model, 'norm', None),
                                                                                    year=year, subset=getattr(self.model, 'subset', None))
                self.reintialize()
                self.train()
            self.load_model()
            result = self.evaluate_year(result, year)
        self.model.last_info()
        self.update_yearly_info()
        return result

    def evaluate_year(self, result, year):
        result[year] = self.evaluate()
        self.save_yearly_info()
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def load_model(self, path=None):
        path = path or self.cache_dir
        model_state, optimizer_state = torch.load(path)
        model_keys = self.model.state_dict().keys()
        if model_state.keys() != model_keys:
            if all(key.startswith('model.') for key in model_state) and {
                key[len('model.'):] for key in model_state
            } == model_keys:
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state, 'model.')
            elif {'model.' + key for key in model_state} == model_keys:
                prefixed_state = type(model_state)(('model.' + key, value) for key, value in model_state.items())
                if hasattr(model_state, '_metadata'):
                    prefixed_state._metadata = {
                        'model.' + key if key else 'model': value
                        for key, value in model_state._metadata.items()
                    }
                model_state = prefixed_state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model(self):
        super().save_model()
        self.saved_model_path = self.cache_dir
        self.model.saved_model_path = self.cache_dir

    def reintialize(self):
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        total_p = sum(p.numel() for p in self.model.parameters())
        train_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Total parameter numbers: {total_p}')
        self.logger.info(f'Trainable parameter numbers: {train_p}')
        self.logger.info( f'Trainable rate: {train_p/total_p:.2%}')

    def find_latest_model(self, year):
        model_files = [f for f in os.listdir(f'cache/{self.dataset_name}') if f.startswith(self.model_name + '_' + str(year))]
        model_files.sort(reverse=True)
        return f'cache/{self.dataset_name}/{model_files[0]}'

    def extra_config(self):
        self.config.load_yaml_config(self.model_name + '_cl', self.dataset_name)

    def save_yearly_info(self):
        self.cl_run_batch.append(self.run_batch)
        self.cl_train_time.append(self.train_time)
        self.cl_infer_time.append(self.infer_time)
        self.cl_peak_mem.append(self.peak_mem)

    def update_yearly_info(self):
        self.run_batch = self.cl_run_batch
        self.train_time = self.cl_train_time
        self.infer_time = self.cl_infer_time
        self.peak_mem = self.cl_peak_mem
        seed = self.seed[0] if isinstance(self.seed, list) and self.seed else self.seed
        self.seed = [seed for i in range(self.start_year, self.end_year + 1)]

    def reinitialize_cl_base(self):
        super().reintialize()
        self.model.year = self.config.start_year

    def train_base_model(self, n_times=5):
        year = self.config.start_year
        self.model.year = year
        self.logger.info(f'Starting training base model for {n_times} times')
        self.train_dl, self.val_dl, self.test_dl = self.data.get_dataloader(getattr(self.model, 'norm', None), year=year)
        norm = self.model.norm

        best_result = None
        best_model = None
        ensure_dir('cache/best_model/')

        for i in range(n_times):
            self.logger.info(f"{'=' * 20} Starting Training {i + 1}/{n_times} times {'=' * 20}")
            if i > 0:
                self.reinitialize_cl_base()
                self.model.norm = norm
            self.cache_dir = f'cache/{self.dataset_name}/{self.model_name}_{year}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.temp'

            self.train()
            self.load_model()
            result = self.evaluate()

            if self._is_better(result, best_result, False, 0.3, True):
                best_result = result
                best_model = (self.model.state_dict(), self.optimizer.state_dict())
                self.logger.info(f'Best model replaced in {i + 1}. Current result: {best_result}')

            gc.collect()
            torch.cuda.empty_cache()

        self.logger.info(f'Best result: {best_result}')
        save_dir = f'cache/best_model/{self.dataset_name}_{self.config.backbone}_{next(iter(best_result.values()))}'
        torch.save(best_model, save_dir)
        self.logger.info(f'Best model saved to {save_dir}')
        return

    def load_data(self, year, test_only=False, subset=False):
        ss = getattr(self.model, 'subset', None) if subset else None
        if test_only:
            self.test_dl = self.data.get_dataloader(getattr(self.model, 'norm', None), year=year, test_only=True, subset=ss)
        else:
            self.train_dl, self.val_dl, self.test_dl = self.data.get_dataloader(getattr(self.model, 'norm', None), year=year, subset=ss)