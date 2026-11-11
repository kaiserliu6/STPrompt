import os
import torch
import time
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import importlib
from tqdm import tqdm
from matplotlib import pyplot as plt
from logging import getLogger

from utils import Config, Evaluator, ensure_dir, Evaluator_Dual

class Trainer():
    def __init__(self, model_name, dataset_name, config_file_list = None, config_dict=None, random_seed=None):
        # set random seed
        self._set_random_seed(random_seed)

        # initial config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config = Config(model_name=model_name, dataset_name=dataset_name, hyper_config=config_dict)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.extra_config()

        # get dataloader and model, data not loaded yet, load config in dataset first
        self.data = self._get_dataloader()
        self.model = self._get_model()

        # set evaluator according to task, for OD and TF simple, for IO evaluate both in-flow and out-flow
        if self.config.task_name == 'IO' and self.config.dual_evaluator:
            self.evaluator = Evaluator_Dual(self.config.metric_list, self.config.metric_len, self.config.metirc_threshold)
            self.valider = Evaluator_Dual(self.config.metric_list, self.config.metric_len, self.config.metirc_threshold, valid=True)
        else:
            self.evaluator = Evaluator(self.config.metric_list, self.config.metric_len, self.config.metirc_threshold, mode=self.config.eval_mode)
            self.valider = Evaluator(self.config.metric_list, self.config.metric_len, self.config.metirc_threshold, valid=self.config.short_valid, mode=self.config.eval_mode)

        # initialize logging
        self.logger = getLogger()
        # self._writer = SummaryWriter(ensure_dir('cache/summary_cache/'))
        time = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self._init_tensorboard()
        ensure_dir(f'cache/{dataset_name}')
        self.cache_dir = f'cache/{dataset_name}/{model_name}_{time}.temp'
        if self.config.mode == 'test':
            self.cache_dir = f'cache/{dataset_name}/{model_name}_temp.temp'
        self.evaluate_res_dir = ensure_dir('result/')

        # intialize training component
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        self.logger.info(f'Total parameter numbers: {sum(param.numel() for param in self.model.parameters())}')
        self.run_batch = 0
        self.train_time = 0
        self.infer_time = 0
        self.peak_mem = 0

    # base function called to train the model
    def run(self, epochs=None):
        # self.load_model()
        epochs = epochs or self.config.epoch
        self.train_dl, self.val_dl, self.test_dl = self.data.get_dataloader(getattr(self.model, 'norm', None))
        self.train(epochs)
        self.load_model()
        result = self.evaluate()
        self.model.last_info()
        return result

    def train(self, epochs=None):
        self.not_improve = 0
        self.best_performance = None
        epochs = epochs or self.config.epoch
        self.logger.info('Start training...')
        train_time, eval_time, peak_mems = [], [], []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        for epoch_idx in range(epochs):
            start_time = time.time()
            start_mem = torch.cuda.memory_allocated() / 1024 ** 3
            train_loss = self._train_epoch(epoch_idx)
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            peak_mems.append(peak_mem)
            mid_time = time.time()
            train_time.append(mid_time - start_time)
            val_loss = self._valid_epoch(epoch_idx)
            eval_time.append(time.time() - mid_time)
            self._write_training_curve(epoch_idx, train_loss, val_loss)

            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss.values()[0] if self.config.scheduler.lower() == 'reducelronplateau' else None)

            if epoch_idx % self.config.log_step == 0:
                self.logger.info(f'Epoch {epoch_idx}/{epochs}: train_loss is {train_loss}, val_result is {val_loss}, '
                               f'train_time {train_time[-1]:.2f}s, val_time {eval_time[-1]:.2f}s, '
                                 f'peak memory consumption {peak_mem - start_mem:.2f}GB')

            if epoch_idx % self.config.val_step == 0:
                if self._is_better(val_loss, getattr(self, 'best_performance', None)):
                    self.logger.debug('Current model is better')
                    self.best_performance = val_loss
                    self.bpt = self._test_epoch(epoch_idx)
                    self.logger.info(f'New best model, test result: {self.bpt}')
                    self.not_improve = 0
                    self.save_model()
                    self.run_batch = epoch_idx
                else:
                    self.not_improve += self.config.val_step
                    self.logger.debug('Not improved for %d epoch', self.not_improve)
                    if self.config.early_stop and self.not_improve >= self.config.early_stop:
                        self.logger.info(f'Early stopping at epoch {epoch_idx}')
                        break
            torch.cuda.reset_peak_memory_stats()
        self.train_time = sum(train_time) / len(train_time)
        self.peak_mem = max(peak_mems)
        self.logger.info(f'Trained totally {len(train_time)} epochs, average train time is {self.train_time}s, average eval time is {sum(eval_time) / len(eval_time)}s')
        return

    def evaluate(self):
        self.logger.info('Start evaluating...')
        start_time = time.time()
        with torch.no_grad():
            self.model.eval()
            self.evaluator.clear()
            for batch in self.test_dl:
                output, truth = self.model.predict(batch)
                self.evaluator.collect(output, truth)
            test_result = self.evaluator.evaluate()
            if self.config.save_result:
                self.evaluator.save_result(f'{self.config.save_dir}{self.model_name}_{self.dataset_name}.npz')
            self.logger.info(test_result)
        self.infer_time = time.time() - start_time
        return test_result

    def _train_epoch(self, epoch_idx):
        self.model.train()
        losses = []
        for batch in tqdm(self.train_dl):
            self.optimizer.zero_grad()
            if self.config.need_idx:
                loss = self.model.calculate_loss(batch, epoch_idx)
            else:
                loss = self.model.calculate_loss(batch)
            loss.backward()
            losses.append(loss.item())
            if self.config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        mean_loss = np.mean(losses)
        return mean_loss

    def _valid_epoch(self, epoch_idx=None):
        with torch.no_grad():
            self.model.eval()
            if self.config.val_use_loss:
                total_loss = []
                for batch in self.val_dl:
                    output, truth = self.model.predict(batch)
                    loss = self.model.loss(output, truth)
                    # loss = self.model.calculate_loss(batch)
                    if not loss.isnan():
                        total_loss.append(loss.item())
                return np.mean(total_loss)
            else:
                self.valider.clear()
                for batch in self.val_dl:
                    output, truth = self.model.predict(batch)
                    self.valider.collect(output, truth)
                return self.valider.evaluate()

    def _test_epoch(self, epoch_idx=None):
        with torch.no_grad():
            self.model.eval()
            self.evaluator.clear()
            for batch in self.test_dl:
                output, truth = self.model.predict(batch)
                self.evaluator.collect(output, truth)
            return self.evaluator.evaluate()

    def _set_random_seed(self, seed):
        self.seed = seed or np.random.randint(10000)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _get_dataloader(self):
        try:
            if self.dataset_name.lower().startswith('nyc_taxi'): # this is an exception since I will create many dataset based on NYC Taxi data
                return getattr(importlib.import_module('dataloader'), 'Data_nyc_taxi')(self.config, self.dataset_name)
            elif self.dataset_name.lower().startswith('syn'):
                return getattr(importlib.import_module('dataloader'), 'Data_syn')(self.config, self.dataset_name)
            else:
                return getattr(importlib.import_module('dataloader'), f'Data_{self.dataset_name.lower()}')(self.config, self.dataset_name)
        except:
            raise Exception(f'Data_{self.dataset_name.lower()} not exist, see dataloader for supported dataset')

    def _get_model(self):
        return getattr(importlib.import_module('model'), self.model_name)(self.config, self.data).to(self.device)

    def _is_better(self, current, best, direct_loss=True, better_ratio=0.5, compare_main=False):
        if best is None:
            return True
        if self.config.val_use_loss and direct_loss:
            return current < best
        c1 = not compare_main or next(iter(current.values())) < next(iter(best.values()))
        c2 = sum(1 for k, v in current.items() if v < best[k]) / len(current) > better_ratio
        return c1 and c2

    def save_model(self):
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), self.cache_dir)

    def load_model(self, path=None):
        path = path or self.cache_dir
        model_state, optimizer_state = torch.load(path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def _build_optimizer(self):
        """
        根据全局参数`optimizer`选择optimizer
        """
        optmizer_name = self.config.optimizer.lower()
        self.logger.info(f'Using {optmizer_name} optimizer.')
        if optmizer_name == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **self.config.optimizer_kwargs())
        elif optmizer_name == 'adamw':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.model.parameters()), **self.config.optimizer_kwargs())
        elif optmizer_name == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), **self.config.optimizer_kwargs())
        elif optmizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), **self.config.optimizer_kwargs())
        elif optmizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()), **self.config.optimizer_kwargs())
        elif optmizer_name == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(filter(lambda p: p.requires_grad, self.model.parameters()), **self.config.optimizer_kwargs())
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **self.config.optimizer_kwargs())
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        scheduler_name = self.config.scheduler or 'none'
        if self.config.lr_decay_ratio and scheduler_name != 'none':
            scheduler_name = scheduler_name.lower()
            self.logger.info(f'Using {scheduler_name} lr_scheduler.')
            if scheduler_name == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **self.config.scheduler_kwargs())
            elif scheduler_name == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **self.config.scheduler_kwargs())
            elif scheduler_name == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **self.config.scheduler_kwargs())
            elif scheduler_name == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.config.scheduler_kwargs())
            elif scheduler_name == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, **self.config.scheduler_kwargs())
            elif scheduler_name == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.config.scheduler_kwargs())
            else:
                self.logger.warning('Received unrecognized lr_scheduler, please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def _init_tensorboard(self):
        if SummaryWriter is None:
            self._writer = None
            self.summary_dir = None
            self.logger.warning('TensorBoard is not available. Please install it with `pip install tensorboard` if needed.')
            return
        summary_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        self.summary_dir = ensure_dir(f'cache/summary_cache/{self.dataset_name}/{self.model_name}_{summary_time}_seed{self.seed}/')
        self._writer = SummaryWriter(self.summary_dir)
        self.logger.info(f'TensorBoard summary will be saved to {self.summary_dir}')

    def _write_training_curve(self, epoch_idx, train_loss, val_loss):
        if self._writer is None:
            return
        self._writer.add_scalar('loss/train', float(train_loss), epoch_idx)
        if isinstance(val_loss, dict):
            first_value = None
            for key, value in val_loss.items():
                value = self._to_scalar(value)
                if value is None:
                    continue
                if first_value is None:
                    first_value = value
                self._writer.add_scalar(f'val/{key}', value, epoch_idx)
            if first_value is not None:
                self._writer.add_scalar('loss/val', first_value, epoch_idx)
        else:
            value = self._to_scalar(val_loss)
            if value is not None:
                self._writer.add_scalar('loss/val', value, epoch_idx)
        self._writer.flush()

    def _to_scalar(self, value):
        try:
            if torch.is_tensor(value):
                return value.detach().cpu().item()
            if isinstance(value, np.ndarray):
                return value.item()
            return float(value)
        except (TypeError, ValueError):
            return None

    def extra_config(self):
        pass

    def find_best_model(self, run_times=5, epochs=None, seed_list=None):
        best_result = None
        best_model = None
        ensure_dir('best_model/')
        if seed_list is not None:
            run_times = min(run_times, len(seed_list))
        else:
            run_times = [None for i in range(run_times)]

        for i in range(run_times):
            self.logger.info(f"{'=' * 20} Starting Run {i + 1}/{run_times} {'=' * 20}")

            if i > 0:
                self.reintialize(seed_list[i])
            result = self.run(epochs)

            if self._is_better(result, best_result, False, 0.3, True):
                best_result = result
                best_model = (self.model.state_dict(), self.optimizer.state_dict())
                self.logger.info(f'Best model replaced in {i + 1}. Current result: {best_result}')

        self.logger.info(f"Best result: {best_result}")
        save_dir = f'best_model/{self.dataset_name}_{self.model_name}_{next(iter(best_result.values()))}'
        torch.save(best_model, save_dir)
        self.logger.info(f"Best model saved to {save_dir}")
        return

    def run_multi_time(self, run_times=5, epochs=None, seed_list=None):
        ensure_dir('cache/')
        if seed_list is not None:
            run_times = min(run_times, len(seed_list))
        else:
            run_times = [None for i in range(run_times)]

        for i in range(run_times):
            self.logger.info(f"{'=' * 20} Starting Run {i + 1}/{run_times} {'=' * 20}")
            self.reintialize(seed_list[i])
            result = self.run(epochs)
            save_dir = f'cache/{self.dataset_name}/{self.model_name}_{self.seed}_{next(iter(result.values()))}'
            torch.save((self.model.state_dict(), self.optimizer.state_dict()), save_dir)
        return

    def reintialize(self, seed=None):
        if getattr(self, '_writer', None) is not None:
            self._writer.close()
        self._set_random_seed(seed)
        self.model = self._get_model()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self.logger.info(f'Total parameter numbers: {sum(param.numel() for param in self.model.parameters())}')
        self.run_batch = 0
        self.train_time = 0
        self.infer_time = 0
        self.peak_mem = 0
        self._init_tensorboard()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class Trainer_test(Trainer):
    def __init__(self, model_name, dataset_name, config_file_list = None, config_dict=None, random_seed=None):
        super().__init__(model_name, dataset_name, config_file_list, config_dict, random_seed)

    def find_latest_model(self, k=1):
        model_files = [f for f in os.listdir(f'cache/{self.dataset_name}') if f.startswith(self.model_name + '_')]
        model_files.sort(reverse=True)
        return [f'cache/{self.dataset_name}/{file}' for file in model_files[:k]]

    def save_result(self, cache_dir=None, k=1):
        cache_dir = cache_dir or self.find_latest_model(k)
        if not isinstance(cache_dir, list):
            cache_dir = [cache_dir]
        _, _, self.test_dl = self.data.get_dataloader(getattr(self.model, 'norm', None))
        for i, model in enumerate(cache_dir):
            self.load_model(model)
            with torch.no_grad():
                self.model.eval()
                outputs = []
                for batch in self.test_dl:
                    output, truth = self.model.predict(batch)
                    outputs.append(output)
            outputs = torch.cat(outputs, 0)
            np.save(f'result/{self.dataset_name}_{self.model_name}_{i}.npy', outputs.cpu().numpy())

        # test function used to run an evaluate
    def test(self, cache_dir=None):
        self.load_model(cache_dir or self.find_latest_model())
        self._test_epoch()

    def save_result_every_step(self):
        self.evaluator = Evaluator(self.config.metric_list, [i + 1 for i in range(12)],
                                   self.config.metirc_threshold, mode='single')
        self.load_model(self.find_latest_model())
        result = self.evaluate()
        self.evaluator.save_result(self.config.save_dir + f'{self.model_name}_{self.dataset_name}.npz')
        return result

    def extra_config(self):
        # This support add extra setting for other trainers
        pass