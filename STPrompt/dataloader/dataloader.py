import numpy as np
from logging import getLogger
import torch
from torch.utils.data import DataLoader, Dataset

class Data_Base():
    def __init__(self):
        self.logger = getLogger()
        self.name_data = 'data.npy'
        self.name_time = 'time.npy'

    def load_data(self):
        #load data
        data = np.load(self.config.data_path + self.file_name + self.name_data)

        #load temporal information (time-of-day, day-of-week)
        if not hasattr(self, 'extern_dir'):
            self.extern_dir = self.config.data_path + self.file_name + self.name_time
        extern = np.load(self.extern_dir)

        #reset the start and end index if needed
        if data.shape[0] > self.config.total_size:
            if self.config.start_date + self.config.total_size > data.shape[0]:
                raise ValueError
            data = data[self.config.start_date:self.config.total_size]
            extern = extern[self.config.start_date:self.config.total_size]

        #if task_mode is IO and need_od is False, then process the data to io data
        keep_od = self.config.need_od and self.config.task_name == 'IO' or self.config.task_name == 'OD'
        if not keep_od and data.shape[-1] == data.shape[-2]:
            data = np.stack([data.sum(-2), data.sum(-1)], -1)

        return data, extern

    def load_graph(self):
        g = np.load(self.config.data_path + self.file_name + 'graph.npy')
        return g

    def load_graph_dis(self):
        try:
            g = np.load(self.config.data_path + self.file_name + 'graph_distance.npy').astype(np.float32)
        except:
            g = None
        return g

    def split_tvt(self, data, extern=None):
        total_time = data.shape[0]
        #split dataset based on previous setting
        if isinstance(self.config.train_size, int):
            num_train = self.config.train_size
            num_test = self.config.test_size
        elif isinstance(self.config.train_size, float):
            num_train = round(total_time * self.config.train_size)
            num_test = round(total_time * self.config.test_size)
        else:
            raise "train_size only support int or float"

        # period processing
        pw = self.config.period_window or 0
        if self.config.use_period:
            t = data[: num_train]
            v = data[num_train: -num_test]
            ts = data[-num_test - self.config.num_times*pw:]
        else:
            t = data[: num_train]
            v = data[num_train: -num_test]
            ts = data[-num_test:]

        #extern data
        if extern is None:
            te = ve = tse = None
        else:
            te = extern[: num_train]
            ve = extern[num_train: -num_test]
            tse = extern[-num_test - self.config.num_times*pw:]

        return [t, te], [v, ve], [ts, tse]

    def get_dataloader(self, norm=None):
        data = self.load_data()
        train, val, test = self.split_tvt(*data)
        if norm is not None:
            norm.initialize(train[0])
        train = self.create_dataloader(train[0], train[1], True, norm=norm)
        val = self.create_dataloader(val[0], val[1], norm=norm)
        test = self.create_dataloader(test[0], test[1], norm=norm)
        return train, val, test

    def create_dataloader(self, data, extern, shuffle=False, norm=None):
        ds = self.create_ds4dl(data, extern, norm)
        dl = DataLoader(ds, self.config.batch_size, shuffle=shuffle)
        return dl

    def create_ds4dl(self, data, extern, norm):
        if not self.config.use_period and data.shape[-1] != data.shape[-2]:
            ds = Fast_Data4dl(self.config.input_window, self.config.output_window, data, extern, self.config.need_time, norm)
        else:
            period_len = self.config.period_window or 7
            period_size = self.config.period_len or [0, 1]
            ds = Data4dl(self.config.input_window, self.config.output_window, data, extern, self.config.use_period, period_len, period_size, self.config.need_time, self.config.task_name)
        return ds

    def get_whole_dataloader(self):
        data = self.load_data()
        dl = self.create_dataloader(*data)
        return dl

class Data_CL(Data_Base):
    def __init__(self):
        super().__init__()
        self.logger = getLogger()
        self.name_data = 'data.npy'
        self.name_time = 'time.npy'

    def load_data(self, year):
        #load data
        data = np.load(self.config.data_path + self.file_name + f'data_{year}.npy')

        #load temporal information (time-of-day, day-of-week)
        extern = np.load(self.config.data_path + self.file_name + f'time_{year}.npy')

        #reset the start and end index if needed
        if data.shape[0] > self.config.total_size:
            if self.config.start_date + self.config.total_size > data.shape[0]:
                raise ValueError
            data = data[self.config.start_date:self.config.start_date + self.config.total_size]
            extern = extern[self.config.start_date:self.config.start_date + self.config.total_size]

        return data, extern

    def get_dataloader(self, norm=None, year=2020, test_only=False, subset=None):
        data = self.load_data(year)
        train, val, test = self.split_tvt(*data)
        if subset is not None:
            train[0] = train[0][:, subset]
            # val[0] = val[0][:, subset]
        if norm is not None:
            norm.initialize(train[0])
        test = self.create_dataloader(test[0], test[1], norm=norm)
        if test_only:
            return test
        train = self.create_dataloader(train[0], train[1], True, norm=norm)
        val = self.create_dataloader(val[0], val[1], norm=norm)
        return train, val, test

    def load_graph(self, year):
        return np.load(self.config.data_path + self.file_name + f'graph_{year}.npy')

    def get_dataloader_subset(self, length=0, year=2020, norm=None, subset=None, renorm=False):
        data = self.load_data(year)
        data = [i[: length] if i is not None else None for i in data]
        if subset is not None:
            data[0] = data[0][:, subset]
        if norm is not None and renorm:
            norm.initialize(data[0])
        dl = self.create_dataloader(data[0], data[1], norm=norm)
        return dl

    # def create_dataloader(self, data, extern, shuffle=False):
    #     period_len = self.config.period_window or 7
    #     period_size = self.config.period_len or [0, 1]
    #     data = Data4dl(self.config.input_window, self.config.output_window, data, extern, self.config.use_period, period_len, period_size, self.config.need_time)
    #     dl = DataLoader(data, self.config.batch_size, shuffle=shuffle)
    #     return dl

class Data4dl(Dataset):
    def __init__(self, input_len, output_len, data, extern=None, use_period=False, period_len=7, period_size=(0, 1), use_time=False, task='IO'):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.data = data
        if data.ndim == 2:
            self.data = data[:, :, np.newaxis]
        if self.data.shape[-1] > 2 and task != 'OD':
            self.data_y = np.stack([data.sum(-2), data.sum(-1)], -1)
        else:
            self.data_y = self.data
        self.extern = extern
        self.use_period = use_period
        self.use_time = use_time
        self.p_len = period_len
        self.p_size = period_size

        if self.use_period:
            self.process_period_data()
            self.len = self.data.shape[0] - self.output_len - max(self.input_len, -self.p_size[1]) + 1 - period_len * 48
        else:
            self.len = self.data.shape[0] - self.output_len - self.input_len + 1

    def __getitem__(self, item):
        if self.use_period:
            a = item + self.input_len
            b = item + self.p_len * 48
            c = b + self.input_len
            rd = [self.data[b: c], self.data[c: c + self.output_len], self.p_data[item: b: 48]]
            if self.use_time:
                rd.extend([self.extern[b: c], self.extern[c: c + self.output_len], self.extern[item: b: 48]])
            return tuple(rd)
        else:
            if self.extern is None or not self.use_time:
                return self.data[item: item + self.input_len], self.data_y[item + self.input_len: item + self.input_len + self.output_len]
            else:
                return self.data[item: item + self.input_len], self.data_y[item + self.input_len: item + self.input_len + self.output_len]\
                    , self.extern[item: item + self.input_len], self.extern[item + self.input_len: item + self.input_len + self.output_len]

    def __len__(self):
        return self.len

    def process_period_data(self):
        s_p = max(self.input_len, -self.p_size[1]) + self.p_size[0]
        p_s = self.p_size[1] - self.p_size[0]
        self.p_data = np.stack([self.data[s_p + i: -p_s + i - 47] for i in range(p_s)], 0).transpose((1, 0, 2, 3)).squeeze()

class Fast_Data4dl(Dataset):
    # This is a fast implement which remove support for OD data and period data
    def __init__(self, input_len, output_len, data, extern=None, use_time=False, norm=None):
        super().__init__()
        assert data.shape[-1] != data.shape[-2], 'Fast dataset DO NOT support OD data'
        if data.ndim == 2:
            data = data[..., np.newaxis]
        self.input_len = input_len
        self.output_len = output_len
        self.use_time = use_time
        self.x, self.y = torch.from_numpy(data).unfold(0, self.input_len + self.output_len, 1).permute(0, 3, 1, 2).chunk(2, 1)
        if norm is not None and norm.pre_norm:
            self.x = norm(self.x)
            if norm.pre_norm == 'all':
                self.y = norm(self.y)
        if self.use_time:
            self.xe, self.ye = torch.from_numpy(extern).unfold(0, self.input_len + self.output_len, 1).permute(0, 2, 1).chunk(2, 1)

    def __getitem__(self, item):
        if self.use_time:
            return self.x[item], self.y[item], self.xe[item], self.ye[item]
        else:
            return self.x[item], self.y[item]

    def __len__(self):
        return self.x.shape[0]