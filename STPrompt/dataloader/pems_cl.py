import numpy as np

from dataloader import Data_CL

class Data_pems_cl(Data_CL):
    def __init__(self, config, dataset_name):
        super().__init__()
        conf = {
            'num_nodes': 655,
            'input_dim': 1,
            'num_times': 288,
            'num_nodes_year': {
                2011: 655,
                2012: 715,
                2013: 786,
                2014: 822,
                2015: 834,
                2016: 850,
                2017: 871
            },
            'metirc_threshold': 0,
            'task_name': 'CL'
        }
        config.update(conf, 7)
        conf = {
            'total_size': 8928,
            'train_size': 0.6,
            'test_size': 0.2,
            'start_date': 0,
            'start_year': 2011,
            'end_year': 2017,
        }
        config.update(conf, 5)
        self.config = config
        self.file_name = f'PEMS_CL/'

    def load_graph(self, year):
        return np.load(self.config.data_path + self.file_name + f'graph_{year}.npy')