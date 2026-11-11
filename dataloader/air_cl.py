import numpy as np

from dataloader import Data_CL

class Data_air_cl(Data_CL):
    def __init__(self, config, dataset_name):
        super().__init__()
        conf = {
            'num_nodes': 1087,
            'input_dim': 1,
            'num_times': 24,
            'num_nodes_year': {
                2016: 1087,
                2017: 1154,
                2018: 1193,
                2019: 1202,
            },
            'metirc_threshold': 0,
            'task_name': 'CL'
        }
        config.update(conf, 7)
        conf = {
            'total_size': 10000, # in fact 8578, 8619, 8378, 8490
            'train_size': 0.6,
            'test_size': 0.2,
            'start_date': 0,
            'start_year': 2016,
            'end_year': 2019,
        }
        config.update(conf, 5)
        self.config = config
        self.file_name = f'AIR_CL/'