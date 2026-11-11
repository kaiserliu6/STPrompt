import numpy as np

from dataloader import Data_CL

class Data_nyc_cl(Data_CL):
    def __init__(self, config, dataset_name):
        super().__init__()
        conf = {
            'num_nodes': 66,
            'input_dim': 2,
            'output_dim': 2,
            'num_times': 48,
            'num_nodes_year': {
                2019: 66,
                2020: 109,
                2021: 170,
                2022: 238,
                2023: 258,
            },
            'metirc_threshold': 0,
            'task_name': 'CL'
        }
        config.update(conf, 7)
        conf = {
            'total_size': 16032,
            'train_size': 212 * 48,
            'test_size': 61 * 48,
            'start_date': 31 * 48,
            'start_year': 2019,
            'end_year': 2023,
        }
        config.update(conf, 5)
        self.config = config
        self.file_name = f'NYC_taxi_CL/'

    def get_dataloader(self, norm=None, year=2019, test_only=False, subset=None):
        if year == 2020:
            self.config.update({'total_size': 17568,
                                'start_date': 0,
                                'train_size': 244 * 48}, 5)
        elif year != 2019:
            self.config.update({'total_size': 17520,
                                'train_size': 243 * 48}, 5)
        return super().get_dataloader(norm, year, test_only, subset)

    def load_graph(self, year=2019):
        g = np.load(self.config.data_path + self.file_name + f'graph_{year}.npy')
        g = (g < 5).astype(np.int32)
        return g