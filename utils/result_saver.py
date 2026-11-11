import os
import numpy as np
import csv

round_save = 4

class result_saver():
    def __init__(self, task, dir, metric=None, mlen=None, thre=None):
        mlen = mlen[:]
        self.task = task
        self.dir = dir
        if os.path.exists(self.dir):
            with open(self.dir, 'r') as csv_file:
                line = csv_file.readline().split(',')
                if self.task in ['IO', 'CL']:
                    self.head = line[2:-5]
                else:
                    self.head = line[1:-5]
        else:
            self.head = []
            assert metric is not None
            if isinstance(mlen, int):
                mlen = [mlen]
            mlen.sort()
            if '_s_' in dir:
                mlen.append('avg')
            if thre is None:
                thre = [None]
            elif isinstance(thre, int):
                thre = [thre]
                thre.sort()
            else:
                thre.sort()
            for i in reversed(mlen):
                for j in thre:
                    for mtc in metric:
                        self.head.append(self.get_key(i, j, mtc))
            with open(self.dir, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                if self.task == 'IO':
                    writer.writerow(['Model Name', 'Type'] + self.head + ['Seed', 'Epoch', 'Train Time', 'Infer Time', 'Peak Mem'])
                elif self.task == 'CL':
                    writer.writerow(['Model Name', 'Year'] + self.head + ['Seed', 'Epoch', 'Train Time', 'Infer Time', 'Peak Mem'])
                else:
                    writer.writerow(['Model Name'] + self.head + ['Seed', 'Epoch', 'Train Time', 'Infer Time', 'Peak Mem'])
        if self.task == 'IO':
            self.result = [[[] for i in self.head], [[] for i in self.head]]
        elif self.task == 'CL':
            self.result = []
        else:
            self.result = [[] for i in self.head]

    def save_result(self, result, run_info, model_name):
        self.write_csv(result, run_info, model_name)

    def write_csv(self, result, run_info, model_name):
        # type 2
        if self.task == 'IO':
            for i, k in enumerate(self.head):
                self.result[0][i].append(result[0][k])
                self.result[1][i].append(result[1][k])
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                row1 = [model_name, 'in-flow'] + [round(result[0][i], round_save) for i in self.head] + run_info
                row2 = [model_name, 'out-flow'] + [round(result[1][i], round_save) for i in self.head] + run_info
                writer.writerow(row1)
                writer.writerow(row2)
        elif self.task == 'CL':
            mean = lambda x: sum(x) / len(x)
            avg_info = [mean(i) for i in run_info]
            avg_info[0] = int(avg_info[0])
            run_info = [list(i) for i in zip(*run_info)]
            years = list(result.keys())
            self.years = years
            temp = np.zeros((len(years) + 1, len(self.head))) # run_times, year, metric
            for y, year in enumerate(years):
                for i, k in enumerate(self.head):
                    temp[y, i] = result[year][k]
            temp[len(years)] = temp[:len(years)].mean(0)
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                for y, year in enumerate(years):
                    row = [model_name, str(year)] + [round(i, round_save) for i in temp[y]] + run_info[y]
                    writer.writerow(row)
                row = [model_name, 'average'] + [round(i, round_save) for i in temp[len(years)]] + avg_info
                writer.writerow(row)
            self.result.append(temp)
        # type 1
        else:
            for i, k in enumerate(self.head):
                self.result[i].append(result[k])
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                row = [model_name] + [round(result[i], round_save) for i in self.head] + run_info
                writer.writerow(row)

    def collect_run_info(self, model):
        def nround(x, a):
            if isinstance(x, list):
                return [round(i, a) for i in x]
            else:
                return round(x, a)
        return [model.seed, model.run_batch, nround(model.train_time, 5), nround(model.infer_time, 5), nround(model.peak_mem, 2)]

    def save_average(self):
        if self.task == 'IO':
            result = [[0 for i in self.head] for j in range(2)]
            for i, k in enumerate(self.head):
                result[0][i] = round(np.mean(self.result[0][i]), round_save)
                result[1][i] = round(np.mean(self.result[1][i]), round_save)
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                row1 = ['Average', 'in-flow'] + [round(i, round_save) for i in result[0]] + ['','']
                row2 = ['Average', 'out-flow'] + [round(i, round_save) for i in result[1]] + ['','']
                writer.writerow(row1)
                writer.writerow(row2)
        elif self.task == 'CL':
            result = np.mean(self.result, 0)
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                for y, year in enumerate(self.years):
                    row = ['Average', str(year)] + [round(i, round_save) for i in result[y]] + ['','']
                    writer.writerow(row)
                row = ['Average', 'average'] + [round(i, round_save) for i in result[len(self.years)]] + ['','']
                writer.writerow(row)
        else:
            result = [0 for i in self.head]
            for i, k in enumerate(self.head):
                result[i] = round(np.mean(self.result[i]), round_save)
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                row = ['Average'] + [round(i, round_save) for i in result] + ['','']
                writer.writerow(row)

    def get_key(self, len, thre, metric):
        if not isinstance(len, int):
            if thre is not None:
                return metric + '>' + str(thre)
            else:
                return metric
        else:
            if thre is not None:
                return metric + '@' + str(len) + '>' + str(thre)
            else:
                return metric + '@' + str(len)