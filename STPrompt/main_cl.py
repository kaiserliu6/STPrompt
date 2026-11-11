import argparse
import logging
from trainer.trainer_cl import Trainer_CL
from utils import result_saver
from utils.run_time_recorder import RunTimeRecorder

DEBUG=True

"""
currently have 2 evaluator:
evaluator: result dict {'{metric}@{len}>{thre}':value}
dual_evaluator: result [{'{metric}@{len}>{thre}':value},{..}]
r, s, b for result, seed, run epoch, respectively
dir: csv save dir
result: shape as above
run_info: [s, b]
model_name: need if is_average=False, save in csv as index
is_average: whether the result is the average result
"""

def get_trainer(model, dataset, cd):
    return Trainer_CL(model, dataset, config_dict=cd)


if __name__ == "__main__":
    model_list = ['Pretrain', 'Retrain', 'NewNode', 'AllNode', 'EAC', 'trafficstream', 'STBP', 'STPrompt']
    data_list = ['NYC_CL', 'PEMS_CL', 'AIR_CL']
    model_name = model_list[-1]
    data_name = data_list[1]
    save_name = None
    param_name = 'tod_mem'
    param_dict = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70]

    run_times = 1
    debug_flag = (0, 0, 0)
    TEST_MODE = True
    # TEST_MODE = False
    ENABLE_RUNTIME_TIMER = False
    RUNTIME_CSV_PATH = "result/run_time.csv"

    cd = {'output_window': 12, 'metric_len': [3, 6, 12], 'eval_mode': 'single', 'metirc_threshold': 0, 'task_name': 'CL'}

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=model_name, help='name of models, see model for more details')
    parser.add_argument("--dataset", "-d", type=str, default=data_name, help='name of datasets, see dataloader for more details')
    parser.add_argument("--config_files", type=str, default=None, help="config files, split with ','")
    parser.add_argument('--run_times', type=int, default=run_times, help='times of running each model')
    parser.add_argument("--debug_mode", type=int, default=debug_flag, help="running mode, model, data, hyper")
    parser.add_argument('--model_list', type=list, default=model_list, help='model list for debug mode 1')
    parser.add_argument('--data_list', type=list, default=data_list, help='data list for debug mode 2')
    parser.add_argument('--runtime_timer', type=int, choices=[0, 1], default=int(ENABLE_RUNTIME_TIMER), help='enable total runtime recorder, 1 for on and 0 for off')
    parser.add_argument('--runtime_csv', type=str, default=RUNTIME_CSV_PATH, help='csv path for total runtime records')
    # parser.add_argument('--param_name', type=list, default=param_name, help='parameter name for debug mode 3')
    # parser.add_argument('--param_dict', type=list, default=param_dict, help='searching value for debug mode 3')
    """
    running mode supported:
    0: one model, one dataset, config unchange
    1: models in model_list, one dataset, config unchange
    2: one model, datasets in data_list, config unchange
    3: one model, one dataset, different value of give parameter
    4: models in model_list, datasets in data_list, config unchange
    5: same as 3, support multiple parameter
    6: one model, datasets in data_list, different value of give parameter
    """

    args, _ = parser.parse_known_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    config_file_list = (args.config_files.strip().split(",") if args.config_files else None)
    runtime_recorder = RunTimeRecorder(enabled=args.runtime_timer, csv_path=args.runtime_csv)

    mlen = cd['metric_len']
    thre = cd['metirc_threshold']
    task = cd['task_name']


    def run_multi_time(model_name, dataset_name, cd, save_name=None):
        save_name = save_name or model_name
        if 'eval_mode' in cd and cd['eval_mode'] == 'single':
            save_dir = f"result/{task}_{dataset_name}_s_{mlen}_{thre}.csv"
        else:
            save_dir = f"result/{task}_{dataset_name}_{mlen}_{thre}.csv"
        rs = result_saver(task, save_dir, ['MAE', 'RMSE', 'MAPE'], mlen, thre)
        for j in range(args.run_times):
            logging.info(f'Training {model_name} on {dataset_name} dataset')
            if TEST_MODE:
                with runtime_recorder.track(model_name, dataset_name, j + 1, args.run_times, save_name) as timer:
                    model = get_trainer(model_name, dataset_name, cd)
                    timer.set_backbone(model.config.backbone)
                    r = model.run()
                rs.save_result(r, rs.collect_run_info(model), f'{save_name}_{model.config.backbone}')
            else:
                try:
                    with runtime_recorder.track(model_name, dataset_name, j + 1, args.run_times, save_name) as timer:
                        model = get_trainer(model_name, dataset_name, cd)
                        timer.set_backbone(model.config.backbone)
                        r = model.run()
                    rs.save_result(r, rs.collect_run_info(model), f'{save_name}_{model.config.backbone}')
                except Exception as e:
                    import traceback
                    error_msg = f"\n{'=' * 50}\n"
                    error_msg += f"Error occurred at: {logging.time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    error_msg += f"Model: {model_name}, Dataset: {dataset_name}, Parameter: {cd}, Run: {j + 1}\n"
                    error_msg += f"Error Details:\n{traceback.format_exc()}"
                    error_msg += f"{'=' * 50}\n"
                    logging.error(f"Failed to run {model_name} on {dataset_name}. Check error.log for details.")
                    with open("error.log", "a", encoding="utf-8") as f:
                        f.write(error_msg)
                    continue
        try:
            rs.save_average()
        except:
            pass


    def run_on_dataset(dataset_name):
        if args.debug_mode[0] == 1:
            for m in args.model_list:
                run_multi_time(m, dataset_name, cd)
        elif args.debug_mode[2] == 1:
            for j, p in enumerate(param_dict):
                if isinstance(param_name, list):
                    for j in range(len(param_name)):
                        cd[param_name[j]] = p[j]
                else:
                    cd[param_name] = p
                run_multi_time(args.model, dataset_name, cd, f'{param_name}={p}')
        else:
            run_multi_time(args.model, dataset_name, cd, save_name)


    if args.debug_mode[1] == 1:
        for d in args.data_list:
            run_on_dataset(d)
    else:
        run_on_dataset(args.dataset)
