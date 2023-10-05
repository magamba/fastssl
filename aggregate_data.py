# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import sys
import logging

plots_path = "./plots"
name = 'ssl'

global missing_runs
missing_runs = []

try:
    root_dir = os.path.join(
        os.environ['SAVE_DIR'],
        name
    )
except KeyError:
    print("Please run setup_env first!")
    sys.exit(1)


def parse_stats(fname, stats, metric, epoch):
    try:
        metric_vals = stats[str(metric)]
    except KeyError:
        logger.info(f"Metric {metric} not found in {fname}")
        missing_runs.append(fname)
        return 0

    if metric not in figure1_conf["performance_metrics"]:
        keys = [ i[0] for i in metric_vals ]
        vals = [ i[1] for i in metric_vals ]
        try:
            key = keys.index(epoch)
            val = vals[key]
        except ValueError:
            logger.info(f"Epoch {epoch} missing for metric {metric} in {fname}")
            missing_runs.append(fname)
            val = 0
    else:
        val = metric_vals[-1]
    return val


""" Figure 1

    SSL pretraining: alpha, feature jacobian, effective rank
    
    algorithms: barlow_twins, byol
    models: resnet18
    datasets: cifar10, stl10
    epoch: 100 (barlow_twins), 300 (byol)
    
    train_configs: ssl, linear
    widths
    seeds
    noise configs
    metrics
"""

figure1_conf = {
    "algorithms": ["barlow_twins", "byol"],
    "base_models": ["resnet18"],
    "widths": list(range(1,65)),
    "seeds" : [0, 1, 2],
    "epochs": {
        "barlow_twins": 100,
        "byol": 300,
        "linear": 200,
    },
    "noise_configs": [0, 5, 10, 15, 20, 40, 60, 80, 100],
    "datasets": ["cifar10", "stl10"],
    "performance_metrics": ["train_acc_1", "train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored", "test_acc_1"],
    "metrics": ["alpha", "feature_input_jacobian", "effective_rank"],
}

figure1_conf.update({
    "filenames": {
        "barlow_twins": {
            dataset: {
                width: [ f"{root_dir}_barlow_twins-{dataset}/resnet18/width{width}/2_augs/lambd_0.005000_pdim_{32 * width}_lr_0.001_wd_1e-05/results_{dataset}_alpha_ssl_100_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]
            } for dataset in figure1_conf["datasets"]
        },
        "byol": {
            dataset: {
                width: [ f"{root_dir}_byol-{dataset}/resnet18/width{width}/2_augs/lambd_0.007812_pdim_{32 * width}_no_autocast_lr_0.001_wd_1e-06/results_{dataset}_alpha_ssl_300_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]
            } for dataset in figure1_conf["datasets"]
        },
    },
})

figure1_conf.update({
    "performance_filenames": {
        "barlow_twins": {
            dataset: {
                0: {
                    width: [ f"{root_dir}_barlow_twins-{dataset}/resnet18/width{width}/2_augs/lambd_0.005000_pdim_{32 * width}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]
                },
                **{noise: {
                    width: [ f"{root_dir}_barlow_twins_noise{noise}-{dataset}/resnet18/width{width}/2_augs/lambd_0.005000_pdim_{32 * width}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]
                } for noise in figure1_conf["noise_configs"][1:]}
            } for dataset in figure1_conf["datasets"]
        },
        "byol": {
            dataset: {
                0 : {
                    width: [ f"{root_dir}_byol-{dataset}/resnet18/width{width}/2_augs/lambd_0.007812_pdim_{width * 32}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]
                },
                **{noise: {
                    width: [ f"{root_dir}_byol_noise{noise}-{dataset}/resnet18/width{width}/2_augs/lambd_0.007812_pdim_{width * 32}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]
                } for noise in figure1_conf["noise_configs"][1:]}
            } for dataset in figure1_conf["datasets"]
        },
    },
})



missing_runs = []
def aggregate_data(destdir=plots_path):
    """ Aggregate results for figure 1
    """
    nseeds = len(figure1_conf["seeds"])
    nmetrics = len(figure1_conf["metrics"])
    nwidths = len(figure1_conf["widths"])
    nnoise = len(figure1_conf["noise_configs"])
    figure1_data = figure1_conf

    plot_data = {}
    for algorithm in figure1_conf["algorithms"]:
        plot_data[algorithm] = {}
        for dataset in figure1_conf["datasets"]:
            plot_data[algorithm][dataset] = {}
            for base_model in figure1_conf["base_models"]:
                plot_data[algorithm][dataset][base_model] = np.zeros((nwidths, nmetrics, nseeds))
                for w_id, width in enumerate(figure1_conf["widths"]):
                    for s_id in range(nseeds):
                        fname = figure1_conf["filenames"][algorithm][dataset][width][s_id]
                        logger.info(f"Loading {fname}")
                        try:
                            stats = np.load(
                                fname,
                                allow_pickle=True
                            ).tolist()
                            for m_id, metric in enumerate(figure1_conf["metrics"]):
                                epoch = figure1_conf["epochs"][algorithm]
                                logger.info(f"Parsing {base_model}_{width} epoch {epoch} {metric} seed {s_id}")
                                plot_data[algorithm][dataset][base_model][w_id, m_id, s_id] = parse_stats(fname, stats, metric, epoch)
                                
                        except FileNotFoundError:
                            logger.info(f"File not found: {fname}")
                            missing_runs.append(fname)
                            
                plot_data[algorithm][dataset][base_model] = \
                    plot_data[algorithm][dataset][base_model].tolist()
        
    figure1_data.pop("filenames")
    figure1_data["plot_data"] = plot_data
    
    nmetrics = len(figure1_conf["performance_metrics"])
    performance_data = {}
    for algorithm in figure1_conf["algorithms"]:
        plot_data[algorithm] = {}
        epoch = figure1_conf["epochs"]["linear"]
        for dataset in figure1_conf["datasets"]:
            plot_data[algorithm][dataset] = {}
            for base_model in figure1_conf["base_models"]:
                plot_data[algorithm][dataset][base_model] = np.zeros((nnoise, nwidths, nmetrics, nseeds))
                for n_id, noise in enumerate(figure1_conf["noise_configs"]):
                    for w_id, width in enumerate(figure1_conf["widths"]):
                        for s_id in range(nseeds):
                            fname = figure1_conf["performance_filenames"][algorithm][dataset][noise][width][s_id]
                            logger.info(f"Loading {fname}")
                            try:
                                stats = np.load(
                                    fname,
                                    allow_pickle=True
                                ).tolist()
                                for m_id, metric in enumerate(figure1_conf["performance_metrics"]):
                                    if noise == 0 and metric in ["train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored"]: continue
                                    logger.info(f"Parsing {base_model}_{width} epoch {epoch} {metric} seed {s_id}")
                                    plot_data[algorithm][dataset][base_model][n_id, w_id, m_id, s_id] = parse_stats(fname, stats, metric, epoch)
                                
                            except FileNotFoundError:
                                logger.info(f"File not found: {fname}")
                                missing_runs.append(fname)
                            
                plot_data[algorithm][dataset][base_model] = \
                    plot_data[algorithm][dataset][base_model].tolist()
    
    figure1_data.pop("performance_filenames")
    figure1_data["performance_data"] = performance_data

    if len(missing_runs) > 0:
        logger.info("The following files were missing:")
        for f in missing_runs:
            logger.info(f)

    filename = os.path.join(destdir, 'plot_data.json')
    logger.info(f"Saving plot data to {filename}")
    with open(filename, 'w') as fp:
        json.dump(figure1_data, fp, allow_nan=False)


"""Main
"""

def aggregate_stats(fig_id):
    logger.info(f"Aggregating data for figure {fig_id}")
    
    if fig_id == 1:
        destdir = os.path.join(plots_path, 'figure1')
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        aggregate_data(destdir)
    else:
        raise ValueError(f"Invalid figure id {fig_id}")


def init_logging(logger_name, logfile, log_level: str):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    f_handler = logging.FileHandler(logfile)
    f_handler.setLevel(numeric_level)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(numeric_level)
    f_handler.setFormatter(formatter)
    c_handler.setFormatter(formatter)
    
    logging.basicConfig(level=numeric_level, handlers=[f_handler, c_handler])
    logger = logging.getLogger(logger_name)
    return logger


def main():

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    logfile = os.path.join(plots_path, 'aggregate_data.log')
    init_logging(None, logfile, 'info')
    
    global logger
    logger = logging.getLogger()

    for fig_id in [1,]:
        aggregate_stats(fig_id)


if __name__ == '__main__':
    main()