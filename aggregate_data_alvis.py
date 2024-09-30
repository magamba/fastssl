# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import sys
import logging

from fastssl.utils.powerlaw import rankme

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


def parse_stats(fname, stats, metric, epoch, ood=False):
    try:
        metric_vals = stats[str(metric)]
    except KeyError:
        logger.info(f"Metric {metric} not found in {fname}")
        missing_runs.append(fname)
        return 0

    if metric not in [
        "train_acc_1", "train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored", 
        "test_acc_1", "train_loss", "train_loss_on_diag", "train_loss_off_diag", "test_loss", 
        "test_loss_on_diag", "test_loss_off_diag"
    ]:
        if metric == "rankme":
            try:
                val = metric_vals.item()
                return val
            except AttributeError:
                pass
        if metric in ["inter_manifold_eigen", "intra_manifold_eigen", "inter_manifold_gen_eigen", "intra_manifold_gen_eigen"]:
            return rankme(metric_vals[-1])
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
        
        
"""Figure 1 -- Model scaling w/ dataset scaling and aug scaling
"""

figure1_conf = {
    "algorithms": ["barlow_twins", "simclr"],
    "base_models": ["resnet18"],
    "expansion": {
        "resnet18": [32,],
        "vit": [6,],
    },
    "model_strings": {
        "resnet18": ["resnet18/"],
        "vit": ["vit_"],
    },
    "widths": {
        "resnet18": {
            "cifar10": list(range(8,65,4)),
        },
        "vit": {
            "cifar10": list(range(8,65,4)),
        },
    },
    "projection_depths": [2],
    "hyperparams": {
        "barlow_twins": {
            width: [ 1. / (width * 32) ] for width in list(range(8,65,4)) # [0.005, ], # 0.02 ],
        },
        "simclr": {
            width: [0.1,] for width in range(8,65,4)
        },
    },
    "seeds" : [0,],
    "epochs": {
        "barlow_twins": 100,
        "simclr": 100,
        "linear": 200,
    },
    "augs": {
        "barlow_twins": {
            "cifar100": [2, 4, 8, 16],
            "imagenet100": [2, 4, 8],
        }
        "simclr": {
            "cifar10": [2, 4, 8, 16],
            "cifar100": [2, 4, 8, 16],
            "imagenet100": [2, 4, 8],
        }
    },
    "noise_configs": [0, 10, 20, 40, 60, 80, 100],
    "nsamples":  [0.002, 0.004, 0.00768, 0.01, 0.02, 0.04096, 0.2, 0.4, 0.6, 0.8, 1.0],
    "nsamples_strings": {
        nsamples: [f"-nsamples_{nsamples}" if nsamples != 1.0 else ""] for nsamples in [0.002, 0.004, 0.00768, 0.01, 0.02, 0.04096, 0.2, 0.4, 0.6, 0.8, 1.0]
    },
    "batch_sizes": {
        nsamples: [ round(nsamples * 50000) if round(nsamples * 50000) < 512 else 512 ] for nsamples in [0.002, 0.004, 0.00768, 0.01, 0.02, 0.04096, 0.2, 0.4, 0.6, 0.8, 1.0]
    },
    "datasets": {
        "barlow_twins": ["cifar100","imagenet100"],
        "simclr": ["cifar10","cifar100","imagenet100"],
    },
    "performance_metrics": ["train_acc_1", "train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored", "test_acc_1"],
    "ood_metrics": ["test_acc_1", "test_loss_ssl"],
    "ood_noise_levels": list(range(1,6)),
    "ood_noise_types": [
        "frost",
        "glass_blur",
        "spatter",
        "gaussian_blur",
        "impulse_noise",
        "motion_blur",
        "shot_noise",
        "speckle_noise",
        "fog",
        "gaussian_noise",
        "jpeg_compression",
        "pixelate",
        "snow",
    ],
    "metrics": ["train_loss", "test_loss", "alpha", "feature_input_jacobian", "rankme", "intra_manifold_eigen", "inter_manifold_eigen", "intra_manifold_gen_eigen", "inter_manifold_gen_eigen"],
}

figure1_conf.update({
    "filenames": {
        "barlow_twins": {
            dataset: {
                model: {
                    augs: {
                        nsamples: {
                            pdepth: {
                                width: {
                                    hparam: [ f"{root_dir}_barlow_twins_robustness_inverse_scaling-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/lambd_{hparam:.6f}_pdim_{base_width * width}_pdepth_{pdepth}_lr_0.001_wd_1e-05/results_{dataset}_alpha_ssl_100_seed_{seed}.npy"  for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for seed in figure1_conf["seeds"]
                                    ] for hparam in figure1_conf["hyperparams"]["barlow_twins"][width]
                                } for width in figure1_conf["widths"][model][dataset]
                            } for pdepth in figure1_conf["projection_depths"]
                        } for nsamples in figure1_conf["nsamples"]
                    } for augs in figure1_conf["augs"]["barlow_twins"][dataset]
                } for model in figure1_conf["base_models"]
            } for dataset in figure1_conf["datasets"]["barlow_twins"]
        },
        "simclr": {
            dataset: {
                model: {
                    augs: {
                        nsamples: {
                            pdepth: {
                                width: {
                                    hparam: [ f"{root_dir}_simclr_robustness-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/temp_{hparam:.3f}_pdim_{base_width * width}_pdepth_{pdepth}_bsz_{bsz}_lr_0.001_wd_1e-05/results_{dataset}_alpha_SimCLR_100_seed_{seed}.npy"  for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for seed in figure1_conf["seeds"] for bsz in figure1_conf["batch_sizes"][nsamples]
                                    ] for hparam in figure1_conf["hyperparams"]["simclr"][width]
                                } for width in figure1_conf["widths"][model][dataset]
                            } for pdepth in figure1_conf["projection_depths"]
                        } for nsamples in figure1_conf["nsamples"]
                    } for augs in figure1_conf["augs"]["simclr"][dataset]
                } for model in figure1_conf["base_models"]
            } for dataset in figure1_conf["datasets"]["simclr"]
        },
    },
})

figure1_conf.update({
    "performance_filenames": {
        "barlow_twins": {
            dataset: {
                model: {
                    augs: {
                        nsamples: {
                            pdepth: {
                                width: {
                                    hparam: {
                                        0: [ f"{root_dir}_barlow_twins_robustness_inverse_scaling-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/lambd_{hparam:.6f}_pdim_{base_width * width}_pdepth_{pdepth}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for seed in figure1_conf["seeds"] ],
                                    } for hparam in figure1_conf["hyperparams"]["barlow_twins"][width]
                                } for width in figure1_conf["widths"][model][dataset]
                            } for pdepth in figure1_conf["projection_depths"]
                        } for nsamples in figure1_conf["nsamples"]
                    } for augs in figure1_conf["augs"]["barlow_twins"][dataset]
                } for model in figure1_conf["base_models"]
            } for dataset in figure1_conf["datasets"]["barlow_twins"]
        },
        "simclr": {
            dataset: {
                model: {
                    augs: {
                        nsamples: {
                            pdepth: {
                                width: {
                                    hparam: {
                                        0: [ f"{root_dir}_simclr_robustness-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/temp_{hparam:.3f}_pdim_{base_width * width}_pdepth_{pdepth}_bsz_{bsz}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for bsz in figure1_conf["batch_sizes"][nsamples] for seed in figure1_conf["seeds"] ],
                                    } for hparam in figure1_conf["hyperparams"]["simclr"][width]
                                } for width in figure1_conf["widths"][model][dataset]
                            } for pdepth in figure1_conf["projection_depths"]
                        } for nsamples in figure1_conf["nsamples"]
                    } for augs in figure1_conf["augs"]["simclr"][dataset]
                } for model in figure1_conf["base_models"]
            } for dataset in figure1_conf["datasets"]["simclr"]
        },
    },
})

figure1_conf.update({
    "ood_filenames": {
        "barlow_twins": {
            dataset: {
                model: {
                    augs: {
                        nsamples: {
                            pdepth: {
                                width: {
                                    hparam: {
                                        0: [ f"{root_dir}_barlow_twins_robustness_inverse_scaling-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/lambd_{hparam:.6f}_pdim_{base_width * width}_pdepth_{pdepth}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for seed in figure1_conf["seeds"] ],
                                        **{noise: [ f"{root_dir}_barlow_twins_robustness_inverse_scaling-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/lambd_{hparam:.6f}_pdim_{base_width * width}_pdepth_{pdepth}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}c_{noise}_ood_eval_linear_200_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for seed in figure1_conf["seeds"] ] for noise in figure1_conf["ood_noise_types"] }
                                    } for hparam in figure1_conf["hyperparams"]["barlow_twins"][width]
                                } for width in figure1_conf["widths"][model][dataset]
                            } for pdepth in figure1_conf["projection_depths"]
                        } for nsamples in figure1_conf["nsamples"]
                    } for augs in figure1_conf["augs"]["barlow_twins"][dataset]
                } for model in figure1_conf["base_models"]
            } for dataset in figure1_conf["datasets"]["barlow_twins"]
        },
        "simclr": {
            dataset: {
                model: {
                    augs: {
                        nsamples: {
                            pdepth: {
                                width: {
                                    hparam: {
                                        0: [ f"{root_dir}_simclr_robustness-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/temp_{hparam:.3f}_pdim_{base_width * width}_pdepth_{pdepth}_bsz_{bsz}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for bsz in figure1_conf["batch_sizes"][nsamples] for seed in figure1_conf["seeds"] ],
                                        **{noise: [ f"{root_dir}_simclr_robustness-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/temp_{hparam:.3f}_pdim_{base_width * width}_pdepth_{pdepth}_bsz_{bsz}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}c_{noise}_ood_eval_linear_200_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for bsz in figure1_conf["batch_sizes"][nsamples] for seed in figure1_conf["seeds"] ] for noise in figure1_conf["ood_noise_types"] }
                                    } for hparam in figure1_conf["hyperparams"]["simclr"][width]
                                } for width in figure1_conf["widths"][model][dataset]
                            } for pdepth in figure1_conf["projection_depths"]
                        } for nsamples in figure1_conf["nsamples"]
                    } for augs in figure1_conf["augs"]["simclr"][dataset]
                } for model in figure1_conf["base_models"]
            } for dataset in figure1_conf["datasets"]["simclr"]
        },
    },
})

figure1_conf.update({
    "ood_ssl_eval_filenames": {
        "barlow_twins": {
            dataset: {
                model: {
                    augs: {
                        nsamples: {
                            pdepth: {
                                width: {
                                    hparam: {
                                        0: [ f"{root_dir}_barlow_twins_robustness_inverse_scaling-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/lambd_{hparam:.6f}_pdim_{base_width * width}_pdepth_{pdepth}_lr_0.001_wd_1e-05/results_{dataset}_ssl_eval_ssl_100_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for seed in figure1_conf["seeds"] ],
                                        **{noise: [ f"{root_dir}_barlow_twins_robustness_inverse_scaling-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/lambd_{hparam:.6f}_pdim_{base_width * width}_pdepth_{pdepth}_lr_0.001_wd_1e-05/results_{dataset}_{noise}_ssl_eval_ssl_100_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for seed in figure1_conf["seeds"] ] for noise in figure1_conf["ood_noise_types"] }
                                    } for hparam in figure1_conf["hyperparams"]["barlow_twins"][width]
                                } for width in figure1_conf["widths"][model][dataset]
                            } for pdepth in figure1_conf["projection_depths"]
                        } for nsamples in figure1_conf["nsamples"]
                    } for augs in figure1_conf["augs"]["barlow_twins"][dataset]
                } for model in figure1_conf["base_models"]
            } for dataset in figure1_conf["datasets"]["barlow_twins"]
        },
        "simclr": {
            dataset: {
                model: {
                    augs: {
                        nsamples: {
                            pdepth: {
                                width: {
                                    hparam: {
                                        0: [ f"{root_dir}_simclr_robustness-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/temp_{hparam:.3f}_pdim_{base_width * width}_pdepth_{pdepth}_bsz_{bsz}_lr_0.001_wd_1e-05/results_{dataset}_ssl_eval_SimCLR_100_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for bsz in figure1_conf["batch_sizes"][nsamples] for seed in figure1_conf["seeds"] ],
                                        **{noise: [ f"{root_dir}_simclr_robustness-{dataset}{nsample_str}/{model_str}width{width}/{augs}_augs/temp_{hparam:.3f}_pdim_{base_width * width}_pdepth_{pdepth}_bsz_{bsz}_lr_0.001_wd_1e-05/results_{dataset}_{noise}_ssl_eval_SimCLR_100_seed_{seed}.npy" for model_str in figure1_conf["model_strings"][model] for base_width in figure1_conf["expansion"][model] for nsample_str in figure1_conf["nsamples_strings"][nsamples] for bsz in figure1_conf["batch_sizes"][nsamples] for seed in figure1_conf["seeds"] ] for noise in figure1_conf["ood_noise_types"] }
                                    } for hparam in figure1_conf["hyperparams"]["simclr"][width]
                                } for width in figure1_conf["widths"][model][dataset]
                            } for pdepth in figure1_conf["projection_depths"]
                        } for nsamples in figure1_conf["nsamples"]
                    } for augs in figure1_conf["augs"]["simclr"][dataset]
                } for model in figure1_conf["base_models"]
            } for dataset in figure1_conf["datasets"]["simclr"]
        },
    },
})


missing_runs = []
def aggregate_fig1(destdir=plots_path):
    """ Aggregate results for figure 1
    """
    nseeds = len(figure1_conf["seeds"])
    nmetrics = len(figure1_conf["metrics"])
    nnoise = 1 # len(figure1_conf["noise_configs"])
    npdepths = len(figure1_conf["projection_depths"])
    nnsamples = len(figure1_conf["nsamples"])
    covariance_metrics = [
        "intra_manifold_eigen", "inter_manifold_eigen", "intra_manifold_gen_eigen", "inter_manifold_gen_eigen"
    ]
    ssl_eval_metrics = ["test_loss"]
    
    metrics_to_skip = [
        "test_loss", "alpha", "feature_input_jacobian", "rankme", "intra_manifold_eigen", 
        "inter_manifold_eigen", "intra_manifold_gen_eigen", "inter_manifold_gen_eigen"
    ]
    
    figure1_data = figure1_conf

    plot_data = {}
    for algorithm in figure1_conf["algorithms"]:
        plot_data[algorithm] = {}
        for dataset in figure1_conf["datasets"][algorithm]:
            plot_data[algorithm][dataset] = {}
            naugs = len(figure1_conf["augs"][algorithm][dataset])
            nhparams = 1 # len(figure1_conf["hyperparams"][algorithm])
            for base_model in figure1_conf["base_models"]:
                nwidths = len(figure1_conf["widths"][base_model][dataset])
                plot_data[algorithm][dataset][base_model] = np.zeros((naugs, nnsamples, npdepths, nwidths, nhparams, nmetrics, nseeds))
                for a_id, augs in enumerate(figure1_conf["augs"][algorithm][dataset]):
                    for n_id, nsamples in enumerate(figure1_conf["nsamples"]):
                        for d_id, pdepth in enumerate(figure1_conf["projection_depths"]):
                            for w_id, width in enumerate(figure1_conf["widths"][base_model][dataset]):
                                for h_id, hparam in enumerate(figure1_conf["hyperparams"][algorithm][width]):
                                    h_id = 0
                                    for s_id in range(nseeds):
                                        fname = figure1_conf["filenames"][algorithm][dataset][base_model][augs][nsamples][pdepth][width][hparam][s_id]
                                        logger.info(f"Loading {fname}")
                                        try:
                                            stats = np.load(
                                                fname,
                                                allow_pickle=True
                                            ).tolist()
                                            metric_ids = []
                                            for m_id, metric in enumerate(figure1_conf["metrics"]):
                                                if metric in metrics_to_skip: continue
                                                if metric in covariance_metrics or metric in ssl_eval_metrics:
                                                    metric_ids.append((m_id, metric))
                                                    continue
                                                epoch = figure1_conf["epochs"][algorithm]
                                                logger.info(f"Parsing {base_model}_{width} {augs} augs nsamples {nsamples} pdepth {pdepth} hparam {hparam} epoch {epoch} {metric} seed {s_id}")
                                                plot_data[algorithm][dataset][base_model][a_id, n_id, d_id, w_id, h_id, m_id, s_id] = parse_stats(fname, stats, metric, epoch)
                                                
                                        except FileNotFoundError:
                                            logger.info(f"File not found: {fname}")
                                            missing_runs.append(fname)
                                            
                                        for (met_id, metric) in metric_ids:
                                            file_dict = "ssl_eval_filenames" if metric in ssl_eval_metrics else "covariance_filenames"
                                            fname = figure1_conf[file_dict][algorithm][dataset][base_model][augs][nsamples][pdepth][width][hparam][s_id]
                                            logger.info(f"Loading {fname}")
                                            try:
                                                stats = np.load(
                                                    fname,
                                                    allow_pickle=True
                                                ).tolist()
                                                epoch = figure1_conf["epochs"][algorithm]
                                                logger.info(f"Parsing {base_model}_{width} {augs} augs nsamples {nsamples} pdepth {pdepth} hparam {hparam} epoch {epoch} {metric} seed {s_id}")
                                                plot_data[algorithm][dataset][base_model][a_id, n_id, d_id, w_id, h_id, met_id, s_id] = parse_stats(fname, stats, metric, epoch)
                                                    
                                            except FileNotFoundError:
                                                logger.info(f"File not found: {fname}")
                                                missing_runs.append(fname)

                plot_data[algorithm][dataset][base_model] = \
                    np.nan_to_num(plot_data[algorithm][dataset][base_model]).tolist()
        
    figure1_data.pop("filenames")
    figure1_data["plot_data"] = plot_data
    
    nmetrics = len(figure1_conf["performance_metrics"])
    performance_data = {}
    for algorithm in figure1_conf["algorithms"]:
        performance_data[algorithm] = {}
        epoch = figure1_conf["epochs"]["linear"]
        for dataset in figure1_conf["datasets"][algorithm]:
            performance_data[algorithm][dataset] = {}
            naugs = len(figure1_conf["augs"][algorithm][dataset])
            nhparams = 1 # len(figure1_conf["hyperparams"][algorithm])
            for base_model in figure1_conf["base_models"]:
                nwidths = len(figure1_conf["widths"][base_model][dataset])
                performance_data[algorithm][dataset][base_model] = np.zeros((naugs, nnoise, nnsamples, npdepths, nwidths, nhparams, nmetrics, nseeds))
                for a_id, augs in enumerate(figure1_conf["augs"][algorithm][dataset]):
                    for n_id, noise in enumerate(figure1_conf["noise_configs"]):
                        if n_id > 0: continue
                        for ns_id, nsamples in enumerate(figure1_conf["nsamples"]):
                            for p_id, pdepth in enumerate(figure1_conf["projection_depths"]):
                                for w_id, width in enumerate(figure1_conf["widths"][base_model][dataset]):
                                    for h_id, hparam in enumerate(figure1_conf["hyperparams"][algorithm][width]):
                                        h_id = 0
                                        for s_id in range(nseeds):
                                            fname = figure1_conf["performance_filenames"][algorithm][dataset][base_model][augs][nsamples][pdepth][width][hparam][noise][s_id]
                                            logger.info(f"Loading {fname}")
                                            try:
                                                stats = np.load(
                                                    fname,
                                                    allow_pickle=True
                                                ).tolist()
                                                for m_id, metric in enumerate(figure1_conf["performance_metrics"]):
                                                    if noise == 0 and metric in ["train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored"]: continue
                                                    logger.info(f"Parsing {base_model}_{width} {augs} augs nsamples {nsamples} epoch {epoch} {metric} seed {s_id}")
                                                    performance_data[algorithm][dataset][base_model][a_id, n_id, ns_id, p_id, w_id, h_id, m_id, s_id] = parse_stats(
                                                        fname, stats, metric, epoch
                                                    )
                                                
                                            except FileNotFoundError:
                                                logger.info(f"File not found: {fname}")
                                                missing_runs.append(fname)
                                
                performance_data[algorithm][dataset][base_model] = \
                    performance_data[algorithm][dataset][base_model].tolist()
    
    figure1_data["performance_data"] = performance_data
    
    nmetrics = len(figure1_conf["ood_metrics"])
    ood_data = {}
    nnoise = len(figure1_conf["ood_noise_types"]) +1
    nlevels = len(figure1_conf["ood_noise_levels"])
    for algorithm in figure1_conf["algorithms"]:
        ood_data[algorithm] = {}
        for dataset in figure1_conf["datasets"][algorithm]:
            ood_data[algorithm][dataset] = {}
            naugs = len(figure1_conf["augs"][algorithm][dataset])
            nhparams = 1 # len(figure1_conf["hyperparams"][algorithm])
            for base_model in figure1_conf["base_models"]:
                nwidths = len(figure1_conf["widths"][base_model][dataset])
                ood_data[algorithm][dataset][base_model] = np.zeros((naugs, nnoise, nnsamples, npdepths, nwidths, nhparams, nmetrics, nseeds, nlevels))
                for a_id, augs in enumerate(figure1_conf["augs"][algorithm]):
                    for n_id, noise in enumerate([0] + figure1_conf["ood_noise_types"]):
                        for ns_id, nsamples in enumerate(figure1_conf["nsamples"]):
                            for p_id, pdepth in enumerate(figure1_conf["projection_depths"]):
                                for w_id, width in enumerate(figure1_conf["widths"][base_model][dataset]):
                                    for h_id, hparam in enumerate(figure1_conf["hyperparams"][algorithm]):
                                        h_id = 0
                                        for s_id in range(nseeds):
                                            file_dict = "performance_filenames" if n_id == 0 else "ood_filenames"
                                            fname = figure1_conf[file_dict][algorithm][dataset][base_model][augs][nsamples][pdepth][width][hparam][noise][s_id]
                                            logger.info(f"Loading {fname}")
                                            try:
                                                stats = np.load(
                                                    fname,
                                                    allow_pickle=True
                                                ).tolist()
                                                metric_ids = []
                                                for m_id, metric in enumerate(figure1_conf["ood_metrics"]):
                                                    if metric in ["test_loss_ssl"]:
                                                        continue # skipping for now
                                                        metric_ids.append((m_id, metric))
                                                        continue
                                                    if noise == 0 and metric in ["train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored"]: continue
                                                    logger.info(f"Parsing noise {noise} {base_model}_{width} {augs} augs nsamples {nsamples} epoch {epoch} {metric} seed {s_id}")
                                                    if n_id == 0:
                                                        ood_data[algorithm][dataset][base_model][a_id, n_id, ns_id, p_id, w_id, h_id, m_id, s_id, :] = parse_stats(
                                                            fname, stats, metric, epoch
                                                        )
                                                    else:
                                                        ood_data[algorithm][dataset][base_model][a_id, n_id, ns_id, p_id, w_id, h_id, m_id, s_id] = parse_stats(
                                                            fname, stats, metric, epoch
                                                        )
                                                
                                            except FileNotFoundError:
                                                logger.info(f"File not found: {fname}")
                                                missing_runs.append(fname)
                                                
                                            for (m_id, metric) in metric_ids:
                                                if metric == "test_loss_ssl": metric = "test_loss"
                                                fname = figure1_conf["ood_ssl_eval_filenames"][algorithm][dataset][base_model][augs][nsamples][pdepth][width][hparam][noise][s_id]
                                                logger.info(f"Loading {fname}")
                                                try:
                                                    stats = np.load(
                                                        fname,
                                                        allow_pickle=True
                                                    ).tolist()
                                                    logger.info(f"Parsing noise {noise} {base_model}_{width} {augs} augs nsamples {nsamples} epoch {epoch} {metric} seed {s_id}")
                                                    if n_id == 0:
                                                        ood_data[algorithm][dataset][base_model][a_id, n_id, ns_id, p_id, w_id, h_id, m_id, s_id, :] = parse_stats(
                                                            fname, stats, metric, epoch
                                                        )
                                                    else:
                                                        ood_data[algorithm][dataset][base_model][a_id, n_id, ns_id, p_id, w_id, h_id, m_id, s_id] = parse_stats(
                                                            fname, stats, metric, epoch
                                                        )
                                                except FileNotFoundError:
                                                    logger.info(f"File not found: {fname}")
                                                    missing_runs.append(fname)

                ood_data[algorithm][dataset][base_model] = \
                    ood_data[algorithm][dataset][base_model].tolist()

    figure1_data.pop("performance_filenames")
    figure1_data.pop("ood_filenames")
    figure1_data.pop("covariance_filenames")
    figure1_data.pop("ood_ssl_eval_filenames")
    figure1_data.pop("ssl_eval_filenames")
    figure1_data["ood_data"] = ood_data

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
        aggregate_fig1(destdir)
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

    for fig_id in [1]:
        aggregate_stats(fig_id)


if __name__ == '__main__':
    main()
