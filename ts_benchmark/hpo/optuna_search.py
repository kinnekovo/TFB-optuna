import copy
import json
import os
import re
from typing import List

import optuna

from ts_benchmark.common.constant import CONFIG_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend

from .search_space import sample_params


WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _resolve_config_path(config_path: str) -> str:
    """
    Resolve config path from either:
    - absolute POSIX path
    - absolute Windows path (e.g. D:\\foo\\bar.json)
    - path relative to CONFIG_PATH
    """
    if os.path.isabs(config_path) or WINDOWS_ABS_PATH_RE.match(config_path):
        return config_path
    return os.path.join(CONFIG_PATH, config_path)


def evaluate_params(
    params, config_data, data_name_list, model_name, strategy_args, save_path
):
    # 每个 trial 使用配置副本，避免多个 trial 之间相互污染。
    data_config = copy.deepcopy(config_data["data_config"])
    model_config = copy.deepcopy(config_data["model_config"])
    evaluation_config = copy.deepcopy(config_data["evaluation_config"])

    # 使用传入的数据集名称列表运行 HPO。
    data_config["data_name_list"] = data_name_list
    evaluation_config["save_path"] = save_path

    model_config["models"] = [{
        "adapter": None,
        "model_name": model_name,
        "model_hyper_params": params
    }]

    # 进行模型训练并评估
    try:
        log_files = pipeline(data_config, model_config, evaluation_config)
    except Exception as e:
        print(f"Error: {e}")
        return float("inf")

    # 假设我们从日志文件中提取损失值（这里我们假设返回0.0作为占位符）
    return 0.0  # 在这里替换为实际计算的验证损失值


def run_optuna_search(config_path: str, data_name_list: List[str], model_name: str, save_path: str, n_trials: int = 10,
                      seed: int = None):
    # 加载配置文件（支持绝对路径、Windows 绝对路径、相对于 config 目录的路径）
    resolved_config_path = _resolve_config_path(config_path)
    with open(resolved_config_path, "r") as f:
        config_data = json.load(f)

    # HPO 流程依赖 ParallelBackend，全局初始化后使用串行后端以避免额外依赖。
    ParallelBackend().init(backend="sequential")
    try:
        study = optuna.create_study(direction="minimize", study_name="hyperparameter_optimization")
        study.optimize(
            lambda trial: evaluate_params(
                sample_params(model_name, trial),
                config_data,
                data_name_list,
                model_name,
                {},
                save_path,
            ),
            n_trials=n_trials)
    finally:
        ParallelBackend().close(force=True)

    # 保存最优超参数
    best_params = study.best_params
    best_value = study.best_value

    best_params_json = {
        "model_name": model_name,
        "series_name": data_name_list[0],  # 假设只有一个数据集
        "objective": "val_loss",  # 假设优化目标是 val_loss
        "best_value": best_value,
        "best_params": best_params
    }

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, f"{model_name}_{data_name_list[0]}_best_params.json"), "w") as f:
        json.dump(best_params_json, f, indent=2)

    return best_params_json
