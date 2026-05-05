import copy
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Optional

import optuna

from ts_benchmark.common.constant import CONFIG_PATH, ROOT_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.recording import read_record_file
from ts_benchmark.utils.parallel import ParallelBackend

from .search_space import sample_params


WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
logger = logging.getLogger(__name__)


def _resolve_config_path(config_path: str) -> str:
    if os.path.isabs(config_path) or WINDOWS_ABS_PATH_RE.match(config_path):
        return config_path
    return os.path.join(CONFIG_PATH, config_path)


def _resolve_output_dir(save_path: str) -> str:
    if not save_path:
        return os.path.join(ROOT_PATH, "result")
    if os.path.isabs(save_path) or WINDOWS_ABS_PATH_RE.match(save_path):
        return save_path
    return os.path.join(ROOT_PATH, "result", save_path)

# 读取评估日志，找 mse、mse_norm、mae 等列，取均值作为 Optuna 要最小化的目标值
def _extract_objective_from_logs(log_files: List[str]) -> float:
    if not log_files:
        raise ValueError("No log files returned by pipeline.")
    df = read_record_file(log_files[0])
    preferred_metrics = [
        "mse",
        "mse_norm",
        "mae",
        "mae_norm",
        "rmse",
        "rmse_norm",
        "mape",
        "mape_norm",
    ]
    for metric in preferred_metrics:
        if metric in df.columns:
            return float(df[metric].mean())
    raise ValueError(f"No supported metric columns found in log file: {df.columns.tolist()}")


def _get_default_model_params(config_data: dict, model_name: str) -> dict:
    models = config_data.get("model_config", {}).get("models", [])
    for model in models:
        if model.get("model_name") == model_name:
            return copy.deepcopy(model.get("model_hyper_params") or {})
    # Fallback to recommended default params in config when models list is empty.
    return copy.deepcopy(
        config_data.get("model_config", {}).get("recommend_model_hyper_params", {}) or {}
    )


def _merge_params(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base or {})
    merged.update(copy.deepcopy(override or {}))
    return merged


def _cleanup_log_files(log_files: List[str]) -> None:
    for file_path in log_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            logger.warning("Failed to remove temporary log file: %s", file_path)


def evaluate_params(
    params, config_data, data_name_list, model_name, strategy_args, save_path, forecast_lengths: Optional[List[int]] = None, return_details: bool = False, eval_mode: str = "val"
):
    """
    Evaluate `params` by running the pipeline for each horizon in `forecast_lengths` and
    aggregating the objective (mean of per-horizon objectives).

    If `forecast_lengths` is None, try to read from `config_data["data_config"]` using
    keys like `forecast_lengths` or `prediction_length`. If still missing, default to [1].

    If `return_details` is True, return a tuple (aggregate_value, {horizon: value, ...}).
    """
    # Determine forecast lengths to evaluate
    if forecast_lengths is None:
        cfg_data = config_data.get("data_config", {})
        cfg_lengths = cfg_data.get("forecast_lengths") or cfg_data.get("forecast_length") or cfg_data.get("prediction_length")
        if isinstance(cfg_lengths, (list, tuple)):
            forecast_lengths = list(cfg_lengths)
        elif cfg_lengths is not None:
            try:
                forecast_lengths = [int(cfg_lengths)]
            except Exception:
                forecast_lengths = [1]
        else:
            forecast_lengths = [1]

    if not forecast_lengths:
        raise ValueError("forecast_lengths must be a non-empty list")

    per_horizon_values: Dict[int, float] = {}

    base_data_config = copy.deepcopy(config_data["data_config"])
    base_model_config = copy.deepcopy(config_data["model_config"])
    base_evaluation_config = copy.deepcopy(config_data["evaluation_config"])
    base_strategy_args = copy.deepcopy(base_evaluation_config.get("strategy_args", {}))

    for h in forecast_lengths:
        # 每个 trial 使用配置副本，避免多个 trial 之间相互污染。
        data_config = copy.deepcopy(base_data_config)
        model_config = copy.deepcopy(base_model_config)
        evaluation_config = copy.deepcopy(base_evaluation_config)

        if eval_mode not in {"val", "test"}:
            raise ValueError(f"Unsupported eval_mode: {eval_mode}")

        # 使用传入的数据集名称列表运行 HPO。
        data_config["data_name_list"] = data_name_list

        # 关键：forecast horizon 是由 evaluation_config.strategy_args.horizon 控制的，
        # 同时部分深度模型会从 model hyper-params 里的 horizon/pred_len 读取输出长度。
        strategy_args = copy.deepcopy(base_strategy_args)
        strategy_args["horizon"] = h
        if eval_mode == "val":
            # HPO 阶段仅在 val 上评估，避免使用 test 指标选参。
            strategy_args["hpo_eval_mode"] = "val"
        evaluation_config["strategy_args"] = strategy_args

        # 为每个 horizon 使用独立子目录，避免互相覆盖
        if save_path:
            evaluation_config["save_path"] = os.path.join(save_path, f"horizon_{h}")
        else:
            evaluation_config["save_path"] = None

        model_hyper_params = copy.deepcopy(params)
        model_hyper_params["horizon"] = h
        model_hyper_params["pred_len"] = h
        model_config["models"] = [{
            "adapter": None,
            "model_name": model_name,
            "model_hyper_params": model_hyper_params
        }]

        # 进行模型训练并评估
        log_files = pipeline(data_config, model_config, evaluation_config)
        try:
            objective = _extract_objective_from_logs(log_files)
        finally:
            _cleanup_log_files(log_files)

        per_horizon_values[h] = objective

    # 聚合策略：取均值（可根据需要改为加权平均等）
    aggregate_value = sum(per_horizon_values.values()) / len(per_horizon_values)

    if return_details:
        return aggregate_value, per_horizon_values
    return aggregate_value


def run_optuna_search(config_path: str, data_name_list: List[str], model_name: str, save_path: str, n_trials: int = 10,
                      seed: int = None, forecast_lengths: Optional[List[int]] = None):
    # 加载配置文件（支持绝对路径、Windows 绝对路径、相对于 config 目录的路径）
    resolved_config_path = _resolve_config_path(config_path)
    with open(resolved_config_path, "r") as f:
        config_data = json.load(f)
    output_dir = _resolve_output_dir(save_path)

    # 如果外部没有传入 forecast_lengths，尝试从配置文件读取
    if forecast_lengths is None:
        cfg_data = config_data.get("data_config", {})
        cfg_lengths = cfg_data.get("forecast_lengths") or cfg_data.get("forecast_length") or cfg_data.get("prediction_length")
        if isinstance(cfg_lengths, (list, tuple)):
            forecast_lengths = list(cfg_lengths)
        elif cfg_lengths is not None:
            try:
                forecast_lengths = [int(cfg_lengths)]
            except Exception:
                forecast_lengths = [1]
        else:
            forecast_lengths = [1]

    # HPO trial 输出写入临时目录，避免在最终结果目录产生大量中间日志文件。
    temp_eval_dir = tempfile.mkdtemp(prefix="hpo_eval_")

    # HPO 流程依赖 ParallelBackend，全局初始化后使用串行后端以避免额外依赖。
    ParallelBackend().init(backend="sequential")
    try:
        baseline_params = _get_default_model_params(config_data, model_name)
        baseline_value, baseline_per_horizon = evaluate_params(
            baseline_params,
            config_data,
            data_name_list,
            model_name,
            {},
            temp_eval_dir,
            forecast_lengths,
            True,
        )
        logger.info("Baseline objective value (aggregate): %.6f", baseline_value)

        study = optuna.create_study(direction="minimize", study_name="hyperparameter_optimization")
        study.optimize(
            lambda trial: evaluate_params(
                _merge_params(baseline_params, sample_params(model_name, trial)),
                config_data,
                data_name_list,
                model_name,
                {},
                temp_eval_dir,
                forecast_lengths,
            ),
            n_trials=n_trials)
    finally:
        ParallelBackend().close(force=True)
        shutil.rmtree(temp_eval_dir, ignore_errors=True)

    # 保存最优超参数
    best_params = study.best_params
    full_best_params = _merge_params(baseline_params, best_params)
    best_value = study.best_value

    # The ParallelBackend was closed after trial evaluation; ensure it's initialized
    # before we call pipeline() again for the best-params recheck.
    if ParallelBackend().backend is None:
        ParallelBackend().init(backend="sequential")
    temp_eval_dir2 = tempfile.mkdtemp(prefix="hpo_eval_best_")
    try:
        best_value_rechecked, best_per_horizon = evaluate_params(
            full_best_params,
            config_data,
            data_name_list,
            model_name,
            {},
            temp_eval_dir2,
            forecast_lengths,
            True,
        )

        final_test_value, final_test_per_horizon = evaluate_params(
            full_best_params,
            config_data,
            data_name_list,
            model_name,
            {},
            temp_eval_dir2,
            forecast_lengths,
            True,
            eval_mode="test",
        )
    finally:
        # Close the backend we (possibly) re-initialized and clean up temp files.
        ParallelBackend().close(force=True)
        shutil.rmtree(temp_eval_dir2, ignore_errors=True)

    best_params_full_per_horizon = {
        str(h): _merge_params(full_best_params, {"horizon": h, "pred_len": h})
        for h in forecast_lengths
    }

    best_params_json = {
        "model_name": model_name,
        "series_name": data_name_list[0] if data_name_list else None,
        "objective": "val_loss",
        "n_trials": n_trials,
        "forecast_lengths": forecast_lengths,
        "baseline_params": baseline_params,
        "baseline_value": baseline_value,
        "baseline_per_horizon": baseline_per_horizon,
        "best_value": best_value,
        "best_value_rechecked": best_value_rechecked,
        "best_per_horizon": best_per_horizon,
        "final_test_value": final_test_value,
        "final_test_per_horizon": final_test_per_horizon,
        "best_params": best_params,
        "best_params_full": full_best_params,
        "best_params_full_per_horizon": best_params_full_per_horizon
    }

    os.makedirs(output_dir, exist_ok=True)

    base_filename = f"{model_name}_{data_name_list[0]}_best_params.json"
    output_path = os.path.join(output_dir, base_filename)
    if os.path.exists(output_path):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            output_dir,
            f"{model_name}_{data_name_list[0]}_best_params_{timestamp}.json",
        )

    with open(output_path, "w") as f:
        json.dump(best_params_json, f, indent=2)

    best_params_json["output_file"] = output_path
    return best_params_json
