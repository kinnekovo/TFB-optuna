from typing import Callable, Dict

import optuna


def _sample_tslib_transformer_like(trial: optuna.trial.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
        "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512, 1024]),
        "e_layers": trial.suggest_int("e_layers", 1, 4),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
    }


def _sample_linear_like(trial: optuna.trial.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "enc_in": trial.suggest_categorical("enc_in", [1, 7, 21, 321]),
    }


def _sample_rnn_like(trial: optuna.trial.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "n_rnn_layers": trial.suggest_int("n_rnn_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }


def _sample_tree_like(trial: optuna.trial.Trial) -> dict:
    return {
        "lags": trial.suggest_categorical("lags", [12, 24, 48, 96]),
        "output_chunk_length": trial.suggest_categorical("output_chunk_length", [12, 24, 48]),
    }


def _sample_stat_like(trial: optuna.trial.Trial) -> dict:
    return {
        "lags": trial.suggest_categorical("lags", [12, 24, 48, 96]),
    }


def _sample_olinear(trial: optuna.trial.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
        "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512, 1024]),
        "e_layers": trial.suggest_int("e_layers", 1, 3),
    }


MODEL_SPACE_BUILDERS: Dict[str, Callable[[optuna.trial.Trial], dict]] = {
    # self-impl / local baselines
    "olinear.OLinear": _sample_olinear,
    "VAR_model": _sample_stat_like,
    "VAR.VAR": _sample_stat_like,

    # tslib family
    "Autoformer": _sample_tslib_transformer_like,
    "Crossformer": _sample_tslib_transformer_like,
    "DLinear": _sample_linear_like,
    "ETSformer": _sample_tslib_transformer_like,
    "FEDformer": _sample_tslib_transformer_like,
    "FiLM": _sample_tslib_transformer_like,
    "Informer": _sample_tslib_transformer_like,
    "iTransformer": _sample_tslib_transformer_like,
    "Koopa": _sample_tslib_transformer_like,
    "LightTS": _sample_tslib_transformer_like,
    "Linear": _sample_linear_like,
    "MICN": _sample_tslib_transformer_like,
    "NLinear": _sample_linear_like,
    "Nonstationary_Transformer": _sample_tslib_transformer_like,
    "PatchTST": _sample_tslib_transformer_like,
    "Pyraformer": _sample_tslib_transformer_like,
    "Reformer": _sample_tslib_transformer_like,
    "TimesNet": _sample_tslib_transformer_like,
    "Transformer": _sample_tslib_transformer_like,
    "Triformer": _sample_tslib_transformer_like,

    # darts deep / regression / classical
    "DLinearModel": _sample_linear_like,
    "NLinearModel": _sample_linear_like,
    "NBEATSModel": _sample_rnn_like,
    "RNNModel": _sample_rnn_like,
    "TCNModel": _sample_rnn_like,
    "BlockRNNModel": _sample_rnn_like,
    "NHiTSModel": _sample_rnn_like,
    "TransformerModel": _sample_tslib_transformer_like,
    "TFTModel": _sample_tslib_transformer_like,
    "TiDEModel": _sample_tslib_transformer_like,
    "RegressionModel": _sample_tree_like,
    "LinearRegressionModel": _sample_tree_like,
    "LightGBMModel": _sample_tree_like,
    "CatBoostModel": _sample_tree_like,
    "XGBModel": _sample_tree_like,
    "RandomForest": _sample_tree_like,
    "ARIMA": _sample_stat_like,
    "AutoARIMA": _sample_stat_like,
    "StatsForecastAutoARIMA": _sample_stat_like,
    "ExponentialSmoothing": _sample_stat_like,
    "StatsForecastAutoETS": _sample_stat_like,
    "StatsForecastAutoCES": _sample_stat_like,
    "StatsForecastAutoTheta": _sample_stat_like,
    "FourTheta": _sample_stat_like,
    "FFT": _sample_stat_like,
    "KalmanForecaster": _sample_stat_like,
    "Croston": _sample_stat_like,
    "VARIMA": _sample_stat_like,
    "NaiveDrift": _sample_stat_like,
    "NaiveMean": _sample_stat_like,
    "NaiveSeasonal": _sample_stat_like,
    "NaiveMovingAverage": _sample_stat_like,
}


def sample_params(model_name: str, trial: optuna.trial.Trial) -> dict:
    # direct exact match
    if model_name in MODEL_SPACE_BUILDERS:
        return MODEL_SPACE_BUILDERS[model_name](trial)

    # allow names like "module.Class" and map by tail class name
    short_name = model_name.split(".")[-1]
    if short_name in MODEL_SPACE_BUILDERS:
        return MODEL_SPACE_BUILDERS[short_name](trial)

    raise NotImplementedError(
        f"Model {model_name} not implemented in search space. "
        "Please add it to MODEL_SPACE_BUILDERS in ts_benchmark/hpo/search_space.py."
    )
