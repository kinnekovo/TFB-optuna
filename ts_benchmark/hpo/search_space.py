import optuna


def sample_params(model_name: str, trial: optuna.trial.Trial) -> dict:
    if model_name == "olinear.OLinear":
        return {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512, 1024]),
            "e_layers": trial.suggest_int("e_layers", 1, 3),
        }

    elif model_name in {
        "time_series_library.Autoformer",
        "time_series_library.Crossformer",
        "time_series_library.ETSformer",
        "time_series_library.FEDformer",
        "time_series_library.FiLM",
        "time_series_library.Informer",
        "time_series_library.iTransformer",
        "time_series_library.Koopa",
        "time_series_library.LightTS",
        "time_series_library.MICN",
        "time_series_library.Nonstationary_Transformer",
        "time_series_library.PatchTST",
        "time_series_library.Pyraformer",
        "time_series_library.Reformer",
        "time_series_library.TimesNet",
        "time_series_library.Transformer",
        "time_series_library.Triformer",
    }:
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512, 1024]),
            "e_layers": trial.suggest_int("e_layers", 1, 4),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        }

    elif model_name in {
        "time_series_library.DLinear",
        "time_series_library.Linear",
        "time_series_library.NLinear",
        "darts.DLinearModel",
        "darts.NLinearModel",
    }:
        return {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "enc_in": trial.suggest_categorical("enc_in", [1, 7, 21, 321]),
        }

    elif model_name in {
        "darts.NBEATSModel",
        "darts.RNNModel",
        "darts.TCNModel",
        "darts.BlockRNNModel",
        "darts.NHiTSModel",
    }:
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
            "n_rnn_layers": trial.suggest_int("n_rnn_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        }

    elif model_name in {
        "darts.TransformerModel",
        "darts.TFTModel",
        "darts.TiDEModel",
    }:
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512, 1024]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        }

    elif model_name in {
        "darts.RegressionModel",
        "darts.LinearRegressionModel",
        "darts.LightGBMModel",
        "darts.CatBoostModel",
        "darts.XGBModel",
        "darts.RandomForest",
    }:
        return {
            "lags": trial.suggest_categorical("lags", [12, 24, 48, 96]),
            "output_chunk_length": trial.suggest_categorical("output_chunk_length", [12, 24, 48]),
        }

    elif model_name in {
        "self_impl.VAR_model",
        "darts.ARIMA",
        "darts.AutoARIMA",
        "darts.StatsForecastAutoARIMA",
        "darts.ExponentialSmoothing",
        "darts.StatsForecastAutoETS",
        "darts.StatsForecastAutoCES",
        "darts.StatsForecastAutoTheta",
        "darts.FourTheta",
        "darts.FFT",
        "darts.KalmanForecaster",
        "darts.Croston",
        "darts.VARIMA",
        "darts.NaiveDrift",
        "darts.NaiveMean",
        "darts.NaiveSeasonal",
        "darts.NaiveMovingAverage",
    }:
        return {
            "lags": trial.suggest_categorical("lags", [12, 24, 48, 96]),
        }

    # TSBenchmark 额外 baselines（当前仓库内）
    elif model_name in {
        "amplifier.Amplifier",
        "cmos.CMoS",
        "crosslinear.CrossLinear",
        "dtaf.DTAF",
        "duet.DUET",
        "fits.FITS",
        "hdmixer.HDMixer",
        "moderntcn.ModernTCN",
        "patchmlp.PatchMLP",
        "pathformer.Pathformer",
        "pdf.PDF",
        "sparsetsf.SparseTSF",
        "srsnet.SRSNet",
        "timebase.TimeBase",
        "timebridge.TimeBridge",
        "timefilter.TimeFilter",
        "timekan.TimeKAN",
        "timeperceiver.TimePerceiver",
        "xpatch.xPatch",
    }:
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512, 1024]),
            "e_layers": trial.suggest_int("e_layers", 1, 4),
        }

    else:
        raise NotImplementedError(f"Model {model_name} not implemented in search space.")
