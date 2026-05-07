import optuna

def sample_params(model_name: str, trial: optuna.trial.Trial) -> dict:
    if model_name == 'olinear.OLinear':
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512, 1024]),
            "e_layers": trial.suggest_int("e_layers", 1, 3)
        }
    elif model_name == "amplifier.Amplifier":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-4, 3e-2),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "label_len": trial.suggest_categorical("label_len", [24, 48, 96]),
        }
    elif model_name == "cmos.CMoS":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "d_ff": trial.suggest_categorical("d_ff", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
        }
    elif model_name == "crosslinear.CrossLinear":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "r_dropout": trial.suggest_float("r_dropout", 0.0, 0.5),
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "d_ff": trial.suggest_categorical("d_ff", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
            "alpha": trial.suggest_categorical("alpha", [0.5, 1.0, 2.0]),
            "beta": trial.suggest_categorical("beta", [0.25, 0.5, 1.0]),
        }
    elif model_name == "dtaf.DTAF":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "d_ff": trial.suggest_categorical("d_ff", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "alpha": trial.suggest_categorical("alpha", [0.1, 0.2, 0.5]),
            "top_k": trial.suggest_categorical("top_k", [3, 5, 7]),
        }
    elif model_name == "duet.DUET":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-4, 5e-2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "fc_dropout": trial.suggest_float("fc_dropout", 0.0, 0.4),
            "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
            "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),
            "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
            "num_experts": trial.suggest_categorical("num_experts", [2, 4, 8]),
            "k": trial.suggest_categorical("k", [1, 2, 4]),
        }
    elif model_name == "fits.FITS":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "base_T": trial.suggest_categorical("base_T", [24, 48]),
            "H_order": trial.suggest_categorical("H_order", [1, 2, 3]),
        }
    elif model_name == "hdmixer.HDMixer":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),

            "d_model": trial.suggest_categorical("d_model", [16, 32, 64]),
            "d_ff": trial.suggest_categorical("d_ff", [32, 64, 128]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),

            "dropout": trial.suggest_float("dropout", 0.3, 0.9),
            "fc_dropout": trial.suggest_float("fc_dropout", 0.0, 0.5),
            "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.3),

            "patch_len": trial.suggest_categorical("patch_len", [8, 16, 32]),
            "stride": trial.suggest_categorical("stride", [4, 8, 16]),

            "lambda_": trial.suggest_loguniform("lambda_", 1e-3, 1e-1),
            "r": trial.suggest_loguniform("r", 1e-4, 1e-2),

            "deform_range": trial.suggest_categorical("deform_range", [0.1, 0.25, 0.5]),
        }

    else:
        raise NotImplementedError(f"Model {model_name} not implemented in search space.")