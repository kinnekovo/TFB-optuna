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
    else:
        raise NotImplementedError(f"Model {model_name} not implemented in search space.")