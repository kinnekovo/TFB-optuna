import optuna

TIME_SERIES_LIBRARY_MODELS = {
    "Autoformer",
    "Crossformer",
    "DLinear",
    "ETSformer",
    "FEDformer",
    "FiLM",
    "Informer",
    "iTransformer",
    "Koopa",
    "LightTS",
    "Linear",
    "MICN",
    "NLinear",
    "Nonstationary_Transformer",
    "PatchTST",
    "Pyraformer",
    "Reformer",
    "TimesNet",
    "Transformer",
    "Triformer",
    "TimeMixer",
}

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
    elif model_name == "patchmlp.PatchMLP":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),

            "d_model": trial.suggest_categorical("d_model", [256, 512, 1024]),
            "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048, 4096]),

            "e_layers": trial.suggest_int("e_layers", 1, 3),
            "d_layers": trial.suggest_int("d_layers", 1, 2),
            "n_heads": trial.suggest_categorical("n_heads", [4, 8]),

            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),

            "label_len": trial.suggest_categorical("label_len", [24, 48, 96]),
            "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
        }
    elif model_name == "pathformer.Pathformer":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),

            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),

            "d_model": trial.suggest_categorical("d_model", [4, 8, 16]),
            "d_ff": trial.suggest_categorical("d_ff", [32, 64, 128]),

            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "drop": trial.suggest_float("drop", 0.0, 0.3),

            "k": trial.suggest_categorical("k", [1, 2, 3]),

            "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),

            "revin": trial.suggest_categorical("revin", [0, 1]),

            "pct_start": trial.suggest_categorical("pct_start", [0.2, 0.3, 0.4]),
        }
    elif model_name == "pdf.PDF":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),

            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),

            "d_model": trial.suggest_categorical("d_model", [16, 32, 64]),
            "d_ff": trial.suggest_categorical("d_ff", [64, 128, 256]),

            "e_layers": trial.suggest_int("e_layers", 1, 3),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),

            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "fc_dropout": trial.suggest_float("fc_dropout", 0.0, 0.3),
            "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.2),

            "revin": trial.suggest_categorical("revin", [0, 1]),
            "individual": trial.suggest_categorical("individual", [False, True]),

            "pct_start": trial.suggest_categorical("pct_start", [0.2, 0.3, 0.4]),
        }
    elif model_name == "sparsetsf.SparseTSF":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
            "period_len": trial.suggest_categorical("period_len", [12, 24, 48, 96]),
        }
    elif model_name == "srsnet.SRSNet":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "d_model": trial.suggest_categorical("d_model", [256, 512, 1024]),
            "patch_len": trial.suggest_categorical("patch_len", [12, 24, 48]),
            "stride": trial.suggest_categorical("stride", [12, 24, 48]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.3),
            "alpha": trial.suggest_categorical("alpha", [0.5, 1.0, 2.0, 4.0]),
        }
    elif model_name.startswith("time_series_library."):
        tsl_model = model_name.split(".")[-1]

        common_params = {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }

        # 线性类：结构简单，重点调学习率、batch、移动平均
        if tsl_model in {"DLinear", "NLinear", "Linear"}:
            return {
                **common_params,
                "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
            }

        # Patch 类：重点调 patch_len / stride / d_model
        if tsl_model in {"PatchTST", "Crossformer", "Triformer"}:
            return {
                **common_params,
                "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
                "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),
                "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
                "e_layers": trial.suggest_int("e_layers", 1, 3),
                "patch_len": trial.suggest_categorical("patch_len", [8, 16, 32]),
                "stride": trial.suggest_categorical("stride", [4, 8, 16]),
            }

        # Transformer 类：重点调层数、宽度、注意力头
        if tsl_model in {
            "Transformer",
            "Informer",
            "Autoformer",
            "FEDformer",
            "Reformer",
            "Pyraformer",
            "Nonstationary_Transformer",
            "iTransformer",
        }:
            return {
                **common_params,
                "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
                "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),
                "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
                "e_layers": trial.suggest_int("e_layers", 1, 3),
                "d_layers": trial.suggest_int("d_layers", 1, 2),
                "factor": trial.suggest_categorical("factor", [1, 3, 5]),
                "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
            }

        # TimesNet / TimeMixer：重点调 top_k、卷积核、下采样结构
        if tsl_model in {"TimesNet", "TimeMixer"}:
            return {
                **common_params,
                "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
                "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),
                "e_layers": trial.suggest_int("e_layers", 1, 3),
                "top_k": trial.suggest_categorical("top_k", [3, 5, 7]),
                "num_kernels": trial.suggest_categorical("num_kernels", [4, 6, 8]),
                "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
                "down_sampling_layers": trial.suggest_categorical(
                    "down_sampling_layers", [1, 2, 3]
                ),
                "down_sampling_windows": trial.suggest_categorical(
                    "down_sampling_windows", [2, 3]
                ),
            }

        # MICN：重点调卷积核、moving_avg、d_model
        if tsl_model == "MICN":
            return {
                **common_params,
                "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
                "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),
                "e_layers": trial.suggest_int("e_layers", 1, 3),
                "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
            }

        # Koopa / FiLM / LightTS / ETSformer：先用通用保守空间
        if tsl_model in {"Koopa", "FiLM", "LightTS", "ETSformer"}:
            return {
                **common_params,
                "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
                "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),
                "e_layers": trial.suggest_int("e_layers", 1, 3),
            }

        raise NotImplementedError(
            f"time_series_library model {tsl_model} not implemented in search space."
        )
    elif model_name == "timebase.TimeBase":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "d_model": trial.suggest_categorical("d_model", [32, 64]),
            "patch_len": trial.suggest_categorical("patch_len", [8, 16]),
            "period_len": trial.suggest_categorical("period_len", [24, 48]),
            "basis_num": trial.suggest_categorical("basis_num", [4, 6, 8]),
            "orthogonal_weight": trial.suggest_categorical("orthogonal_weight", [0.05, 0.1, 0.16]),
        }
    elif model_name == "timebridge.TimeBridge":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "d_model": trial.suggest_categorical("d_model", [16, 32, 64]),
            "d_ff": trial.suggest_categorical("d_ff", [64, 128, 256]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
            "fc_dropout": trial.suggest_float("fc_dropout", 0.0, 0.3),
            "period": trial.suggest_categorical("period", [12, 24, 48, 96]),
            "num_p": trial.suggest_categorical("num_p", [1, 2, 4]),
            "stable_len": trial.suggest_categorical("stable_len", [2, 3, 4, 6]),
            "ia_layers": trial.suggest_int("ia_layers", 1, 3),
            "pd_layers": trial.suggest_int("pd_layers", 1, 2),
            "ca_layers": trial.suggest_int("ca_layers", 1, 3),
            "revin": trial.suggest_categorical("revin", [0, 1]),
            "individual": trial.suggest_categorical("individual", [False, True]),
        }
    elif model_name == "timefilter.TimeFilter":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 3e-3),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),

            "d_model": trial.suggest_categorical("d_model", [256, 512]),
            "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),
            "e_layers": trial.suggest_int("e_layers", 1, 3),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),

            "patch_len": trial.suggest_categorical("patch_len", [16, 24, 48, 96]),
            "alpha": trial.suggest_categorical("alpha", [0.05, 0.1, 0.2]),
            "top_p": trial.suggest_categorical("top_p", [0.3, 0.5, 0.7]),
        }
    elif model_name == "timekan.TimeKAN":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 3e-3),

            "batch_size": trial.suggest_categorical(
                "batch_size", [8, 16, 32]
            ),

            "dropout": trial.suggest_float(
                "dropout", 0.0, 0.3
            ),

            "d_model": trial.suggest_categorical(
                "d_model", [16, 32, 64]
            ),

            "d_ff": trial.suggest_categorical(
                "d_ff", [32, 64, 128]
            ),

            "e_layers": trial.suggest_int(
                "e_layers", 1, 3
            ),

            "n_heads": trial.suggest_categorical(
                "n_heads", [2, 4, 8]
            ),

            "top_k": trial.suggest_categorical(
                "top_k", [3, 5, 7]
            ),

            "num_kernels": trial.suggest_categorical(
                "num_kernels", [4, 6, 8]
            ),

            "moving_avg": trial.suggest_categorical(
                "moving_avg", [13, 25, 49]
            ),

            "down_sampling_layers": trial.suggest_categorical(
                "down_sampling_layers", [1, 2, 3]
            ),

            "down_sampling_window": trial.suggest_categorical(
                "down_sampling_window", [1, 2, 4]
            ),

            "channel_independence": trial.suggest_categorical(
                "channel_independence", [0, 1]
            ),
        }
    elif model_name == "timeperceiver.TimePerceiver":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "d_model": trial.suggest_categorical("d_model", [32, 64]),
            "d_ff": trial.suggest_categorical("d_ff", [1024, 2048]),
            "num_latents": trial.suggest_categorical("num_latents", [4, 8, 16]),
            "latent_dim": trial.suggest_categorical("latent_dim", [64, 128]),
            "latent_d_ff": trial.suggest_categorical("latent_d_ff", [128, 256]),
            "patch_len": trial.suggest_categorical("patch_len", [8, 16]),
            "stride": trial.suggest_categorical("stride", [4, 8]),
        }
    elif model_name == "xpatch.xPatch":
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "patch_len": trial.suggest_categorical("patch_len", [8, 16, 32]),
            "stride": trial.suggest_categorical("stride", [4, 8, 16]),
            "alpha": trial.suggest_float("alpha", 0.1, 0.9),
            "beta": trial.suggest_float("beta", 0.1, 0.9),
            "ma_type": trial.suggest_categorical("ma_type", ["ema", "sma"]),
            "revin": trial.suggest_categorical("revin", [0, 1]),
        }
    else:
        raise NotImplementedError(f"Model {model_name} not implemented in search space.")