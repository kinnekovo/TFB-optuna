import argparse
import json
import logging
import os
import sys
from typing import Optional

from ts_benchmark.hpo import run_optuna_search

# 确保可以导入 ts_benchmark 包
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))



def _build_auto_save_path(data_name_list, model_name: str) -> str:
    primary_series = (data_name_list[0] if data_name_list else "unknown_series").replace(".csv", "")
    model_short_name = model_name.split(".")[-1] if model_name else "unknown_model"
    return os.path.join("hpo", primary_series, model_short_name)

def _parse_forecast_lengths(s: Optional[str]):
    """Parse a string like '1,24,168' or '1 24 168' into List[int]. Returns None if s is None."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    # accept comma or whitespace separators
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("forecast-lengths must be comma-or-space-separated integers, e.g. 1,24,168")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimisation using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Relative path to the config JSON under the config directory.",
    )
    parser.add_argument(
        "--data-name-list",
        type=str,
        nargs="+",
        required=True,
        help="One or more series names on which to perform HPO.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Fully qualified model name to optimise (e.g. 'olinear.OLinear').",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help=(
            "Relative path under the result directory to store HPO outputs. "
            "If omitted, auto-generate: 'result/hpo/<series>/<model_class>/'."
        ),
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional adapter name to wrap the model during evaluation.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--forecast-lengths",
        type=_parse_forecast_lengths,
        default=None,
        help="Comma-or-space-separated list of forecast lengths, e.g. 1,24,168. If omitted, read from config.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("optuna").setLevel(logging.WARNING)

    save_path = args.save_path or _build_auto_save_path(args.data_name_list, args.model_name)

    result = run_optuna_search(
        config_path=args.config_path,
        data_name_list=args.data_name_list,
        model_name=args.model_name,
        save_path=save_path,
        n_trials=args.n_trials,
        seed=args.seed,
        forecast_lengths=args.forecast_lengths,
        adapter=args.adapter,
    )

    logging.info("HPO results will be saved under: %s", save_path)

    # Pretty-print the best trial results to stdout.
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()