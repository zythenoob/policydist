from pathlib import Path

import pandas as pd

from ablator import Optim, PlotAnalysis, Results
from distributed import PDParallelConfig


def get_best(x: pd.DataFrame):
    return x.sort_values("val_reward", na_position="first").iloc[-1]


def test_analysis(results, tmp_path: Path):
    df = results.data
    df = (
        df.groupby("path")
        .apply(lambda x: get_best(x))
        .reset_index(drop=True)
    )
    # categorical_name_remap = {
    #     "model_config.activation": "Activation",
    #     "model_config.initialization": "Weight Init.",
    #     "train_config.optimizer_config.name": "Optimizer",
    #     "model_config.mask_type": "Mask Type",
    #     "train_config.cat_nan_policy": "Policy for Cat. Missing",
    #     "train_config.normalization": "Dataset Normalization",
    # }
    numerical_name_remap = {
        "train_config.train_iter": "Optim. Iter",
        "model_config.replay_size": "Replay Size",
        "model_config.recent_replay_ratio": "Recent Replay Ratio",
        "model_config.sup_decay": "Surprise Decay",
        "model_config.threshold": "Magnitude T",
        "model_config.direction_threshold": "Direction T",
    }
    analysis = PlotAnalysis(
        df,
        save_dir=tmp_path.as_posix(),
        cache=True,
        optim_metrics={"val_reward": Optim.max},
        numerical_attributes=list(numerical_name_remap.keys()),
        categorical_attributes=results.categorical_attributes,
    )
    attribute_name_remap = {**numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_reward": "Reward",
        },
        attribute_name_remap=attribute_name_remap,
    )
    # assert all(
    #     tmp_path.joinpath("violinplot", "val_acc", f"{file_name}.png").exists()
    #     for file_name in categorical_name_remap
    # )

    assert all(
        tmp_path.joinpath("linearplot", "val_reward", f"{file_name}.png").exists()
        for file_name in numerical_name_remap
    )
    pass


if __name__ == "__main__":
    res_path = '/home/ji/experiments/tune_walker_spd/experiment_7fc8_909a'
    res = Results(PDParallelConfig, res_path)
    import shutil
    tmp_path = Path("/home/ji/experiments/analysis/hparams")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True, parents=True)
    print(res.data)
    test_analysis(res, tmp_path)