from pathlib import Path

import pandas as pd

from ablator import Optim, PlotAnalysis, Results
from distributed import PDParallelConfig


def get_best(x: pd.DataFrame):
    return x.sort_values(metric, na_position="first").iloc[-1]
    # return x.iloc[-1]


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
    best_df = df.sort_values(metric, na_position="first", ascending=False).iloc[:10]
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
        optim_metrics={metric: Optim.max},
        numerical_attributes=list(numerical_name_remap.keys()),
        categorical_attributes=results.categorical_attributes,
    )
    attribute_name_remap = {**numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            metric: "Reward",
        },
        attribute_name_remap=attribute_name_remap,
    )
    # assert all(
    #     tmp_path.joinpath("violinplot", "val_acc", f"{file_name}.png").exists()
    #     for file_name in categorical_name_remap
    # )
    best_df['path'] = best_df['path'].apply(lambda x: x.split('/')[-1])
    best_df['updates_per_ep'] = best_df['current_iteration'] / best_df['current_epoch']
    best_df = best_df[list(numerical_name_remap.keys()) + ['avg_val_reward', 'val_reward', 'updates_per_ep', 'path']]
    print(best_df.to_markdown())
    with open(tmp_path.joinpath("best.md").as_posix(), 'w') as f:
        f.write(best_df.to_markdown())

    assert all(
        tmp_path.joinpath("linearplot", metric, f"{file_name}.png").exists()
        for file_name in numerical_name_remap
    )
    pass


if __name__ == "__main__":
    metric = "avg_val_reward"
    # res_path = '/home/ji/experiments/tune_walker_spd/experiment_9468_eb88'
    res_path = '/home/ji/experiments/tune_walker_spd/experiment_9468_374c'
    res = Results(PDParallelConfig, res_path)
    import shutil
    tmp_path = Path("/home/ji/experiments/analysis/hparams")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True, parents=True)
    test_analysis(res, tmp_path)