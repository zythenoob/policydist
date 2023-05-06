from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

from ablator import Optim, PlotAnalysis, Results
from distributed import PDParallelConfig
from analysis import fig2img


def get_best(x: pd.DataFrame):
    # return x.sort_values(metric, na_position="first").iloc[-1]
    return x.iloc[-1]


def pareto_front(x, y, x_label, y_label, optim_max=True):
    '''Pareto frontier selection process'''
    # https://sirinnes.wordpress.com/2013/04/25/pareto-frontier-graphic-via-python/
    sorted_list = sorted([[x[i], y[i]] for i in range(len(x))], reverse=optim_max)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if optim_max:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    
    '''Plotting process'''
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    ax.plot(pf_X, pf_Y)
    ax.set(xlabel=x_label, ylabel=y_label)
    fig.tight_layout(pad=0.4)
    img = fig2img(fig)
    plt.close()
    return img


def make_hparam_analysis(df, numerical_name_remap):
    # drop outlier
    df = df[df[metric] < 4000]
    perf = df[metric].tolist()
    for k, name in numerical_name_remap.items():
        x = df[k].tolist()
        img = pareto_front(x, perf, name, 'Avg. Reward')
        img_path = tmp_path.joinpath(f"{k}.png")
        img.save(img_path)
        assert 0


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
    best_df = df.sort_values(metric, na_position="first", ascending=False)
    numerical_name_remap = {
        "train_config.train_iter": "Optim. Iter",
        # "model_config.replay_size": "Replay Size",
        "model_config.recent_replay_ratio": "Recent Replay Ratio",
        # "model_config.sup_decay": "Surprise Decay",
        "model_config.threshold": "Magnitude T",
        "model_config.direction_threshold": "Direction T",
        "model_config.sample_surprise_count": "Sample Steps",
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
    best_df['updates_per_ep'] = best_df['buf_update'] / best_df['current_epoch']
    best_df = best_df[list(numerical_name_remap.keys()) + ['avg_val_reward', 'val_reward', 'updates_per_ep', 'path']]
    best_df.to_csv(tmp_path.joinpath('result.csv').as_posix())
    print(best_df.to_markdown())
    with open(tmp_path.joinpath("best.md").as_posix(), 'w') as f:
        f.write(best_df.to_markdown())

    # pareto front
    # make_hparam_analysis(df, numerical_name_remap)


if __name__ == "__main__":
    metric = "avg_val_reward"
    res_path = '/home/ji/experiments/tune_walker_spdb/experiment_d1f4_68d1'
    # res_path = '/home/ji/experiments/tune_walker_spd/experiment_d1f4_68d1'
    res = Results(PDParallelConfig, res_path)
    import shutil
    tmp_path = Path("/home/ji/experiments/analysis/hparams")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True, parents=True)
    test_analysis(res, tmp_path)